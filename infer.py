import argparse
import os
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F
import kiui
import math
import imageio
import rembg

import pytorch_lightning as pl
from omegaconf import OmegaConf

from lrm.utils import ops
from lrm.utils.base import find_class
from lrm.utils.zero123pp_utils import get_zero123plus_input_cameras, remove_background, resize_foreground
from einops import rearrange


def predict(system, batch, args, extras) -> None:
    with torch.no_grad():
        pred_rgbs, mesh = system.predict_single(batch, refine_mesh=args.refine_mesh)
    pred_rgbs = pred_rgbs.squeeze(0)
    
    # save video
    imageio.mimwrite(f"{args.output}/{batch['scene_name']}.mp4", (pred_rgbs * 255).cpu().numpy().astype(np.uint8), fps=30)
    if args.refine_mesh:
        mesh.save_current_mesh(f"{args.output}/{batch['scene_name']}_mesh.obj")
    else:
        mesh.export(f"{args.output}/{batch['scene_name']}_mesh.obj")

def get_rays(fovy, c2w, height, width):
    focal_length = 0.5 * height / math.tan(0.5 * fovy)
    directions = ops.get_ray_directions(
        H=height,
        W=width,
        focal=focal_length,
    )
    rays_o, rays_d = ops.get_rays(directions, c2w, keepdim=True)
    return rays_o, rays_d

def prepare_data(img_path, pipe, args, scene_name=""):

    no_rembg = args.no_rembg
    refined_mv_model = args.refined_mv_model
    radius = args.radius
    num_views = args.num_views

    rembg_session = None if args.no_rembg else rembg.new_session()

    img_cond = Image.open(img_path)

    if not no_rembg:
        img_cond = remove_background(img_cond, rembg_session)
        img_cond = resize_foreground(img_cond, args.resize_scale)

    mv_image = pipe(
        img_cond, 
        num_inference_steps=75, 
    ).images[0]

    mv_image = np.asarray(mv_image, dtype=np.float32) / 255.0
    mv_image = torch.from_numpy(mv_image).permute(2, 0, 1).contiguous().float()
    mv_image = rearrange(mv_image, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    mv_image = F.interpolate(mv_image, size=(504, 504), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()

    # save the image
    for i, img in enumerate(mv_image):
        img = Image.fromarray((img * 255).astype(np.uint8))
        img.save(f"{args.output}/{scene_name}_mv_image_{i}.png")
    
    fovy = np.deg2rad(30)

    intrinsic_cond = ops.get_intrinsic_from_fov(fovy, 504, 504).repeat(num_views, 1, 1)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=radius)
    c2w_cond = torch.zeros(1, num_views, 4, 4)
    c2w_cond[:, :, 3, 3] = 1
    c2w_cond[:, :, :3, :4] = input_cameras[:, :, :12].reshape(-1, num_views, 3, 4)
    c2w_cond = c2w_cond.squeeze(0)
    w2c_cond = torch.inverse(c2w_cond)
    rays_o_cond, rays_d_cond = get_rays(fovy, c2w_cond, 504, 504)

    intrinsic = ops.get_intrinsic_from_fov(fovy, 512, 512).repeat(num_views, 1, 1)
    c2w = []
    for azimuth in np.arange(0, 360, 3, dtype=np.float32):
        c2w.append(torch.from_numpy(kiui.cam.orbit_camera(10, azimuth, radius=radius, opengl=True)))
    c2w = torch.stack(c2w, dim=0).float()
    # c2w[:, :3, 1:3] *= -1
    w2c = torch.inverse(c2w)
    rays_o, rays_d = get_rays(fovy, c2w, 512, 512)

    c2w_cond, _ = ops.transform_relative_pose(c2w_cond, c2w)
    w2c_cond = torch.inverse(c2w_cond)
    w2c = torch.inverse(c2w)

    ret_dict = {
        "rgb_cond": mv_image,
        "rays_o_cond": rays_o_cond,
        "rays_d_cond": rays_d_cond,
        "w2c_cond": w2c_cond,
        "c2w_cond": c2w_cond,
        "intrinsic_cond": intrinsic_cond,
        "rays_o": rays_o,
        "rays_d": rays_d,
        "w2c": w2c,
        "c2w": c2w,
        "intrinsic": intrinsic,
        "background_color": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    }

    for key, value in ret_dict.items():
        ret_dict[key] = torch.Tensor(value).to(torch.device("cuda")).unsqueeze(0)
    
    ret_dict["scene_name"] = scene_name
    return ret_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--input", type=str)
    parser.add_argument("--no_rembg", action="store_true")
    parser.add_argument("--refined_mv_model", action="store_true")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--num_views" , type=int, default=6)
    parser.add_argument("--output", type=str, default="test_output")
    parser.add_argument("--resize_scale", type=float, default=0.85)
    parser.add_argument("--refine_mesh", action="store_true")

    args, extras = parser.parse_known_args()

    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
    )

    if args.refined_mv_model:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
        print('Loading custom white-background unet ...')
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipe.unet.load_state_dict(state_dict, strict=True)

    pipe = pipe.to(torch.device("cuda"))

    cfg = OmegaConf.load(args.config)

    pl.seed_everything(cfg.seed, workers=True)

    system = find_class(cfg.system_cls)(cfg.system)
    system.from_pretrained()
    system = system.to(torch.device("cuda"))
    system.eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if os.path.isdir(args.input):
        for img_name in os.listdir(args.input):
            scene_path = os.path.join(args.input, img_name)
            batch = prepare_data(scene_path, pipe, args, img_name.split(".")[0])
            predict(system, batch, args, extras)
    else:
        scene = os.path.basename(args.input).split(".")[0]
        batch = prepare_data(args.input, pipe, args, scene)
        predict(system, batch, args, extras)
