
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import pymeshlab as ml
import trimesh
import lpips
import imageio
from einops import rearrange
from functools import partial
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lrm.utils.ops import scale_tensor, chunk_batch
from lrm.utils.base import BaseModule, find_class, parse_structured
from lrm.utils.misc import get_device
from lrm.utils.typing import *
from lrm.models.volumers.volumenet import SpatialVolumeNet, extract_feat_tokens_by_triplaneIndex, get_mask_from_index
from lrm.models.isosurface import MarchingCubeHelper


lpips_loss = lpips.LPIPS(net="vgg").to(get_device())
lpips_loss.eval()
for param in lpips_loss.parameters():
    param.requires_grad = False


class MLRM_system(LightningModule):
    @dataclass
    class Config():
        save_dir: str = ""
        weights: Optional[str] = None

        camera_mlp_cls: str = ""
        camera_mlp: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        cost_volumer_cls: str = ""
        cost_volumer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        decoder_cls: str = ""
        decoder: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        extract_features_only: bool = False
        features_save_dir: str = ""
        export_mesh: bool = False
        threshold: float = 25.0

        image_tokenizer_adapter_dim: int = 0

        apply_cost_volume: bool = False
        apply_sparse_attention: bool = False
        use_attn_mask: bool = False

        lambda_mse: float = 1.0
        lambda_lpips: float = 1.0
        lambda_mask: float = 0.1

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)

        self.configure()

    def from_pretrained(self):
        if self.cfg.weights is None:
            return
        state_dict = torch.load(self.cfg.weights, map_location="cpu")
        self.load_state_dict(state_dict)

    def configure(self):
        self.camera_mlp = find_class(self.cfg.camera_mlp_cls)(self.cfg.camera_mlp)
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.renderer.set_chunk_size(262144)
        self.cost_volumer = find_class(self.cfg.cost_volumer_cls)(self.cfg.cost_volumer)

        self.isosurface_helper = None

        if self.cfg.image_tokenizer_adapter_dim > 0:
            self.adapter = torch.nn.Linear(768, self.cfg.image_tokenizer_adapter_dim)

    def forward(self, batch):
        bs, N = batch["rgb_cond"].shape[:2]
        rgb_cond = batch["rgb_cond"]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W"),
            modulation_cond=self.camera_mlp(batch["c2w_cond"].reshape(bs, N, -1)),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=N
        )

        if self.cfg.image_tokenizer_adapter_dim > 0:
            input_image_tokens = self.adapter(input_image_tokens)

        # remove cls token
        B, Nv, H, W, _ = batch["rgb_cond"].shape
        dino_feats = rearrange(input_image_tokens, "B (Nv Nt) C -> (B Nv) Nt C", B=B, Nv=Nv).contiguous()
        dino_feats = dino_feats[:, 1:]
        dino_feats = rearrange(dino_feats, "(B Nv) (H W) C -> B Nv C H W", B=B, Nv=Nv, H=H//14, W=W//14).contiguous()
        
        target_pose = batch["w2c_cond"][..., :3, :4] #.view(-1, 3, 4)
        target_K = batch["intrinsic_cond"] * torch.tensor([1, -1, -1]).to(batch["intrinsic_cond"]) #.view(-1, 3, 3)

        dino_tokens = None
        if self.cfg.apply_cost_volume:
            dino_tokens = self.cost_volumer.construct_spatial_volume(
                dino_feats, target_pose, target_K
            )

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(bs, dino_tokens)

        # for sparse attention
        if self.cfg.apply_sparse_attention:
            
            encoder_hidden_states = rearrange(dino_feats, "B Nv C H W -> B (Nv H W) C")

            kv_index, _ = self.cost_volumer.prepare_sparse_token_index(dino_feats, target_pose, target_K)
            dino_shape = torch.Tensor([Nv, H//14, W//14]).long()
            attn_mask_bool = None

        else:
            encoder_hidden_states = input_image_tokens

            kv_index = None
            dino_shape = None
            attn_mask_bool = None
        
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=encoder_hidden_states,
            modulation_cond=None,
            kv_index=kv_index,
            dino_shape=dino_shape,
            encoder_attention_mask=attn_mask_bool
        )

        triplanes = self.post_processor(self.tokenizer.detokenize(tokens))
        return triplanes
    
    def predict_single(self, batch, refine_mesh=False):
        triplanes = self(batch)
        comp_rgb, _ = self.renderer(self.decoder, triplanes, batch["rays_o"], batch["rays_d"])
        meshes = self.extract_mesh(triplanes, True, 320, self.cfg.threshold, refine_mesh)
        return comp_rgb, meshes[0]

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, triplanes, has_vertex_color, resolution: int = 256, threshold: float = 25.0, refine_mesh=False):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for triplane in triplanes:
            with torch.no_grad():
                query_call = partial(self.renderer.query_triplane, decoder=self.decoder, triplane=triplane)
                density = chunk_batch(
                    query_call,
                    2097152,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(triplanes.device),
                        self.isosurface_helper.points_range,
                        (-self.cfg.renderer.radius, self.cfg.renderer.radius),
                    ),
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if has_vertex_color:
                with torch.no_grad():
                    color = query_call(
                        v_pos,
                    )["features"]
                    color = torch.sigmoid(color)

            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
            )

            if refine_mesh:
                pts, _ = trimesh.sample.sample_surface(mesh, 20000)

                mesh = ml.MeshSet()
                mesh.add_mesh(ml.Mesh(vertex_matrix=pts))
                mesh.apply_filter('compute_normal_for_point_clouds', k=10, smoothiter=3)
                mesh.apply_filter('generate_surface_reconstruction_screened_poisson', depth=8)

            meshes.append(mesh)
        return meshes

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=4e-4,
            betas=(0.9, 0.95),
            weight_decay=0.05
        )

        linear_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=1
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=7000,
            eta_min=0.0
        )

        sequential_scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[1]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sequential_scheduler,
                "interval": "step",
            }
        }

    def training_step(self, batch, batch_idx):
        triplanes = self(batch)
        comp_rgb, mask = self.renderer(self.decoder, triplanes, batch["rays_o"], batch["rays_d"])
        gt_rgb = batch["rgb"]
        gt_mask = batch["mask"]

        loss_mse = F.mse_loss(comp_rgb, gt_rgb) * self.cfg.lambda_mse
        loss_lpips = lpips_loss(
            scale_tensor(rearrange(comp_rgb, "B N H W C -> (B N) C H W"), (0, 1), (-1, 1)),
            scale_tensor(rearrange(gt_rgb, "B N H W C -> (B N) C H W"), (0, 1), (-1, 1))
        ) * self.cfg.lambda_lpips

        EPS = 1e-4

        flatten_mask = mask.view(-1).clamp(min=EPS, max=1-EPS)
        flatten_gt = gt_mask.view(-1)

        log_term = flatten_gt * torch.log(flatten_mask)
        neg_log_term = (1 - flatten_gt) * torch.log(1 - flatten_mask)

        loss_mask = -(log_term + neg_log_term).mean() * self.cfg.lambda_mask

        loss_mse = loss_mse.mean()
        loss_lpips = loss_lpips.mean()
        loss_mask = loss_mask.mean()

        loss = loss_mse + loss_lpips + loss_mask
        self.log("train/loss", loss.mean())
        self.log("train/loss_mse", loss_mse.mean())
        self.log("train/loss_lpips", loss_lpips.mean())
        self.log("train/loss_mask", loss_mask.mean())
        
        return loss

    def validation_step(self, batch, batch_idx):
        triplanes = self(batch)
        comp_rgb, mask = self.renderer(self.decoder, triplanes, batch["rays_o"], batch["rays_d"])
        gt_rgb = batch["rgb"]

        batch_size = comp_rgb.shape[0]

        num_views = comp_rgb.shape[1]

        for b in range(batch_size):
            for v in range(num_views):
                comp_rgb_single = comp_rgb[b, v]
                gt_rgb_single = gt_rgb[b, v]

                # save images
                comp_rgb_single = comp_rgb_single.cpu().numpy()
                gt_rgb_single = gt_rgb_single.cpu().numpy()

                comp_rgb_single = np.clip(comp_rgb_single, 0, 1)
                gt_rgb_single = np.clip(gt_rgb_single, 0, 1)

                comp_rgb_single = (comp_rgb_single * 255).astype(np.uint8)
                gt_rgb_single = (gt_rgb_single * 255).astype(np.uint8)

                imageio.imwrite(
                    os.path.join(
                        self.cfg.save_dir,
                        f"{batch['scene_id'][b]}_{v}_step{self.global_step}.png",
                    ),
                    comp_rgb_single,
                )

                imageio.imwrite(
                    os.path.join(
                        self.cfg.save_dir,
                        f"{batch['scene_id'][b]}_{v}_gt.png",
                    ),
                    gt_rgb_single,
                )

        comp_rgb = rearrange(comp_rgb, "B N H W C -> (B N) C H W")
        gt_rgb = rearrange(gt_rgb, "B N H W C -> (B N) C H W")

        psnr = -10 * torch.log10(F.mse_loss(comp_rgb, gt_rgb))
        self.log("val/psnr", psnr)
        lpips = lpips_loss(
            scale_tensor(comp_rgb, (0, 1), (-1, 1)),
            scale_tensor(gt_rgb, (0, 1), (-1, 1))
        ).mean()
        self.log("val/lpips", lpips)
    
    def test_step(self, batch, batch_idx):
        if not os.path.exists(self.cfg.save_dir):
            os.makedirs(self.cfg.save_dir)

        triplanes = self(batch)
        comp_rgb, mask = self.renderer(self.decoder, triplanes, batch["rays_o"], batch["rays_d"])
        gt_rgb = batch["rgb"]

        batch_size = comp_rgb.shape[0]
        num_views = comp_rgb.shape[1]

        for b in range(batch_size):
            for v in range(num_views):
                comp_rgb_single = comp_rgb[b, v]
                gt_rgb_single = gt_rgb[b, v]

                # save images
                comp_rgb_single = comp_rgb_single.cpu().numpy()
                gt_rgb_single = gt_rgb_single.cpu().numpy()

                comp_rgb_single = np.clip(comp_rgb_single, 0, 1)
                gt_rgb_single = np.clip(gt_rgb_single, 0, 1)

                comp_rgb_single = (comp_rgb_single * 255).astype(np.uint8)
                gt_rgb_single = (gt_rgb_single * 255).astype(np.uint8)

                imageio.imwrite(
                    os.path.join(
                        self.cfg.save_dir,
                        f"test_{batch['scene_id'][b]}_{v}_step{self.global_step}.png",
                    ),
                    comp_rgb_single,
                )

                imageio.imwrite(
                    os.path.join(
                        self.cfg.save_dir,
                        f"test_{batch['scene_id'][b]}_{v}_gt.png",
                    ),
                    gt_rgb_single,
                )
