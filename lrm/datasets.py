import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from lrm.utils.ops import *
from lrm.utils.typing import *

class DatasetConfig():
    data_dir: str = "data_preparation/data_example"  # path to your data
    scenes_json: str = "data/lvis.json"
    background_color: List[float] = [1.0, 1.0, 1.0]

    cache_dino_feats: bool = False
    dino_feats_dir: str = ""

    num_views_total: int = 24
    num_views_condition: int = 6
    num_views_supervise: int = 4
    
    crop_spare_pixels: int = 0
    img_size: int = 128
    rand_min_size: Optional[int] = None
    rand_max_size: int = 128
    cond_size: int = 504
    eval_size: int = 256

    perturb_c2w: bool = True
    perturb_c2w_prob: float = 0.5
    perturb_c2w_scale_ele: float = 10.0
    perturb_c2w_scale_azi: float = 10.0
    perturb_c2w_scale_dist: float = 0.1

    grid_distort: bool = True
    grid_distort_prob: float = 0.5
    grid_distort_strength: float = 0.5

    train_indices: Optional[Tuple[int, int]] = [0, 31000]
    val_indices: Optional[Tuple[int, int]] = [31000, 31500]
    test_indices: Optional[Tuple[int, int]] = [31500, 31550]


class ObjaverseDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg = DatasetConfig()
        with open(self.cfg.scenes_json) as f:
            self.all_scenes = json.loads(f.read())
        
        if split == "train":
            indices = self.cfg.train_indices
        elif split == "val":
            indices = self.cfg.val_indices
        else:
            indices = self.cfg.test_indices
        
        if indices is not None:
            assert len(indices) == 2
            self.all_scenes = self.all_scenes[indices[0]:indices[1]]

        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.split = split
        self.cfg = DatasetConfig()

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, index):
        scene_index = index

        cond_ids = np.random.choice(
            self.cfg.num_views_total, self.cfg.num_views_condition, replace=False
        )

        sup_ids = np.random.choice(
            self.cfg.num_views_total, self.cfg.num_views_supervise, replace=False
        )

        scene_dir = os.path.join(self.cfg.data_dir, self.all_scenes[scene_index])
        scene_id = os.path.basename(scene_dir)
        with open(os.path.join(scene_dir, "meta.json")) as f:
            meta = json.loads(f.read())

        data_cond, data_sup = [], []
        data_out = {}

        cano_c2w, ref_w2c, canonical_c2w = None, None, None

        for i, view_index in enumerate(cond_ids):
            crop_size = self.cfg.cond_size
            resize_size = self.cfg.cond_size

            frame_info = meta["locations"][view_index]
            img_path = os.path.join(scene_dir, frame_info["frames"][0]["name"])
            img = Image.fromarray(imageio.v2.imread(img_path))
            img = img.convert("RGBA").resize((resize_size, resize_size))
            img = np.asarray(img) / 255.0
            img = torch.from_numpy(img).float()
            mask = img[:, :, 3:]
            rgb = img[:, :, :3] * mask + self.background_color[None, None, :] * (1 - mask)

            dino_feat = None
            if self.cfg.cache_dino_feats:
                feats_file_name = frame_info["frames"][0]["name"].replace(
                    ".png", "_feats.npy"
                )
                dino_feat_path = os.path.join(self.cfg.dino_feats_dir, feats_file_name)
                dino_feat = torch.from_numpy(np.load(dino_feat_path))

            c2w = torch.Tensor(frame_info["transform_matrix"]).float()

            if self.split == "train" and self.cfg.perturb_c2w and random.random() < self.cfg.perturb_c2w_prob:
                elevation, azimuth, distance = c2w_to_spherical(c2w)

                elevation += self.cfg.perturb_c2w_scale_ele * math.pi / 180.0 * random.random()
                azimuth += self.cfg.perturb_c2w_scale_azi * math.pi / 180.0 * random.random()
                distance += self.cfg.perturb_c2w_scale_dist * random.random()
                c2w = spherical_to_c2w(elevation, azimuth, distance)

            if self.split == "train" and self.cfg.grid_distort and random.random() < self.cfg.grid_distort_prob:
                rgb = grid_distortion(rgb, strength=self.cfg.grid_distort_strength)         

            if cano_c2w is None:
                cano_c2w = c2w.clone()
                cano_w2c = torch.inverse(c2w)
                cano_distance = c2w[0:3, 3].norm()
                transform = torch.eye(4)
                transform[2, 3] = cano_distance
            c2w = transform @ cano_w2c @ c2w
            w2c = torch.inverse(c2w)

            fovy = meta["camera_angle_x"]
            intrinsic = get_intrinsic_from_fov(fovy, H=resize_size, W=resize_size)
            focal_length = 0.5 * resize_size / math.tan(0.5 * fovy)
            directions = get_ray_directions(
                H=resize_size,
                W=resize_size,
                focal=focal_length,
            )
            directions = F.normalize(directions, dim=-1)
            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

            data_cond.append(
                {
                    "view_index": torch.as_tensor(view_index),
                    "rgb": rgb,
                    "mask": mask,
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "c2w": c2w,
                    "w2c": w2c,
                    "intrinsic": intrinsic,
                    "dino_feat": dino_feat,
                }
            )

        for k in data_cond[0].keys():
            if data_cond[0][k] is not None:
                data_out[k + "_cond"] = torch.stack([d[k] for d in data_cond], dim=0)

        for i, view_index in enumerate(sup_ids):

            if self.split == "train":
                crop_size = self.cfg.img_size
                resize_size = np.random.randint(
                    self.cfg.rand_min_size or self.cfg.img_size,
                    self.cfg.rand_max_size + 1,
                )
            else:
                crop_size = self.cfg.eval_size
                resize_size = self.cfg.eval_size

            frame_info = meta["locations"][view_index]
            img_path = os.path.join(scene_dir, frame_info["frames"][0]["name"])
            img = Image.fromarray(imageio.v2.imread(img_path))
            img = img.convert("RGBA").resize((resize_size, resize_size))
            img = np.asarray(img) / 255.0
            img = torch.from_numpy(img).float()
            mask = img[:, :, 3:]
            rgb = img[:, :, :3] * mask + self.background_color[None, None, :] * (1 - mask)

            dino_feat = None

            c2w = torch.Tensor(frame_info["transform_matrix"]).float()

            if cano_c2w is None:
                cano_c2w = c2w.clone()
                cano_w2c = torch.inverse(c2w)
                cano_distance = c2w[0:3, 3].norm()
                transform = torch.eye(4)
                transform[2, 3] = cano_distance
            c2w = transform @ cano_w2c @ c2w
            w2c = torch.inverse(c2w)

            fovy = meta["camera_angle_x"]
            intrinsic = get_intrinsic_from_fov(fovy, H=resize_size, W=resize_size)
            focal_length = 0.5 * resize_size / math.tan(0.5 * fovy)
            directions = get_ray_directions(
                H=resize_size,
                W=resize_size,
                focal=focal_length
            )
            directions = F.normalize(directions, dim=-1)
            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

            if crop_size < resize_size:
                x0 = np.random.randint(0 + self.cfg.crop_spare_pixels, resize_size - crop_size + 1 - self.cfg.crop_spare_pixels)
                y0 = np.random.randint(0 + self.cfg.crop_spare_pixels, resize_size - crop_size + 1 - self.cfg.crop_spare_pixels)
                intrinsic[..., 0, 2] -= y0
                intrinsic[..., 1, 2] -= x0
                x1 = x0 + crop_size
                y1 = y0 + crop_size
                rgb = rgb[x0:x1, y0:y1]
                mask = mask[x0:x1, y0:y1]
                rays_o = rays_o[x0:x1, y0:y1]
                rays_d = rays_d[x0:x1, y0:y1]

            data_sup.append(
                {
                    "view_index": torch.as_tensor(view_index),
                    "rgb": rgb,
                    "mask": mask,
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "c2w": c2w,
                    "w2c": w2c,
                    "intrinsic": intrinsic,
                    "dino_feat": dino_feat,
                }
            )

        for k in data_sup[0].keys():
            if data_sup[0][k] is not None:
                data_out[k] = torch.stack([d[k] for d in data_sup], dim=0)

        sample = {
            **data_out,
            "background_color": self.background_color,
            "index": torch.as_tensor(scene_index),
            "scene_id": scene_id,
        }
        return sample



class ObjaverseDataModule(LightningDataModule):
    def __init__(self, batch_size=4, batch_size_eval=1, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ObjaverseDataset(split="train")
        self.val_dataset = ObjaverseDataset(split="val")
        self.test_dataset = ObjaverseDataset(split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
        )