
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from functools import partial

from lrm.utils.ops import scale_tensor, chunk_batch
from lrm.utils.base import BaseModule, find_class
from lrm.utils.typing import *
from lrm.models.volumers.volumenet import SpatialVolumeNet, extract_feat_tokens_by_triplaneIndex, get_mask_from_index
from lrm.models.isosurface import MarchingCubeHelper


class MLRM_system(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
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

    cfg: Config

    def from_pretrained(self):
        state_dict = torch.load(self.cfg.weights, map_location="cpu")
        self.load_state_dict(state_dict)

    def configure(self):
        super().configure()
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
    
    def predict_single(self, batch):
        triplanes = self(batch)
        comp_rgb = self.renderer(self.decoder, triplanes, batch["rays_o"], batch["rays_d"])
        meshes = self.extract_mesh(triplanes, True, 320, self.cfg.threshold)
        return comp_rgb, meshes[0]

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, triplanes, has_vertex_color, resolution: int = 256, threshold: float = 25.0):
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

            meshes.append(mesh)
        return meshes
