from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

import lrm
from lrm.utils.base import BaseModule
from lrm.utils.ops import (
    rays_intersect_bbox, 
    chunk_batch, 
    get_activation, 
    scale_tensor,
)
from lrm.utils.typing import *


class TriplaneNeRFRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 0.5
        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0
        color_activation: str = "sigmoid"
        num_samples_per_ray: int = 128 
        randomized: bool = False

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        assert self.cfg.feature_reduction in ["concat", "mean"]
        self.chunk_size = 262144

    def set_chunk_size(self, chunk_size: int):
        assert (
            chunk_size >= 0
        ), "chunk_size must be a non-negative integer (0 for no chunking)."
        self.chunk_size = chunk_size

    def query_triplane(
        self,
        positions: torch.Tensor,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)

        positions = scale_tensor(
            positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
        )

        def _query_chunk(x):
            indices2D: torch.Tensor = torch.stack(
                (x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]),
                dim=-3,
            )
            out: torch.Tensor = F.grid_sample(
                rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
                rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
                align_corners=False,
                mode="bilinear",
            )
            if self.cfg.feature_reduction == "concat":
                out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            elif self.cfg.feature_reduction == "mean":
                out = reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")
            else:
                raise NotImplementedError

            net_out: Dict[str, torch.Tensor] = decoder(out)
            return net_out

        if self.chunk_size > 0:
            net_out = chunk_batch(_query_chunk, self.chunk_size, positions)
        else:
            net_out = _query_chunk(positions)

        net_out["density_act"] = get_activation(self.cfg.density_activation)(
            net_out["density"] + self.cfg.density_bias
        )
        net_out["color"] = get_activation(self.cfg.color_activation)(
            net_out["features"]
        )

        net_out = {k: v.view(*input_shape, -1) for k, v in net_out.items()}

        return net_out

    def _forward(
        self,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        **kwargs,
    ):
        rays_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        n_rays = rays_o.shape[0]

        t_near, t_far, rays_valid = rays_intersect_bbox(rays_o, rays_d, self.cfg.radius)
        rays_o, rays_d = rays_o[rays_valid], rays_d[rays_valid]
        t_near, t_far = t_near[rays_valid], t_far[rays_valid]

        t_vals = torch.linspace(
            0, 1, self.cfg.num_samples_per_ray + 1, device=triplane.device
        )
        t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
        z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]  # (N_rays, N_samples)

        xyz = (
            rays_o[:, None, :] + z_vals[..., None] * rays_d[..., None, :]
        )  # (N_rays, N_sample, 3)

        mlp_out = self.query_triplane(
            positions=xyz,
            decoder=decoder,
            triplane=triplane,
        )

        eps = 1e-10
        # deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
        deltas = t_vals[1:] - t_vals[:-1]  # (N_rays, N_samples)
        alpha = 1 - torch.exp(
            -deltas * mlp_out["density_act"][..., 0]
        )  # (N_rays, N_samples)
        accum_prod = torch.cat(
            [
                torch.ones_like(alpha[:, :1]),
                torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1),
            ],
            dim=-1,
        )
        weights = alpha * accum_prod  # (N_rays, N_samples)
        comp_rgb_ = (weights[..., None] * mlp_out["color"]).sum(dim=-2)  # (N_rays, 3)
        opacity_ = weights.sum(dim=-1)  # (N_rays)

        comp_rgb = torch.zeros(
            n_rays, 3, dtype=comp_rgb_.dtype, device=comp_rgb_.device
        )
        opacity = torch.zeros(n_rays, dtype=opacity_.dtype, device=opacity_.device)
        comp_rgb[rays_valid] = comp_rgb_
        opacity[rays_valid] = opacity_

        comp_rgb += 1 - opacity[..., None]
        comp_rgb = comp_rgb.view(*rays_shape, 3)

        return comp_rgb, opacity

    def forward(
        self,
        decoder: torch.nn.Module,
        triplanes: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        chunk_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        batch_size = triplanes.shape[0]
        num_views = rays_o.shape[1]
        batch_rgbs = []
        batch_opacities = []
        for b in range(batch_size):
            comp_rgbs = []
            opacities = []
            for c in range(0, num_views, chunk_size):
                out, opacity = self._forward(
                    decoder,
                    triplanes[b],
                    rays_o[b, c : c + chunk_size],
                    rays_d[b, c : c + chunk_size]
                )
                comp_rgbs.append(out)
                opacities.append(opacity)

            comp_rgbs = torch.cat(comp_rgbs, dim=0)
            opacities = torch.cat(opacities, dim=0)
            batch_rgbs.append(comp_rgbs)
            batch_opacities.append(opacities)

        return torch.stack(batch_rgbs, dim=0), torch.stack(batch_opacities, dim=0)

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()