from dataclasses import dataclass, field

import torch
import torch.nn as nn
from einops import rearrange

from lrm.utils.base import BaseModule
from lrm.utils.ops import get_activation
from lrm.utils.typing import *


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 768
        out_channels: int = 40

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
        )

    def forward(self, triplanes: torch.Tensor) -> torch.Tensor:
        triplanes_up = rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )
        return triplanes_up


class MLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        dim_in: int = 16
        dim_out: int = 768
        n_neurons: int = 768
        n_hidden_layers: int = 1
        activation: str = "relu"
        output_activation: Optional[str] = None
        bias: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()
        layers = [
            self.make_linear(
                self.cfg.dim_in, self.cfg.n_neurons, bias=self.cfg.bias
            ),
            self.make_activation(self.cfg.activation),
        ]
        for i in range(self.cfg.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.cfg.n_neurons, self.cfg.n_neurons, bias=self.cfg.bias
                ),
                self.make_activation(self.cfg.activation),
            ]
        layers += [
            self.make_linear(
                self.cfg.n_neurons, self.cfg.dim_out, bias=self.cfg.bias
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(self.cfg.output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError


class Decoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        share_mlp: dict = field(default_factory=dict)
        hidden_dim: int = 64
        activation: str = "relu"
        bias: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.mlp = MLP(self.cfg.share_mlp)

        density_mlp = [
            self.make_linear(self.cfg.share_mlp.dim_out, self.cfg.hidden_dim, bias=self.cfg.bias),
            self.make_activation(self.cfg.activation),
            self.make_linear(self.cfg.hidden_dim, 1, bias=self.cfg.bias),
        ]
        self.density_head = nn.Sequential(*density_mlp)

        features_mlp = [
            self.make_linear(self.cfg.share_mlp.dim_out, self.cfg.hidden_dim, bias=self.cfg.bias),
            self.make_activation(self.cfg.activation),
            self.make_linear(self.cfg.hidden_dim, 3, bias=self.cfg.bias),
        ]
        self.features_head = nn.Sequential(*features_mlp)

    def make_linear(self, dim_in, dim_out, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid(inplace=True)
        elif activation == "none":
            return nn.Identity(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        hidden_features = self.mlp(x)
        out = {
            "density": self.density_head(hidden_features),
            "features": self.features_head(hidden_features),
        }
        return out
