import torch
import torch.nn as nn

from einops import rearrange

class Image2DResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        norm = lambda c: nn.GroupNorm(8, c)
        self.conv = nn.Sequential(
            norm(in_dim),
            nn.SiLU(True),
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1),
            norm(hidden_dim),
            nn.SiLU(True),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )

    def forward(self, x):
        # return x+self.conv(x)
        return self.conv(x)


class Volume2Plane(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__()
        d0, d1, d2, d3 = hidden_dims
        self.init_conv = nn.Conv2d(in_dim, d0, 3, 1, 1)
        self.out_conv0 = Image2DResBlock(d0, d1, d1)
        # self.out_conv1 = Image2DResBlock(d1, d2, d2)
        # self.out_conv2 = Image2DResBlock(d2, d3, d3)
        self.final_out = nn.Sequential(
            nn.GroupNorm(8, d3),
            nn.SiLU(True),
            nn.Conv2d(d3, out_dim, 3, 1, 1)
        )

    def forward(self, x):

        x = self.init_conv(x)
        x = self.out_conv0(x)
        # x = self.out_conv1(x)
        # x = self.out_conv2(x)
        x = self.final_out(x)
        return x
    
class Volume2XYZPlanesNetwork(nn.Module):
    def __init__(self, volume_dim, plane_dim, hidden_dim) -> None:
        super().__init__()
        # self.plane0 = Volume2Plane(volume_dim, plane_dim, hidden_dim)
        # self.plane1 = Volume2Plane(volume_dim, plane_dim, hidden_dim)
        # self.plane2 = Volume2Plane(volume_dim, plane_dim, hidden_dim)
        self.plane0 = nn.Conv2d(volume_dim, plane_dim, 3, 1, 1)
        self.plane1 = nn.Conv2d(volume_dim, plane_dim, 3, 1, 1)
        self.plane2 = nn.Conv2d(volume_dim, plane_dim, 3, 1, 1)

    def forward(self, x):
        #  x [B Np V V (V C)]
        x = rearrange(x, "B Np Va Vb VC -> B Np VC Va Vb")
        out0 = self.plane0(x[:, 0])
        out1 = self.plane1(x[:, 1])
        out2 = self.plane2(x[:, 2])
        
        out = torch.stack([out0, out1, out2], dim=1)  # B Np c Va Vb
        # out = rearrange(out, "B Np c Va Vb -> B Np Va Vb c")
        
        return out

class SpatialUp3DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.res_conv = nn.Conv3d(in_dim, in_dim, kernel_size=3, padding=1)  # 16
        self.norm = norm_act(in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x):
        x = x + self.res_conv(x)
        return self.conv(self.silu(self.norm(x)))

class Spatial3DBlock(nn.Module):
    def __init__(self, x_in_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.conv0 = nn.Conv3d(x_in_dim, x_in_dim, 1, 1)  # 16
        self.bn = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_in_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        return self.conv(self.silu(self.bn(x)))

# class SpatialCost3DNet(nn.Module):
#     def __init__(self, input_dim=128, dims=(32, 64, 128, 256)):
#         super().__init__()
#         d0, d1, d2, d3 = dims

#         self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
#         self.conv0 = Spatial3DBlock(d0,  d0, stride=1)

#         self.conv1 = Spatial3DBlock(d0,  d1,  stride=2)
#         self.conv2_0 = Spatial3DBlock(d1,  d1,  stride=1)
#         self.conv2_1 = Spatial3DBlock(d1,  d1, stride=1)

#         self.conv3 = Spatial3DBlock(d1,  d2, stride=2)
#         self.conv4_0 = Spatial3DBlock(d2,  d2, stride=1)
#         self.conv4_1 = Spatial3DBlock(d2,  d2, stride=1)

#         self.conv5 = Spatial3DBlock(d2,  d3,  stride=2)
#         self.conv6_0 = Spatial3DBlock(d3,  d3, stride=1)
#         self.conv6_1 = Spatial3DBlock(d3,  d3, stride=1)

#         self.conv7 = SpatialUp3DBlock(d3,  d2)
#         self.conv8 = SpatialUp3DBlock(d2,  d1)
#         self.conv9 = SpatialUp3DBlock(d1,  d0)

#     def forward(self, x):
#         B, C, D, H, W = x.shape

#         x = self.init_conv(x)
#         conv0 = self.conv0(x)

#         x = self.conv1(conv0)
#         x = self.conv2_0(x)
#         conv2 = self.conv2_1(x)

#         x = self.conv3(conv2)
#         x = self.conv4_0(x)
#         conv4 = self.conv4_1(x)

#         x = self.conv5(conv4)
#         x = self.conv6_0(x)
#         x = self.conv6_1(x)

#         x = conv4 + self.conv7(x)
#         x = conv2 + self.conv8(x)
#         x = conv0 + self.conv9(x)
#         return x

class SpatialCost3DNet(nn.Module):
    def __init__(self, input_dim=128, dims=(32, 64)):
        super().__init__()
        d0, d1= dims

        self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
        self.conv0 = Spatial3DBlock(d0, d0, stride=1)

        self.conv1 = Spatial3DBlock(d0, d1, stride=2)
        self.conv2_0 = Spatial3DBlock(d1, d1, stride=1)
        self.conv2_1 = Spatial3DBlock(d1, d1, stride=1)

        self.conv9 = SpatialUp3DBlock(d1, d0)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.init_conv(x)
        conv0 = self.conv0(x)

        x = self.conv1(conv0)
        x = self.conv2_0(x)
        x = self.conv2_1(x)

        x = conv0 + self.conv9(x)
        return x