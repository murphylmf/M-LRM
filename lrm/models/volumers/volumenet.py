import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from lrm.models.volumers.cam_utils import get_warp_coordinates
from lrm.models.volumers.cost_networks import  SpatialCost3DNet, Volume2XYZPlanesNetwork
from lrm.utils.base import BaseModule
from lrm.utils.typing import *

from dataclasses import dataclass, field

from kornia import create_meshgrid
from kornia.morphology import dilation
from einops import rearrange, repeat

import cv2
import os
from PIL import Image
import pdb


def get_mask_from_index(image, indices):
    """
    - image: [B, NV, C, H, W]
    - indices: [B 3p V V (V N) C] 
    - [B, Tokens_NUM(3*V*V), NV, mask_per_view, 2]
    
    return:
    - mask: [B, NV, H, W]
    """
    B, NV, C, H, W = image.shape
    # B, Tokens_NUM, NV, mask_per_view, _ = indices.shape  # [B 3p V V (V N) C]
    B, Np, V, _, VNv, _ = indices.shape  # [B 3p V V (V N) C]
    Tokens_NUM = Np * V * V
    Nv = VNv // V
    mask = torch.zeros(B, Tokens_NUM, NV, W, H).to(indices.device)
    #mask = repeat(mask,'B NV H W ->B TN NV H W',TN=Tokens_NUM)
    # 将索引拆分为行和列
    indices = rearrange(indices, "B Np Va Vb (Vc Nv) C -> B (Np Va Vb) (Nv Vc) C", Np=Np, Vc=V, Nv=Nv)
    v, u, i  = indices[..., 0], indices[..., 1], indices[..., 2]  # [B 3p V V (V N)]
    # i, u ,v
    index = i * W * H + v * W + u

    # 直接更新像素值
    # mask[..., i, u, v] = True
    mask = mask.reshape(B,Tokens_NUM,-1).contiguous()
    mask = mask.scatter_(2, index, 1.0)

    mask = mask.reshape(B, Tokens_NUM * NV, H, W).contiguous()
    mask = dilation(mask, torch.ones(3, 3).to(mask))
    mask = mask.reshape(B, Tokens_NUM, -1).contiguous().bool()

    return mask


def extract_feat_tokens_by_triplaneIndex(feats, triplane_index, grid_sample_mode='bilinear'):
    """_summary_

    Args:
        feats ([B Nv C H W]): dino features, where Nv means the numver of views
        triplane_index ([B Np (Va Vb) (Vc Nv) C]): , where C denotes [u(W) v(H) viewid]
    """
    B, Nv, C, H, W = feats.shape
    assert len(triplane_index.shape) == 5
    B, Np, V2, VcNv, c = triplane_index.shape

    V = int(V2 ** 0.5)
    
    feats = rearrange(feats, "B Nv C H W -> B C Nv H W")
    
    # triplane_index = rearrange(triplane_index, "B Np Va Vb VcNv c -> B Np (Va Vb) VcNv c") # [B 3p (V V) (V ViewNum) 3c]
    # kv_index_normalized = triplane_index.clone()
    # # normalize the index into -1~1
    # kv_index_normalized[:, :, :, :, 0] = 2 * kv_index_normalized[:, :, :, :, 0] / (W - 1) - 1 # W
    # kv_index_normalized[:, :, :, :, 1] = 2 * kv_index_normalized[:, :, :, :, 1] / (H - 1) - 1 # h
    # kv_index_normalized[:, :, :, :, 2] = 2 * kv_index_normalized[:, :, :, :, 2] / (Nv - 1) - 1 # NumViews
    
    # extract sparse keys and values
    sparse_tokens = F.grid_sample(feats, triplane_index, mode=grid_sample_mode, padding_mode='border', align_corners=True)  # [B c Np (Va Vb) (Vc Nv)]
    
    sparse_tokens = rearrange(sparse_tokens, "B c Np (Va Vb) VcNv -> (B Np Va Vb) VcNv c", B=B, Np=Np, Va=V, Vb=V)
    
    return sparse_tokens # [B Np Va Vb VcNv c]

class SpatialVolumeNet(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        view_dim: int = 768
        view_num: int = 4
        input_image_size: int = 252
        spatial_volume_size: int = 32
        spatial_volume_length: float = 0.71925
        triplane_dim: int = 512
        construct_cost_volume: bool = False
        robust_sampling: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        if self.cfg.construct_cost_volume:
            # dims = (64, 128, 256, 512)
            # dims = (32, 64, 128, 256)
            dims = (64, 128)
            self.cost_network = SpatialCost3DNet(input_dim=self.cfg.view_dim * self.cfg.view_num, dims=dims)
            # self.triplane_network = Volume2XYZPlanesNetwork(
            #     volume_dim=dims[0] * spatial_volume_size, 
            #     plane_dim=triplane_dim, 
            #     hidden_dim=(512, 512, 512, 512)
            # )
            self.triplane_network = Volume2XYZPlanesNetwork(
                volume_dim=dims[0] * self.cfg.spatial_volume_size, 
                plane_dim=self.cfg.triplane_dim, 
                hidden_dim=(512, 512, 512, 512)  # not used
            )

        self.input_image_size = self.cfg.input_image_size  # its the shape of RGB color image, not dino feature
        self.spatial_volume_size = self.cfg.spatial_volume_size
        self.spatial_volume_length = self.cfg.spatial_volume_length

        self.view_dim = self.cfg.view_dim

    def construct_spatial_volume(self, x, target_poses, target_Ks, grid_sample_mode='bilinear', grid_sample_padding='zeros', apply_3DCNN=True):
        """
        @param x:            B,N,C,H,W   feature maps
        @param target_poses: N,3,4   w2c matrices
        @param target_Ks:    N,3,3  the intrinsic of the original images instead of feature maps
        @return:
        """
        B, N, C, H, W = x.shape
        V = self.spatial_volume_size
        device = x.device
        dtype = x.dtype

        spatial_volume_verts = torch.linspace(-self.spatial_volume_length, self.spatial_volume_length, V, dtype=dtype, device=device) # range [-1, 1] by default
        spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)]  # change [B C D(Z) H(Y) W(X)] to [B C X Y Z]  ？？
        spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1) # B,3,V,V,V
        
        # check the coordinate system

        # extract 2D image features
        spatial_volume_feats = []
        for ni in range(0, N):
            pose_source_ = target_poses[:, ni]
            K_source_ = target_Ks[:, ni]
            x_ = x[:, ni]
            C = x_.shape[1]

            coords_source = get_warp_coordinates(spatial_volume_verts, x_.shape[-1], self.input_image_size, K_source_, pose_source_).view(B, V, V * V, 2)
            unproj_feats_ = F.grid_sample(x_, coords_source, mode=grid_sample_mode, padding_mode=grid_sample_padding, align_corners=True)
            unproj_feats_ = unproj_feats_.view(B, C, V, V, V)
            spatial_volume_feats.append(unproj_feats_)

        spatial_volume_feats = torch.stack(spatial_volume_feats, 1) # B,N,C,V,V,V

        if apply_3DCNN:
            N = spatial_volume_feats.shape[1]
            spatial_volume_feats = spatial_volume_feats.view(B, N*C, V, V, V)

            spatial_volume_feats = self.cost_network(spatial_volume_feats)  # b,64,32,32,32  (B C Z Y X)
            
            triplane_cost = self.volume2triplane(spatial_volume_feats[:, None]) # [B 3p (V V) (V) C]
            
            triplane = self.triplane_network(rearrange(triplane_cost, "B Np (H W) V C -> B Np H W (V C)", Np=3, H=V, W=V, V=V))
            
            return triplane # [B Np C V V]
            
        return spatial_volume_feats, spatial_volume_verts

    def construct_spatial_volume_fixed_views(self, x, target_poses, target_Ks, grid_sample_mode='bilinear', grid_sample_padding='zeros', apply_3DCNN=True):
        """
        @param x:            B,N,C,H,W   feature maps
        @param target_poses: N,3,4   w2c matrices
        @param target_Ks:    N,3,3  the intrinsic of the original images instead of feature maps
        @return:
        """
        B, N, C, H, W = x.shape
        V = self.spatial_volume_size
        device = x.device
        dtype = x.dtype

        spatial_volume_verts = torch.linspace(-self.spatial_volume_length, self.spatial_volume_length, V, dtype=dtype, device=device) # range [-1, 1] by default
        spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)]  # change [B C D(Z) H(Y) W(X)] to [B C X Y Z]  ？？
        spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1) # B,3,V,V,V
        
        # check the coordinate system

        # extract 2D image features
        spatial_volume_feats = []
        for ni in range(0, N):
            pose_source_ = target_poses[:, ni]
            K_source_ = target_Ks[:, ni]
            x_ = x[:, ni]
            C = x_.shape[1]

            coords_source = get_warp_coordinates(spatial_volume_verts, x_.shape[-1], self.input_image_size, K_source_, pose_source_).view(B, V, V * V, 2)
            unproj_feats_ = F.grid_sample(x_, coords_source, mode=grid_sample_mode, padding_mode=grid_sample_padding, align_corners=True)
            unproj_feats_ = unproj_feats_.view(B, C, V, V, V)
            spatial_volume_feats.append(unproj_feats_)

        spatial_volume_feats = torch.stack(spatial_volume_feats, 1) # B,N,C,V,V,V
        
        if apply_3DCNN:
            N = spatial_volume_feats.shape[1]
            spatial_volume_feats = spatial_volume_feats.view(B, N*C, V, V, V)

            spatial_volume_feats = self.cost_network(spatial_volume_feats)  # b,64,32,32,32  (B C Z Y X)
            
            triplane_cost = self.volume2triplane(spatial_volume_feats[:, None]) # [B 3p (V V) (V) C]
            
            triplane = self.triplane_network(rearrange(triplane_cost, "B Np (H W) V C -> B Np H W (V C)", Np=3, H=V, W=V, V=V))
            
            return triplane # [B Np C V V]
            
        return spatial_volume_feats, spatial_volume_verts
    
    # todo: need to check the xyz dims direction
    def volume2triplane(self, dense_volume): 
        """
        @param spatial_volume_feats_before_cv:    B,N,C,Z,Y,X  features in each voxels
        @return: B,3,V,V,(V*N),C kv for each triplane features [XY XZ YZ]
        """        
        B,N,C,V,_,_ = dense_volume.shape
        # from torchvision.utils import save_image
        #from [B,N,C,X,Y,Z] to [B N X Y Z C]  
        
        dense_volume = rearrange(dense_volume, "B N C Z Y X  -> B Z Y X N C")

        plane_yx = rearrange(dense_volume, "B Z Y X N C-> B (Y X) (Z N) C")
        plane_zx = rearrange(dense_volume, "B Z Y X N C -> B (Z X) (Y N) C")
        plane_zy = rearrange(dense_volume, "B Z Y X N C -> B (Z Y) (X N) C")
 
        triplane = torch.stack([plane_yx, plane_zx, plane_zy],dim=1)  # [B 3p (V V) (V N) C]
        return triplane
    
    # to check
    def prepare_sparse_token_index(self, x, target_poses, target_Ks, grid_sample_mode='bilinear'):
        B, N, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        ref_grid = create_meshgrid(H, W, normalized_coordinates=False).to(dtype).to(device)  # (1, H, W, 2)
        ones = torch.ones([1, H, W, 1]).to(dtype).to(device)
        grids = []
        for n in range(0, N):
            # the index order (w, h, viewid), used to grid_sample
            grids.append(torch.cat([ref_grid, ones*n], dim=-1)) # append a view id to each index map, from 0 to N-1
        grids = torch.cat(grids, dim=0) # (N, H, W, 3)
        grids = repeat(grids, "N H W C -> B N H W C", B=B)
        grids = rearrange(grids, "B N H W C -> B N C H W")
        
        dense_volume, _ = self.construct_spatial_volume(
            grids, target_poses, target_Ks, 
            grid_sample_mode=grid_sample_mode, 
            grid_sample_padding='border',
            apply_3DCNN=False
        )  # B,N,C,Z,X,Y

        # [B 3p V V (V N) C] where C means (view_idx, u, v), where view_idx is from (0, N-1)
        triplane_index = self.volume2triplane(dense_volume)
        if self.cfg.robust_sampling:
            triplane_index = self.robust_sampling(triplane_index, H, W)

        # normalize the index into -1~1
        triplane_index[:, :, :, :, 0] = 2 * triplane_index[:, :, :, :, 0] / (W - 1) - 1 # W
        triplane_index[:, :, :, :, 1] = 2 * triplane_index[:, :, :, :, 1] / (H - 1) - 1 # h
        triplane_index[:, :, :, :, 2] = 2 * triplane_index[:, :, :, :, 2] / (N - 1) - 1 # NumViews
        
        return triplane_index, dense_volume # [B 3p (V V) (V N) C], [B,N,C,Z,X,Y]
    
    def robust_sampling(self, triplane_index, H, W):
        top_left = triplane_index.clone()
        top_right = triplane_index.clone()
        bottom_left = triplane_index.clone()
        bottom_right = triplane_index.clone()

        top_left[..., :2] = top_left[..., :2].floor()
        top_right[..., :2] = top_right[..., :2].floor()
        top_right[..., 0] += 1
        bottom_left[..., :2] = bottom_left[..., :2].floor()
        bottom_left[..., 1] += 1
        bottom_right[..., :2] = bottom_right[..., :2].floor()
        bottom_right[..., 0] += 1
        bottom_right[..., 1] += 1

        robust_index = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=-2)
        robust_index[..., 0] = torch.clamp(robust_index[..., 0], 0, W)
        robust_index[..., 1] = torch.clamp(robust_index[..., 1], 0, H)

        return robust_index

    # def get_sparse_feats(self, x, target_poses, target_Ks, grid_sample_mode='bilinear'):
    #     #B, N, C, H, W = x.shape
        
    #     triplane_index, _ = self.prepare_sparse_token_index(x, target_poses, target_Ks, grid_sample_mode)
        
    #     sparse_x = extract_feat_tokens_by_triplaneIndex(x, triplane_index,grid_sample_mode)
        
    #     return sparse_x  # [B Np Va Vb VcNv c]
        

def load_data(root_dir, views=['front', 'left', 'right', 'back']):

    def np2tensor(x):
        return torch.from_numpy(x).float()

    depths = []
    rgbs = []
    projs = []
    masks = []
    Ks = []
    w2cs = []
    coord_maps = []

    for view in views:
        img_path = os.path.join(root_dir, "rgb_%03d_%s.webp" % (0, view))
        normal_path = os.path.join(root_dir, "normals_%03d_%s.webp" % (0, view))
        coordinate_path = os.path.join(root_dir, "coordinate_%03d_%s.npy" % (0, view))
        depth_path = os.path.join(root_dir, "depth_%03d_%s.png" % (0, view))
        K_path = os.path.join(root_dir, "%03d_%s_K.txt" % (0, view))
        w2c_path = os.path.join(root_dir, "%03d_%s_RT.txt" % (0, view))
        img = np.array(Image.open(img_path)).astype(np.float32)
        mask = img[:, :, 3] > 128
        img = img[:, :, :3]
        depth = cv2.imread(depth_path, -1) / 65535. * 10
        # depth[~mask] = 0.

        intrinsic = torch.eye(4)
        intrinsic[:3, :3] = np2tensor(np.loadtxt(K_path))
        intrinsic = (intrinsic @ torch.diag(torch.Tensor([1, -1, -1, 1])))

        w2c = torch.eye(4)
        w2c[:3, :4] = np2tensor(np.loadtxt(w2c_path))

        # pdb.set_trace()
        
        proj =  intrinsic @ w2c 

        coord_map = np.load(coordinate_path, allow_pickle=True)

        depths.append(depth)
        rgbs.append(img)
        masks.append(mask)
        projs.append(proj)
        Ks.append(intrinsic)
        w2cs.append(w2c)
        coord_maps.append(coord_map)

    depths = np.stack(depths)[:, None, :, :]  # [B 1 H W]
    projs = torch.stack(projs)
    rgbs = np.stack(rgbs)
    masks = np.stack(masks)
    coord_maps = np.stack(coord_maps)

    Ks = torch.stack(Ks)
    w2cs = torch.stack(w2cs)

    depths, projs, rgbs, masks = np2tensor(depths), projs, np2tensor(rgbs), np2tensor(masks)

    # pdb.set_trace()
    return depths, projs, rgbs, masks, Ks, w2cs, np2tensor(coord_maps).permute(0, 3, 1, 2)



if __name__ == '__main__':
    bs = 1
    view_num = 1
    input_image_size = 256
    spatial_volume_size = 256
    spatial_volume_length = 1
    model = SpatialVolumeNet(3,  # the channel number of input feature map
                                        view_num,
                                        input_image_size,
                                        spatial_volume_size, 
                                        spatial_volume_length,)

    depths, projs, rgbs, masks, Ks, w2cs, coord_maps = load_data(root_dir="/mnt/nas/xiaoxiao/workspace/ELRM/DepthFusion/data/render_xx/owl", views=['front'])



    dense_volume , spatial_volume_verts= model.construct_spatial_volume(coord_maps.unsqueeze(0), w2cs[...,:3, :4].unsqueeze(0), Ks[...,:3, :3].unsqueeze(0), 
                                                  grid_sample_mode='bilinear', grid_sample_padding='zeros', apply_3DCNN=False)
    

    xyz = coord_maps[0, :, 128, 110]  # -1.1127e-01, -2.6917e-01,  1.0317e-07]
    positions = rearrange(xyz, "C -> 1 1 1 1 C")
    print(xyz)
    value = F.grid_sample(dense_volume[:, 0], positions)
    # value = F.grid_sample(spatial_volume_verts, positions)
    print("value, ", value)


    # triplane = model.convert_triplane_index(spatial_volume_verts.unsqueeze(1))

    # positions = rearrange(xyz, "C -> 1 1 C")
    # # indices2D = torch.stack(
    # #     (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
    # #     dim=-3,
    # # )

    # indices2D = torch.stack(
    #     [positions[..., [0, 2]]],
    #     dim=-3,
    # ) 

    # pdb.set_trace()

    # # [B 3p V V (V N) C]
    # triplane = rearrange(triplane, "B Np Va Vb n c -> B Np Va Vb (n c) ") # [B 3p (V V) (V ViewNum) 3c]
    # triplane = rearrange(triplane, "B Np Va Vb nc -> B Np nc Va Vb")

    # pdb.set_trace()

    # out = F.grid_sample(
    #     rearrange(triplane[:, 1:2], "B Np nc Va Vb -> (B Np) nc Va Vb", Np=1),
    #     rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=1, Nd=2),
    #     align_corners=True,
    #     mode="bilinear",
    # )

    # out_0_unq = rearrange(out, "(B Np) (n c) h w -> B Np n c h w", Np=1, c=3)

    # for i in range(256):
    #     if not out_0_unq[0,0,i].squeeze()[0] < -9:
    #         print(out_0_unq.squeeze()[i])

    # pdb.set_trace()



    # print(value)
    # pdb.set_trace()

    ref_grid = create_meshgrid(coord_maps.shape[-2], coord_maps.shape[-1], normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid =rearrange(ref_grid, "N H W C -> 1 N C H W")
    triplane_index, grid_volume = model.prepare_sparse_token_index(coord_maps.unsqueeze(0), 
                                                   w2cs[...,:3, :4].unsqueeze(0), Ks[...,:3, :3].unsqueeze(0),
                                                   grid_sample_mode='bilinear')
    sparse_tokens = extract_feat_tokens_by_triplaneIndex(ref_grid, triplane_index)
    pdb.set_trace()
    uvi = F.grid_sample(grid_volume[:, 0], positions, align_corners=True)
    print(uvi) # 110,128,0

    positions = rearrange(xyz, "C -> 1 1 C")
    indices2D = torch.stack(
        (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
        dim=-3,
    )

    # [B 3p V V (V N) C]
    triplane_index = rearrange(triplane_index, "B Np Va Vb n c -> B Np Va Vb (n c) ") # [B 3p (V V) (V ViewNum) 3c]
    triplane_index = rearrange(triplane_index, "B Np Hp Wp Cp -> B Np Cp Hp Wp")

    out = F.grid_sample(
        rearrange(triplane_index, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).cuda(),
        rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3, Nd=2).cuda(),
        align_corners=True,
        mode="bilinear",
    )

    out_0_unq = rearrange(out, "(B Np) (V c) h w -> B Np V c h w", Np=3, c=3)

    for i in range(256):
        print(out_0_unq[0,1,i].squeeze())

    pdb.set_trace()
    
        # [B 3p V V (V N) C] [B Np Va Vb VcNv c]
    sparse_tokens = rearrange(sparse_tokens, "B Np Va Vb n c -> B Np Va Vb (n c) ") # [B 3p (V V) (V ViewNum) 3c]
    sparse_tokens = rearrange(sparse_tokens, "B Np Hp Wp Cp -> B Np Cp Hp Wp")

    out = F.grid_sample(
        rearrange(sparse_tokens, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3).cuda(),
        rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3, Nd=2).cuda(),
        align_corners=True,
        mode="bilinear",
    )

    out_0_unq = rearrange(out, "(B Np) (V c) h w -> B Np V c h w", Np=3, c=2)

    for i in range(256):
        print(out_0_unq[0,1,i].squeeze())

    pdb.set_trace()