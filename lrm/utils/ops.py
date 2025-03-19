from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from einops import rearrange

import lrm
from .typing import *

import PIL


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    # rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)


def points_projection(points: Float[Tensor, "B Np 3"],
                    c2ws: Float[Tensor, "B 4 4"],
                    intrinsics: Float[Tensor, "B 3 3"],
                    local_features: Float[Tensor, "B C H W"],
                    # Rasterization settings
                    raster_point_radius: float = 0.0075,  # point size
                    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                    bin_size: int = 0):
    B, C, H, W = local_features.shape
    device = local_features.device
    raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )
    Np = points.shape[1]
    R = raster_settings.points_per_pixel

    w2cs = torch.inverse(c2ws)
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(w2cs.shape[0], -1).to(device)
    cameras = cameras_from_opencv_projection(w2cs[:, :3, :3], w2cs[:, :3, 3], intrinsics, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]

    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Np, C)

    return local_features_proj

def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform


def grid_distortion(image, strength=0.5):
    # image: [H, W, 3]
    # strength: float in [0, 1], strength of distortion

    H, W, C = image.shape
    image = image.permute(2, 0, 1).unsqueeze(0)

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    # construct displacement
    x_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
    x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
    x_steps = (x_steps * W).long() # [num_steps]
    x_steps[0] = 0
    x_steps[-1] = W
    xs = []
    for i in range(num_steps - 1):
        xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
    xs = torch.cat(xs, dim=0) # [W]

    y_steps = torch.linspace(0, 1, num_steps) # [num_steps], inclusive
    y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1) # perturb
    y_steps = (y_steps * H).long() # [num_steps]
    y_steps[0] = 0
    y_steps[-1] = H
    ys = []
    for i in range(num_steps - 1):
        ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
    ys = torch.cat(ys, dim=0) # [H]

    # construct grid
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy') # [H, W]
    grid = torch.stack([grid_x, grid_y], dim=-1) # [H, W, 2]

    grids.append(grid)
    
    grids = torch.stack(grids, dim=0).to(image.device) # [B, H, W, 2]

    # grid sample
    image = F.grid_sample(image, grids, align_corners=False)

    return image.squeeze(0).permute(1, 2, 0)

def transform_relative_pose(c2ws_cond, c2ws):
    n_views_cond = c2ws_cond.shape[0]
    n_views = c2ws.shape[0]
    ret_c2ws_cond = []
    ret_c2ws = []
    ref_c2w = None
    for i in range(n_views_cond):
        c2w = c2ws_cond[i]
        if ref_c2w is None:
            # use the first frame as the reference frame
            ref_c2w = c2w.clone()
            ref_w2c = torch.linalg.inv(ref_c2w)
            ref_distance = ref_c2w[0:3, 3].norm()
            canonical_c2w = torch.as_tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, ref_distance],
                    [0, 0, 0, 1],
                ]
            ).float()
        c2w = canonical_c2w @ ref_w2c @ c2w
        ret_c2ws_cond.append(c2w)
    
    for i in range(n_views):
        c2w = c2ws[i]
        c2w = canonical_c2w @ ref_w2c @ c2w
        ret_c2ws.append(c2w)
    
    return torch.stack(ret_c2ws_cond, dim=0).to(c2ws_cond), torch.stack(ret_c2ws, dim=0).to(c2ws)

def convert_depth_to_distance(depths, intrinsics):
    '''
    depths, distances: [B, H, W, 1]
    intrinsics: [B, 3, 3]
    grid: [2, H, W]
    '''

    assert depths.shape[-1] == 1
    
    batched = True
    if depths.ndim == 3:
        batched = False
        depths = depths.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)

    dtype = depths.dtype
    device = depths.device
    batch_size, H, W, _ = depths.shape
    
    depths = depths.permute(0, 3, 1, 2)
    
    grid = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=dtype, device=device),
        torch.arange(W, dtype=dtype, device=device),
        indexing='ij',
    )).flip(0)
        
    uv = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, 2, H, W]
    uvz = torch.cat([depths * uv, depths], dim=1)
    uvz = uvz.permute(0, 2, 3, 1) # [B, H, W, 3]
    intrinsics_inv = intrinsics.inverse()
    xyz = torch.matmul(uvz, intrinsics_inv.unsqueeze(1).transpose(2, 3)) # [B, H, W, 3]
    distances = torch.norm(xyz, p=2, dim=-1, keepdim=True)

    if not batched:
        distances = distances.squeeze(0)

    return distances

def c2w_to_spherical(camera_matrix):
    pos = camera_matrix[:3, 3]
    dx, dy, dz = pos[0].item(), pos[1].item(), pos[2].item()
    
    radius = torch.linalg.norm(pos).item()
    
    xy_projection = math.hypot(dx, dz)
    if radius < 1e-7:
        polar_angle = 0.0
    else:
        polar_angle = math.atan2(dy, xy_projection)
    
    if xy_projection < 1e-6:
        azimuth = 0.0
    else:
        azimuth = math.atan2(dx, dz) % (2 * math.pi)
    
    return polar_angle, azimuth, radius

def spherical_to_c2w(polar, azimuth, radius):
    vertical = radius * math.sin(polar)
    horizontal = radius * math.cos(polar)
    px = horizontal * math.cos(azimuth)
    pz = horizontal * math.sin(azimuth)
    camera_pos = torch.tensor([px, vertical, pz], dtype=torch.float32)
    
    view_dir = -camera_pos
    view_dir_normalized = view_dir / torch.linalg.norm(view_dir)
    
    global_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    right_axis = torch.linalg.cross(view_dir_normalized, global_up)
    right_axis = right_axis / torch.linalg.norm(right_axis)
    
    true_up = torch.linalg.cross(right_axis, view_dir_normalized)
    true_up = true_up / torch.linalg.norm(true_up)
    
    orientation = torch.column_stack([right_axis, true_up, -view_dir_normalized])
    
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = orientation
    transform[:3, 3] = camera_pos
    
    return transform

def rays_intersect_bbox(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    radius: float,
    near: float = 0.0,
    valid_thresh: float = 0.01,
):
    input_shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    rays_d_valid = torch.where(
        rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d
    )
    if type(radius) in [int, float]:
        radius = torch.FloatTensor(
            [[-radius, radius], [-radius, radius], [-radius, radius]]
        ).to(rays_o.device)
    radius = (
        1.0 - 1.0e-3
    ) * radius  # tighten the radius to make sure the intersection point lies in the bounding box
    interx0 = (radius[..., 1] - rays_o) / rays_d_valid
    interx1 = (radius[..., 0] - rays_o) / rays_d_valid
    t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(interx0, interx1).amin(dim=-1)

    # check wheter a ray intersects the bbox or not
    rays_valid = t_far - t_near > valid_thresh

    t_near[torch.where(~rays_valid)] = 0.0
    t_far[torch.where(~rays_valid)] = 0.0

    t_near = t_near.view(*input_shape, 1)
    t_far = t_far.view(*input_shape, 1)
    rays_valid = rays_valid.view(*input_shape)

    return t_near, t_far, rays_valid


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged

def get_spherical_cameras(
    n_views: int,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float,
    height: int,
    width: int,
):
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)

    fovy = torch.full_like(elevation_deg, fovy_deg) * math.pi / 180

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    # get directions by dividing directions_unit_focal by focal length
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)
    directions_unit_focal = get_ray_directions(
        H=height,
        W=width,
        focal=1.0,
    )
    directions = directions_unit_focal[None, :, :, :].repeat(n_views, 1, 1, 1)
    directions[:, :, :, :2] = (
        directions[:, :, :, :2] / focal_length[:, None, None, None]
    )
    # must use normalize=True to normalize directions here
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

    return rays_o, rays_d


def remove_background(
    image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image


def save_video(
    frames: List[PIL.Image.Image],
    output_path: str,
    fps: int = 30,
):
    # use imageio to save video
    frames = [np.array(frame) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def to_gradio_3d_orientation(mesh):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    return mesh