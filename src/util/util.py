from unittest.mock import patch
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings
from random import randint
from torchvision.transforms.functional_tensor import crop
# from dotmap import DotMap
from math import pi
from torch.nn.functional import normalize

def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)


def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor.
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )


def batched_index_select_nd_last(t, inds):
    """
    Index select on dim -1 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one data dimension 'n'
    :param t (batch..., n, m)
    :param inds (batch..., k)
    :return (batch..., n, k)
    """
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2), inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


def homogeneous(points):
    """
    Concat 1 to each point
    :param points (..., 3)
    :return (..., 4)
    """
    return F.pad(points, (0, 1), "constant", 1.0)


def gen_grid(*args, ij_indexing=False):
    """
    Generete len(args)-dimensional grid.
    Each arg should be (lo, hi, sz) so that in that dimension points
    are taken at linspace(lo, hi, sz).
    Example: gen_grid((0,1,10), (-1,1,20))
    :return (prod_i args_i[2], len(args)), len(args)-dimensional grid points
    """
    return torch.from_numpy(
        np.vstack(
            np.meshgrid(
                *(np.linspace(lo, hi, sz, dtype=np.float32) for lo, hi, sz in args),
                indexing="ij" if ij_indexing else "xy"
            )
        )
        .reshape(len(args), -1)
        .T
    )


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def coord_from_blender(dtype=torch.float32, device="cpu"):
    """
    Blender to standard coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def coord_to_blender(dtype=torch.float32, device="cpu"):
    """
    Standard to Blender coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def masked_sample(masks, num_pix, prop_inside, thresh=0.5):
    """
    :return (num_pix, 3)
    """
    num_inside = int(num_pix * prop_inside + 0.5)
    num_outside = num_pix - num_inside
    inside = (masks >= thresh).nonzero(as_tuple=False)
    outside = (masks < thresh).nonzero(as_tuple=False)

    pix_inside = inside[torch.randint(0, inside.shape[0], (num_inside,))]
    pix_outside = outside[torch.randint(0, outside.shape[0], (num_outside,))]
    pix = torch.cat((pix_inside, pix_outside))
    return pix


def bbox_sample(bboxes, num_pix):
    """
    :return (num_pix, 3)
    """
    image_ids = torch.randint(0, bboxes.shape[0], (num_pix,))
    pix_bboxes = bboxes[image_ids]
    x = (
        torch.rand(num_pix) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0])
        + pix_bboxes[:, 0]
    ).long()
    y = (
        torch.rand(num_pix) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1])
        + pix_bboxes[:, 1]
    ).long()
    pix = torch.stack((image_ids, y, x), dim=-1)
    return pix


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    if ndc:
        if not (z_near == 0 and z_far == 1):
            warnings.warn(
                "dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW"
            )
        z_near, z_far = 0.0, 1.0
        cam_centers, cam_raydir = ndc_rays(
            width, height, focal, 1.0, cam_centers, cam_raydir
        )

    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        @ c2w
    )
    return c2w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_conv_2d(
    dim_in,
    dim_out,
    padding_type="reflect",
    norm_layer=None,
    activation=None,
    kernel_size=3,
    use_bias=False,
    stride=1,
    no_pad=False,
    zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError(
            "Padding with stride > 1 not supported, use same_pad_conv2d"
        )

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    conv_block.append(
        nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, bias=use_bias, stride=stride
        )
    )
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    #  else:
    #  nn.init.kaiming_normal_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    mse = ((pred - target) ** 2).mean()
    psnr = -10 * math.log10(mse)
    return psnr


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def rot_to_quat(R):
    """
    Rotation matrix to quaternion
    """
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_module(net):
    """
    Shorthand for either net.module (if net is instance of DataParallel) or net
    """
    if isinstance(net, torch.nn.DataParallel):
        return net.module
    else:
        return net

def ssh_normalization(img_tensor):
    return (img_tensor * 2.0) - 1.0 # normalization used for SSH encoder

def ssh_denormalization(img_tensor):
    return (img_tensor + 1.0) * 0.5

def get_random_patch(t, Hp, Wp):
    H, W = t.shape[-2:]
    i = randint(0, H - Hp)
    j = randint(0, W - Wp)

    return crop(t, i, j, Hp, Wp)

def decompose_to_subpatches(patch, sub_factor):
    SB, _, C, _, _ = patch.shape
    HWp = patch.shape[-1] // sub_factor

    subpatches = []
    for i in range(sub_factor):
        row = []
        for j in range(sub_factor):
            row.append(crop(patch, HWp * i, HWp * j, HWp, HWp).permute(0, 1, 3, 4, 2).reshape(SB, -1, C))
        subpatches.append(row)
    
    return subpatches

def recompose_subpatch_render_dicts_depth(render_dicts, SB, P, sub_factor):
    HWp = P // sub_factor
    coarse = []
    fine = []
    for i in range(len(render_dicts[i])):
        for j in range(len(render_dicts[i])):
            coarse.append(render_dicts[i][j].coarse.depth.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))
            fine.append(render_dicts[i][j].fine.depth.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

    patch_coarse_depth = torch.hstack(coarse)
    patch_fine_depth = torch.hstack(fine)
    
    for i in range(len(1, render_dicts)):
        coarse = []
        fine = []
        for j in range(len(render_dicts[i])):
            coarse.append(render_dicts[i][j].coarse.depth.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))
            fine.append(render_dicts[i][j].fine.depth.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

        patch_coarse_depth = torch.vstack([
            patch_coarse_depth,
            torch.hstack(coarse)
        ])
        patch_fine_depth = torch.vstack([
            patch_fine_depth,
            torch.hstack(coarse)
        ])
    
    return patch_coarse_depth, patch_fine_depth

def recompose_subpatch_rgb_env(subpatch_rgb_env, SB, P, sub_factor):
    HWp = P // sub_factor
    row = []

    for j in range(len(subpatch_rgb_env[0])):
        row.append(subpatch_rgb_env[0][j].permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

    out = torch.cat(row, dim=3)
    
    for i in range(1, len(subpatch_rgb_env)):
        row = []
        for j in range(len(subpatch_rgb_env[i])):
            row.append(subpatch_rgb_env[i][j].permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

        out = torch.cat(
            [out, torch.cat(row, dim=3)],
            dim=2)
    
    return out

def recompose_subpatch_render_dicts_rgb(render_dicts, SB, P, sub_factor):
    HWp = P // sub_factor
    coarse = []
    fine = []

    for j in range(len(render_dicts[0])):
        coarse.append(render_dicts[0][j].coarse.rgb.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))
        fine.append(render_dicts[0][j].fine.rgb.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

    patch_coarse_rgb = torch.cat(coarse, dim=3)
    patch_fine_rgb = torch.cat(fine, dim=3)
    
    for i in range(1, len(render_dicts)):
        coarse = []
        fine = []
        for j in range(len(render_dicts[i])):
            coarse.append(render_dicts[i][j].coarse.rgb.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))
            fine.append(render_dicts[i][j].fine.rgb.permute(0, 2, 1).reshape(SB, 3, HWp, HWp))

        patch_coarse_rgb = torch.cat(
            [patch_coarse_rgb, torch.cat(coarse, dim=3)],
            dim=2)
        patch_fine_rgb = torch.cat(
            [patch_fine_rgb, torch.cat(fine, dim=3)],
            dim=2)
    
    return patch_coarse_rgb, patch_fine_rgb

def bounding_sphere_radius(all_rays):
    _, H, W, _ = all_rays.shape

    # We take all corners of our viewing planes
    corner_ul = all_rays[:, 0, 0, :]
    corner_ur = all_rays[:, H - 1, 0, :]
    corner_ll = all_rays[:, 0, W - 1, :]
    corner_lr = all_rays[:, H - 1, W - 1, :]
    corners = torch.cat([corner_ul, corner_ur, corner_ll, corner_lr])

    # The radius of our bounding sphere, assuming origin (0, 0, 0)
    corners_z_near = corners[:, :3] + corners[:, 3:6] * corners[:, [6]]
    dist_to_origin = torch.norm(corners_z_near, p=2, dim=1)

    return dist_to_origin.max()

def sample_spherical_enc_patches(rays, radii, app_imgs, patch_size):
    sph_intersects = sphere_intersection(rays, radii)
    uv_env = rays_blinn_newell_uv(sph_intersects, app_imgs)
    enc_patches = uv_to_rgb_patches(app_imgs, uv_env, patch_size)
    return enc_patches

def sample_spherical_uv(rays, radii, app_imgs):
    sph_intersects = sphere_intersection(rays, radii)
    uv_env = rays_blinn_newell_uv(sph_intersects, app_imgs)
    return uv_env

def sample_spherical_harm_patch(rays, radii, app_imgs, patch_size):
    sph_intersects = sphere_intersection(rays, radii)
    mean_uv = spherical_intersection_to_mean_map_proj(app_imgs, sph_intersects, radii, patch_size)
    harm_patch = mean_uv_to_harm_patch(app_imgs, mean_uv, patch_size)
    return harm_patch

def sphere_intersection(rays, radii):
    SB, B, _ = rays.shape
    cam_pos = rays[:, :, [0, 1, 2]]
    cam_dir = rays[:, :, [3, 4, 5]]
    cam_pos_dist = torch.norm(cam_pos, p=2, dim=2)

    # Since our sphere center is at 0, 0, 0, calculations simplify
    cam_pos_proj_len = torch.abs((cam_pos * cam_dir).sum(dim=2))
    dist_proj_cent = torch.sqrt(torch.clamp((cam_pos_dist ** 2) - (cam_pos_proj_len ** 2), min=0.0))

    radii = radii.expand(SB, B)
    dist_intersect = torch.sqrt((radii ** 2) - (dist_proj_cent ** 2))
    t = cam_pos_proj_len - dist_intersect
    t = t.unsqueeze(2)

    return cam_pos + cam_dir * t

def rays_blinn_newell_uv(intersections, app_imgs):
    SB, B, _ = intersections.shape
    H, W = app_imgs.shape[2:4]

    cam_pos_norm = normalize(intersections)
    x = cam_pos_norm[:, :, [0]]
    y = cam_pos_norm[:, :, [1]]
    z = cam_pos_norm[:, :, [2]]

    azimuth = (torch.atan2(z, x) + (2.0 * pi)) % (2.0 * pi)
    u = (W * (azimuth / (2 * pi))).long()
    v = (H * (torch.asin(-y) + (pi / 2)) / pi).long()   # Negative y since top-left is 0, 0

    return torch.cat((u, v), dim=-1)

def uv_to_rgb_patches(app_imgs, uv_env, patch_size):
    u, v = uv_env

    SB = app_imgs.shape[0]
    B = u.shape[1]
    P = patch_size

    t = torch.arange(SB)
    t = repeat_interleave(t, B)
    u = u.flatten()
    v = v.flatten()

    app_imgs = app_imgs.unfold(2, P, 1).unfold(3, P, 1)
    return app_imgs[t, :, v, u, :, :]

def uv_to_bounding_patches(app_imgs, uv):
    SB = app_imgs.shape[0]

    u_min = uv[:, 0].min(dim=1)[0]
    u_max = uv[:, 0].max(dim=1)[0]
    v_min = uv[:, 1].min(dim=1)[0]
    v_max = uv[:, 1].max(dim=1)[0]

    # Since SB is really small, this is ok
    bound_patches = []
    for i in range(SB):
        bound_patches.append(app_imgs[i, :, v_min[i]:v_max[i], u_min[i]:u_max[i]])
    
    return bound_patches
