from operator import sub
import pdb
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
from math import pi
from torch.nn.functional import normalize
from dotmap import DotMap

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

def uv_sphere(radius, subdiv):
    subdiv_long = (torch.linspace(0, 1, 2 * subdiv) * 2 * pi).unsqueeze(-1)
    subdiv_lat = (torch.linspace(0, 1, subdiv) * pi).unsqueeze(-1)

    sin_long = torch.sin(subdiv_long)
    cos_long = torch.cos(subdiv_long)

    sin_lat = torch.sin(subdiv_lat)
    cos_lat = torch.cos(subdiv_lat)

    x = radius * torch.matmul(cos_long, sin_lat.T).unsqueeze(-1)
    y = radius * torch.matmul(sin_long, sin_lat.T).unsqueeze(-1)
    z = radius * cos_lat.T.expand(2 * subdiv, subdiv).unsqueeze(-1)

    return torch.cat((x, y, z), dim=-1)

def bounding_long_lat(patch_rays, radii, app_imgs):
    all_ll = longitude_lattitude_norm(patch_rays, radii, app_imgs)
    all_long = all_ll[:, :, 0]
    all_lat = all_ll[:, :, 1]

    long_min = all_long.min(dim=-1)[0]
    long_max = all_long.max(dim=-1)[0]
    lat_min = all_lat.min(dim=-1)[0]
    lat_max = all_lat.max(dim=-1)[0]

    return long_min, long_max, lat_min, lat_max

def bounded_sphere_verts(sphere_verts, bounding_ll, radii, app_imgs):
    long_min, long_max, lat_min, lat_max = bounding_ll
    SB = long_min.shape[0]

    sphere_ll = longitude_lattitude_norm(sphere_verts[None], radii, app_imgs).reshape(-1, 2)
    sphere_long = sphere_ll[:, [0]]
    sphere_lat = sphere_ll[:, [1]]

    all_sphere_verts = []
    for i in range(SB):
        final_long = torch.logical_and(sphere_long >= long_min[i], sphere_long <= long_max[i])
        final_lat = torch.logical_and(sphere_lat >= lat_min[i], sphere_lat <= lat_max[i])
        final_ind = torch.logical_and(final_long, final_lat).flatten()

        all_sphere_verts.append(sphere_verts[final_ind])

    return all_sphere_verts

def viewing_plane_sphere_coords(rays, radii):
    SB, B, _ = rays.shape
    radii = radii.expand(SB, B).unsqueeze(-1)
    
    cam_pos = rays[:, :, [0, 1, 2]]
    cam_dir = rays[:, :, [3, 4, 5]]
    cam_near = rays[:, :, [6]]

    view_coords = cam_pos + cam_dir * cam_near
    view_coords = normalize(view_coords, dim=-1) * radii

    return view_coords

def sample_spherical_rand_rays(rays, sphere_verts, radii, app_imgs, patch_size, subdiv):
    view_coords = viewing_plane_sphere_coords(rays, radii)
    closest_verts = closest_sphere_verts(view_coords, sphere_verts, radii, subdiv)

    uv_env = rays_blinn_newell_uv(closest_verts, radii, app_imgs, patch_size)
    enc_patches = uv_to_rgb_patches(app_imgs, uv_env, patch_size)
    long_lat = longitude_lattitude_norm(closest_verts, radii, app_imgs)
    return enc_patches, long_lat

def sample_spherical_patch_rays(patch_rays, sphere_verts, radii, app_imgs, patch_size, subdiv):
    view_coords = viewing_plane_sphere_coords(patch_rays, radii)
    closest_verts = closest_sphere_verts(view_coords, sphere_verts, radii, subdiv)

    uv_env = rays_blinn_newell_uv(closest_verts, radii, app_imgs, patch_size)
    long_lat = longitude_lattitude_norm(closest_verts, radii, app_imgs)
    return torch.cat(uv_env, dim=-1), long_lat

def sample_spherical_ray_encs(patch_rays, sphere_verts, sphere_encs, radii, app_imgs, patch_size):
    view_coords = viewing_plane_sphere_coords(patch_rays, radii)

    verts, encs = closest_sphere_verts_encs(view_coords, sphere_verts, sphere_encs)
    uv_env = rays_blinn_newell_uv(verts, radii, app_imgs, patch_size)
    long_lat = longitude_lattitude_norm(view_coords, radii, app_imgs)
    return encs, long_lat, uv_env

def inverse_distance_weighting(view_coords, bounded_sph, sphere_encs, radii):
    all_results = []
    for i in range(len(bounded_sph)):
        curr_view_coords = view_coords[i]
        curr_bound_sph = bounded_sph[i]
        curr_encs = sphere_encs[i]

        inner_prod_ci = torch.matmul(curr_bound_sph, curr_view_coords.T)
        norm_ci = torch.norm(curr_bound_sph, dim=-1) * radii[i]

        angle_ci = torch.acos(inner_prod_ci / norm_ci)
        inv_weight = 1 / angle_ci
        
        weighed_result = curr_encs * inv_weight
        weighed_result = weighed_result.sum(dim=0) / inv_weight.sum(dim=0)

        all_results.append(weighed_result)
    
    return all_results

def sphere_intersection(rays, radii):
    SB, B, _ = rays.shape
    cam_pos = rays[:, :, [0, 1, 2]]
    cam_dir = rays[:, :, [3, 4, 5]]

    radii = radii.expand(SB, B)

    cam_pos_proj = (cam_pos * cam_dir).sum(dim=2)
    cam_pos_dist = (cam_pos * cam_pos).sum(dim=2) - (radii ** 2)
    if torch.any(cam_pos_dist < 0.0) or torch.any(cam_pos_proj > 0.0):
        return None
    
    discriminant = (cam_pos_proj ** 2) - cam_pos_dist
    if torch.any(discriminant < 0.0):
        return None
    
    t = torch.clamp(-cam_pos_proj - torch.sqrt(discriminant), min=0.0).unsqueeze(-1)
    return cam_pos + cam_dir * t

def rays_blinn_newell_uv(intersections, radii, app_imgs, patch_size):
    SB, B, _ = intersections.shape
    H, W = app_imgs.shape[2:4]
    H -= patch_size
    W -= patch_size

    radii = radii.expand(SB, B).unsqueeze(-1)
    x = intersections[:, :, [0]]
    y = intersections[:, :, [1]]
    z = intersections[:, :, [2]]

    azimuth = (torch.atan2(z, x) + (2.0 * pi)) % (2.0 * pi)
    u = (W * (azimuth / (2 * pi))).long()
    v = (H * (torch.asin(-y / radii) + (pi / 2)) / pi).long()   # Negative y since top-left is 0, 0

    return u, v

def closest_sphere_verts(view_coords, sphere_verts, radii, subdiv):
    SB, B, _ = view_coords.shape
    sphere_verts = sphere_verts.reshape(-1, 3)
    view_coords = view_coords.reshape(-1, 3)

    max_dot_inds = torch.matmul(sphere_verts, view_coords.T).argmax(dim=0)
    return sphere_verts[max_dot_inds, :].reshape(SB, B, -1)

def closest_sphere_verts_encs(view_coords, sphere_verts, sphere_encs):
    SB, B, _ = view_coords.shape
    sphere_verts = sphere_verts.reshape(-1, 3)
    view_coords = view_coords.reshape(-1, 3)

    max_dot_inds = torch.matmul(sphere_verts, view_coords.T).argmax(dim=0)

    verts = sphere_verts[max_dot_inds, :].reshape(SB, B, -1)
    encs = sphere_encs[max_dot_inds, :].reshape(SB, B, -1)
    return verts, encs

def longitude_lattitude_norm(intersections, radii, app_imgs):
    SB, B, _ = intersections.shape
    H, W = app_imgs.shape[2:4]

    x = intersections[:, :, [0]]
    y = intersections[:, :, [1]]
    z = intersections[:, :, [2]]
    radii = radii.expand(SB, B).unsqueeze(-1)

    long = torch.atan2(z, x) / pi
    lat = torch.asin(-y / radii) / (pi / 2)

    return torch.cat((long, lat), dim=-1)

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

def update_uv_min_max(unq_u, unq_v, uv_min_max, offset):
    u_min = unq_u.min()
    u_max = unq_u.max() + offset
    v_min = unq_v.min()
    v_max = unq_v.max() + offset

    u_min = torch.min(u_min, uv_min_max[0])
    u_max = torch.max(u_max, uv_min_max[1])
    v_min = torch.min(v_min, uv_min_max[2])
    v_max = torch.max(v_max, uv_min_max[3])

    return (u_min, u_max, v_min, v_max)

def recompose_render_dicts(render_dicts):
    coarse_rgb = []
    coarse_depth = []
    fine_rgb = []
    fine_depth = []

    for rend_dict in render_dicts:
        coarse_rgb.append(rend_dict.coarse.rgb)
        coarse_depth.append(rend_dict.coarse.depth)
        fine_rgb.append(rend_dict.fine.rgb)
        fine_depth.append(rend_dict.fine.depth)
    coarse_rgb = torch.cat(coarse_rgb, dim=-2)
    coarse_depth = torch.cat(coarse_depth, dim=-1)
    fine_rgb = torch.cat(fine_rgb, dim=-2)
    fine_depth = torch.cat(fine_depth, dim=-1)
    
    out_dict = DotMap()
    out_dict.coarse = DotMap()
    out_dict.fine = DotMap()

    out_dict.coarse.rgb = coarse_rgb
    out_dict.coarse.depth = coarse_depth
    out_dict.fine.rgb = fine_rgb
    out_dict.fine.depth = fine_depth

    return out_dict


