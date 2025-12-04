#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import torch

def pixel_to_disc_shirley(seed):
    """seed is a point on the unit square [0, 1]"""
    a = 2.0 * seed[:, 0] - 1.0
    b = 2.0 * seed[:, 1] - 1.0
    mask = a * a > b * b
    pi = torch.pi
    r = torch.where(mask, a, b)
    phi = torch.where(mask, (pi / 4.0) * (b / a), (pi / 4.0) * (a / b) + (pi / 2.0))
    disc_coords = r * torch.cos(phi), r * torch.sin(phi)
    return torch.stack(disc_coords)


def gen_dof_rays(focus_z, aperture_size, camera_R, rays):
    rays_ori, rays_dir = rays.rays_ori, rays.rays_dir
    ray_count = rays_ori.shape[1] * rays_ori.shape[2]
    lookat = rays_ori + rays_dir * focus_z

    seed = torch.rand([ray_count, 2], device=rays_ori.device)

    blur = aperture_size * pixel_to_disc_shirley(seed)
    expanded_cam = camera_R[:3, :2][None].expand(ray_count, 3, 2)
    rays_ori = rays_ori + (expanded_cam @ blur.T[:, :, None]).reshape_as(rays_ori)
    rays_dir = (lookat - rays_ori) / focus_z

    return rays_ori, rays_dir
