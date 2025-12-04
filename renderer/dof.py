#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import torch
from diff_tritracer import Tracer, TracerSettings
from renderer.basic import trace
from scene.cameras import Camera
from scene.triangle_model import TriangleModel


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


def gen_dof_rays(focus_z, aperture_size, camera_R, rays_ori, rays_dir, seed=None):
    if aperture_size <= 0 or focus_z <= 0:
        return rays_ori, rays_dir

    rays_ori = rays_ori.clone()
    rays_dir = rays_dir.clone()
    original_shape = rays_ori.shape
    ray_count = rays_ori.reshape(-1, 3).shape[0]
    
    # Pefect focus point
    lookat = rays_ori.reshape(-1, 3) + rays_dir.reshape(-1, 3) * focus_z

    if seed is None:
        rnd = torch.rand((ray_count, 2), device=rays_ori.device)
    else:
        rnd = seed

    disk = aperture_size * pixel_to_disc_shirley(rnd)
    disk = disk.permute(1, 0)  # (N, 2)

    cam_basis = camera_R[:3, :2].to(rays_ori.device)
    offsets = disk @ cam_basis.T  # (N, 3)

    new_ori = rays_ori.reshape(-1, 3) + offsets
    new_dir = (lookat - new_ori) / focus_z

    return new_ori.reshape(original_shape), new_dir.reshape(original_shape)


def render_dof(tracer: Tracer, viewpoint_camera: Camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, rebuild_gas=0):
    samples = 16
    dof_focus = 4.0
    dof_aperture = 0.04

    ray_o, ray_d = viewpoint_camera.gen_rays()

    camera_R = torch.as_tensor(viewpoint_camera.R, dtype=torch.float32, device=ray_o.device)

    acc_render = None

    for sample in range(samples):
        sample_o, sample_d = gen_dof_rays(dof_focus, dof_aperture, camera_R, ray_o, ray_d)
        settings = TracerSettings(
            bg=bg_color,
            scale_modifier=1.0,
            sh_degree=pc.active_sh_degree,
            debug=False,
            rebuild_gas=rebuild_gas if sample == 0 else 2,
            extra_params=torch.tensor(0.0, device="cuda"),
        )

        rendered, *_ = trace(tracer, settings, viewpoint_camera, pc, pipe, sample_o, sample_d)

        if acc_render is None:
            acc_render = rendered
        else:
            acc_render = acc_render + rendered

    rendered_image = acc_render / samples

    rets = {
        "render": rendered_image,
        "gpu_time": 0,
    }
    return rets
