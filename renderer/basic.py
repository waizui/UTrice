#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import torch
from diff_tritracer import TracerSettings, Tracer
from scene.cameras import Camera
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh


def squeeze_if_not_scalar(tensor: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Squeeze tensor but keep at least 1 dimension to avoid scalar outputs."""
    squeezed = tensor.squeeze(dim) if dim is not None else tensor.squeeze()
    return squeezed if squeezed.ndim > 0 else tensor


def trace(tracer: Tracer, settings: TracerSettings, viewpoint_camera: Camera, pc: TriangleModel, pipe, ray_org, ray_dir, override_color=None):

    tri_points = squeeze_if_not_scalar(pc.get_triangles_points[:, 0, 0])
    scaling = torch.zeros_like(tri_points, dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()
    density_factor = torch.zeros_like(tri_points, dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()

    opacity = pc.get_opacity
    triangles_points = pc.get_triangles_points_flatten
    sigma = pc.get_sigma
    num_points_per_triangle = pc.get_num_points_per_triangle
    cumsum_of_points_per_triangle = pc.get_cumsum_of_points_per_triangle
    number_of_points = pc.get_number_of_points

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color

    mask = ((torch.sigmoid(pc._mask) > 0.01).float() - torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
    opacity = opacity * mask

    return tracer(
        ray_o=ray_org,
        ray_d=ray_dir,
        triangles_points=triangles_points,
        sigma=sigma,
        num_points_per_triangle=num_points_per_triangle,
        cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
        number_of_points=number_of_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scaling=scaling,
        density_factor=density_factor,
        tracer_settings=settings,
    )


def render_basic(tracer: Tracer, settings: TracerSettings, viewpoint_camera: Camera, pc: TriangleModel, pipe, override_color=None):
    ray_o, ray_d = viewpoint_camera.gen_rays()
    render_color, gpu_time, scaling, density_factor, allmap, max_blending = trace(tracer, settings, viewpoint_camera, pc, pipe, ray_o, ray_d, override_color)
    rets = {
        "render": render_color,
        "visibility_filter": 0,
        "radii": 0,
        "scaling": scaling,
        "density_factor": density_factor,
        "max_blending": max_blending,
        "gpu_time": gpu_time[0].item(),
    }

    return rets
