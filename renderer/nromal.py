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
from utils.point_utils import depth_to_normal

def render_normal(tracer: Tracer, viewpoint_camera: Camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, override_color=None, rebuild_gas=0):
    """
    Render all for training
    """

    settings = TracerSettings(
        bg=bg_color,
        scale_modifier=1.0,
        sh_degree=pc.active_sh_degree,
        debug=False,
        rebuild_gas=rebuild_gas,
        extra_params=torch.tensor(0.0, device="cuda"),
    )

    ray_o, ray_d = viewpoint_camera.gen_rays()
    rendered_image, gpu_time, scaling, density_factor, allmap, max_blending = trace(tracer, settings, viewpoint_camera, pc, pipe, ray_o, ray_d, override_color)
    rets = {
        "render": rendered_image,
        "visibility_filter": 0,
        "radii": 0,
        "scaling": scaling,
        "density_factor": density_factor,
        "max_blending": max_blending,
        "gpu_time": gpu_time[0].item(),
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    # already in world space using tracer
    # render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    # depth_ratio=0
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal,
        }
    )

    return rets
