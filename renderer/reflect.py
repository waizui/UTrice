#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import torch
from diff_tritracer import Tracer, TracerSettings
from renderer.basic import render_basic
from scene.cameras import Camera
from scene.triangle_model import TriangleModel


def render_reflect(tracer: Tracer, viewpoint_camera: Camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, rebuild_gas=0):
    r = [0.5]
    cnt = [-1.0, 0.5, -0.0]
    r.extend(cnt)
    settings = TracerSettings(
        bg=bg_color,
        scale_modifier=1.0,
        sh_degree=pc.active_sh_degree,
        debug=False,
        rebuild_gas=rebuild_gas,
        extra_params=torch.tensor(r, dtype=torch.float32, device="cuda"),
    )

    return render_basic(
        tracer,
        settings,
        viewpoint_camera,
        pc,
        pipe,
    )
