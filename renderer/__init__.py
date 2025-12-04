#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import torch
from diff_tritracer import Tracer
from arguments import PipelineParams
from renderer.dof import render_dof
from renderer.envlight import render_envlight
from renderer.nromal import render_normal
from renderer.reflect import render_reflect
from renderer.refract import render_refract
from scene.cameras import Camera
from scene.triangle_model import TriangleModel

_tracer = None


def get_tracer(mode: str):
    global _tracer
    if _tracer is None:
        forward = "forward.ptx" if (mode == "normal" or mode == "dof") else f"{mode}.ptx"
        _tracer = Tracer(forward, "backward.ptx")

    return _tracer


def render(viewpoint_camera: Camera, pc: TriangleModel, pipe: PipelineParams, bg_color: torch.Tensor, override_color=None, rebuild_gas=0):
    tracer = get_tracer(pipe.render_mode)
    if pipe.render_mode == "normal":
        return render_normal(tracer, viewpoint_camera, pc, pipe, bg_color, override_color, rebuild_gas)
    if pipe.render_mode == "dof":
        return render_dof(tracer, viewpoint_camera, pc, pipe, bg_color, rebuild_gas)
    if pipe.render_mode == "reflect":
        return render_reflect(tracer, viewpoint_camera, pc, pipe, bg_color, rebuild_gas)
    if pipe.render_mode == "refract":
        return render_refract(tracer, viewpoint_camera, pc, pipe, bg_color, rebuild_gas)
    if pipe.render_mode == "envlight":
        return render_envlight(tracer, viewpoint_camera, pc, pipe, bg_color, rebuild_gas)
