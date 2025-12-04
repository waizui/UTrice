#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the Apache License, Version 2.0.
#
# For inquiries contact jan.held@uliege.be
#
# Additional modifications by:
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0.
#
# For inquiries contact lch01234@gmail.com
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from renderer import TriangleModel


def render_set(model_path, name, iteration, views, triangles, pipeline, background, postfix=""):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    frame_time = 0.0
    frame_count = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rebuild_gas = 0 if frame_count == 0 else 2
        res = render(view, triangles, pipeline, background, rebuild_gas=rebuild_gas)
        rendering = res["render"]
        frame_time += res["gpu_time"]
        frame_count += 1
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + postfix + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + postfix + ".png"))
    if pipeline.debug:
        mean_frame_time = frame_time / frame_count
        print(f"{name} mean frame time: {mean_frame_time}, fps:{1000.0/mean_frame_time}")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        scene = Scene(args=dataset, triangles=triangles, init_opacity=None, init_size=None, nb_points=None, set_sigma=None, no_dome=False, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), triangles, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), triangles, pipeline, background)


def render_scene(triangles: TriangleModel, scene: Scene, dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, img_size: int):
    with torch.no_grad():
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), triangles, pipeline, background, str(img_size))

        if not skip_test:
            render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), triangles, pipeline, background, str(img_size))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
