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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def gen_rays(self):
        return rays_pinholecam(self)


def pinhole_camera_rays(x, y, f_x, f_y, w, h, ray_jitter=None, device=None):
    """
    Args:
        x, y: torch tensors of shape [N] or [H, W]
        f_x, f_y: focal lengths (float or scalar tensor)
        w, h: image width and height
        ray_jitter: optional callable -> generates random jitter offsets
        device: torch device (optional)

    Returns:
        ray_origin: (H, W, 3)
        ray_direction: (H, W, 3) normalized
    """
    if device is None:
        device = x.device

    x = x.to(torch.float32)
    y = y.to(torch.float32)

    if ray_jitter is not None:
        jitter = ray_jitter(x.shape).to(device)
        jitter_xs = jitter[..., 0]
        jitter_ys = jitter[..., 1]
    else:
        jitter_xs = torch.full_like(x, 0.5)
        jitter_ys = torch.full_like(y, 0.5)

    xs = ((x + jitter_xs) - 0.5 * w) / f_x
    ys = ((y + jitter_ys) - 0.5 * h) / f_y

    ray_lookat = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    ray_origin = torch.zeros_like(ray_lookat)

    ray_direction = ray_lookat / torch.norm(ray_lookat, dim=-1, keepdim=True)
    return ray_origin, ray_direction


def camera_to_world_rays(ray_o, ray_d, R, T):
    """
    poses: [N, 4, 4] camera-to-world transformation matrices

    """

    R = R[:3, :3]

    ray_o_world = ray_o @ R.T + T  # [R @ V].T
    ray_d_world = ray_d @ R.T

    return ray_o_world, ray_d_world


def rays_pinholecam(viewpoint_cam: Camera, ray_jitter=None):
    W = viewpoint_cam.image_width
    H = viewpoint_cam.image_height
    ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, 0, 1]]).float().cuda().T
    intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
    cx = intrins[0, 2].item()
    cy = intrins[1, 2].item()

    fx = intrins[0, 0].item()
    fy = intrins[1, 1].item()

    R = viewpoint_cam.world_view_transform
    T = viewpoint_cam.camera_center

    u = torch.tile(torch.arange(W).unsqueeze(0), (H, 1)).cuda()  # [H, W]
    v = torch.tile(torch.arange(H).unsqueeze(1), (1, W)).cuda()  # [H, W]

    rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, fx, fy, W, H, ray_jitter)

    ray_o_world, ray_d_world = camera_to_world_rays(rays_o_cam, rays_d_cam, R, T)
    return ray_o_world, ray_d_world


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
