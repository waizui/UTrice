#
# Copyright (c) 2025 Changhe Liu 
# Licensed under the Apache License, Version 2.0. 
#
# For inquiries contact lch01234@gmail.com
#

import os
from pathlib import Path

import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
import matplotlib as mpl


def save_img(data: torch.Tensor, path: os.PathLike | str):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(data.cpu(), save_path)


def render_triangle(
    vertices,  # [(x,y), (x,y), (x,y)]
    triangle_color,  # (r,g,b)
    background_color,  # (r,g,b)
    sigma=1.0,
    width=512,
    height=512,
):

    V = np.array(vertices, dtype=float)

    if V.max() <= 1.0:
        V[:, 0] *= width
        V[:, 1] *= height

    A, B, C = V

    AB = B - A
    AC = C - A
    edge_cross = np.cross(np.array([AB[0], AB[1], 0]), np.array([AC[0], AC[1], 0]))

    # incenter
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    incenter = (a * A + b * B + c * C) / (a + b + c)

    img = np.zeros((height, width, 3), dtype=np.float32)

    tri_color = np.array(triangle_color, dtype=np.float32)
    bg_color = np.array(background_color, dtype=np.float32)

    verts = [A, B, C]

    # raster
    for y in range(height):
        for x in range(width):

            hit_p = np.array([x, y], dtype=float)

            phi = -np.inf
            dist = None

            for i in range(3):
                p1 = verts[i]
                p2 = verts[(i + 1) % 3]

                # compute edge normal (same logic as CUDA)
                edge_normal_3d = np.cross(edge_cross, np.array([p1[0] - p2[0], p1[1] - p2[1], 0]))
                edge_normal = edge_normal_3d[:2]
                edge_normal /= np.linalg.norm(edge_normal)

                offset = -np.dot(edge_normal, p1)

                if dist is None:
                    dist = np.dot(edge_normal, incenter) + offset
                    if dist > 0:
                        edge_normal = -edge_normal
                        offset = -offset
                        dist = -dist

                edge_dist = np.dot(edge_normal, hit_p) + offset
                phi = max(phi, edge_dist)

            # phi_final
            Cx = (phi / dist) ** sigma if phi < 0 else 0.0

            # pixel color = mix
            img[y, x] = Cx * tri_color + (1 - Cx) * bg_color

    return img


if __name__ == "__main__":
    vertices = [
        (0.0, 0.8660),
        (1.0, 0.8660),
        (0.5, 0.0),
    ]

    triangle_color = (0, 0, 1)
    background_color = (1, 1, 1)

    sigmas = [0.01, 0.4, 1.0, 2.0]

    mpl.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    fig, axes = plt.subplots(1, len(sigmas), figsize=(11, 3))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.25, wspace=0.02)

    for ax, sigma in zip(axes, sigmas):
        img = render_triangle(vertices, triangle_color, background_color, sigma=sigma, width=256, height=256)
        ax.imshow(img)
        ax.set_title(f"σ = {sigma}", pad=10)
        ax.axis("off")

    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1

    gradient = np.linspace(1, 0, 256).reshape(1, 256, 1)
    gradient = gradient * np.array(triangle_color) + (1 - gradient) * np.array(background_color)

    cax = fig.add_axes([left, 0.08, right - left, 0.10])
    cax.imshow(gradient, aspect="auto")
    cax.set_xticks([0, 85, 170, 255])
    cax.set_xticklabels(["1.0", "0.75", "0.25", "0.0"])
    cax.set_yticks([])
    fig.savefig("./output/triangle_window.pdf", dpi=300, bbox_inches="tight")
    plt.show()
