#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Transform:
    def __init__(self, height, width, pixel_center=True, mode='bilinear', padding_mode='zeros', align_corners=None):
        self.height = height
        self.width = width
        self.pixel_center = pixel_center
        self.pixel_offset = 0.5 if pixel_center else 0.0
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        zz = torch.ones_like(xx)
        self.pixels = torch.stack([xx + self.pixel_offset, yy + self.pixel_offset, zz], axis=0).cuda()

    def backproject(self, depth, intrinsics):
        inv_K = intrinsics.inverse()
        return ((inv_K @ self.pixels.reshape(3,-1)) * depth.reshape(1,-1)).reshape((3, self.height, self.width))
    
    def project(self, points, intrinsics):
        K = intrinsics
        return ((K @ points.reshape(3,-1)) / points[2].reshape(1,-1)).reshape((3, self.height, self.width))[:2]
    
    def normalize_pixels(self, pixels):
        normalized = pixels - self.pixel_offset
        normalized[0] /= (self.width - 1)
        normalized[1] /= (self.height - 1)
        return 2 * normalized - 1
    
    def transform_T(self, points, T):
        return (T @ torch.cat([points, torch.ones_like(points[:1])], axis=0).reshape(4, -1)).reshape((4, self.height, self.width))[:3]
    
    def transform_stereo(self, points, baseline):
        return points + torch.tensor([baseline, 0, 0]).reshape(3,1,1).cuda()
    
    def warp(self, source, grid):
        return torch.nn.functional.grid_sample(source.unsqueeze(0), self.normalize_pixels(grid).permute(1,2,0).unsqueeze(0), mode=self.mode, align_corners=self.align_corners, padding_mode=self.padding_mode).squeeze(0)
    
    def warp_source2target(self, source_cam, target_cam, target_depth):
        target_points = self.backproject(target_depth, target_cam.K)
        source_points = self.transform_T(target_points, source_cam.world_view_transform.T @ target_cam.world_view_transform.T.inverse())
        source_pixels = self.project(source_points, source_cam.K)
        return self.warp(source_cam.original_image.cuda(), source_pixels)