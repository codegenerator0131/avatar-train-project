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
from torch import nn
import math
import numpy as np
from typing import NamedTuple
from src.utils.graphics_utils import *


class Camera(nn.Module):
    def __init__(self, R, t, height, width, trans=torch.tensor([0.0, 0.0, 0.0]), K=None, scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()
        #
        fx, fy = K[0, 0], K[1, 1]
        # Calculate FoV for x and y axes
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))

        self.zfar = 100.0
        self.znear = 0.01

        self.projection_matrix = getProjectionMatrix_from_K(znear=self.znear, zfar=self.zfar,
                                                            K=K, h=height, w=width).transpose(0, 1)

        self.trans = trans.float().to(data_device)
        self.scale = scale

        self.world_view_transform = getWorld2View2_tensor(R, t, self.trans, self.scale).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = height
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")


def get_camera_for_canon(N, device):
    ### debug
    # front camera for canonical pose
    R_front = torch.eye(3)[None, None, ...].repeat(N, 1, 1, 1).to(device)
    R_front[..., 1, 1] *= -1
    R_front[..., 2, 2] *= -1

    return R_front  # [N, 1, 3, 3]

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

def getWorld2View_tensor(R, t):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0,1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()


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


def getWorld2View2_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()


def getWorld2View2_tensor_batch(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    # R: [B, 3, 3], t: [B, 3]
    B = R.shape[0]
    Rt = torch.zeros((B, 4, 4), device=R.device)
    Rt[:, :3, :3] = R.transpose(1, 2)
    Rt[:, :3, 3] = t
    Rt[:, 3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:, :3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:, :3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()


def getProjectionMatrix_from_K(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4, device=K.device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix_from_K_batch(znear, zfar, K, h, w):
    # K: [B, 3, 3]
    B = K.shape[0]
    near_fx = znear / K[:, 0, 0]
    near_fy = znear / K[:, 1, 1]
    left = - (w - K[:, 0, 2]) * near_fx
    right = K[:, 0, 2] * near_fx
    bottom = (K[:, 1, 2] - h) * near_fy
    top = K[:, 1, 2] * near_fy

    P = torch.zeros((B, 4, 4), device=K.device)
    z_sign = 1.0
    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 3, 2] = z_sign
    P[:, 2, 2] = z_sign * zfar / (zfar - znear)
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    return P


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


def getProjectionMatrix_(znear=0.1, zfar=100.0, fovX=np.deg2rad(45), fovY=np.deg2rad(45)):
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
    P[2, 2] = z_sign * (znear + zfar) / (zfar - znear)
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix_data_render(znear=0.1, zfar=100.0, fovX=np.deg2rad(45), fovY=np.deg2rad(45)):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = -1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = z_sign * 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (znear+zfar) / (zfar - znear)
    P[2, 3] = -(2*zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrix_taichi(znear=0.1, zfar=100.0, fovX=np.deg2rad(45), fovY=np.deg2rad(45)):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = -1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (znear+zfar) / (zfar - znear)
    P[2, 3] = -(2*zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))