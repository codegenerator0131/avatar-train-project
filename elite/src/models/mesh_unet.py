# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torchvision.io

import src.nn.layers as la

import cv2
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.nn.layers import make_linear
from src.nn.unet import UNetWBConcat
from src.nn.unet_gs import SongUNet
from roma import rotmat_to_unitquat, quat_product, quat_xyzw_to_wxyz, unitquat_to_rotmat, quat_composition, quat_action, \
    quat_normalize, rotvec_to_unitquat
from src.utils.geom import compute_tbn_rotmat
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)

primscale_range: Tuple[float, float] = (0.0, 20.0)


class ExprEncoder(nn.Module):
    def __init__(
            self,
            n_embs: int,
    ):
        super().__init__()
        self.n_embs = n_embs
        self.proj_expr = nn.Sequential(
            *make_linear(
                100, self.n_embs,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_rottrans = nn.Sequential(
            *make_linear(
                7, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_neck = nn.Sequential(
            *make_linear(
                4, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_jaw = nn.Sequential(
            *make_linear(
                4, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.proj_eyes = nn.Sequential(
            *make_linear(
                8, self.n_embs // 4,
                "wn",
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        self.mlp = nn.Sequential(
            la.LinearWN(2 * self.n_embs, 2 * self.n_embs),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.LinearWN(2 * self.n_embs, 2 * self.n_embs),
            th.nn.LeakyReLU(0.2, inplace=True),
            la.LinearWN(2 * self.n_embs, self.n_embs)
        )
        self.apply(lambda m: la.glorot(m, 0.2))

    def forward(self, batch):
        rot = batch['flame_rot']
        trans = batch['flame_trans']
        neck = batch['flame_neck']
        jaw = batch['flame_jaw']
        eyes = batch['flame_eyes']
        expr = batch['flame_expr']

        expr_proj = self.proj_expr(expr)
        jaw_proj = self.proj_jaw(rotvec_to_unitquat(jaw))
        leye_quat = rotvec_to_unitquat(eyes[..., :3])
        reye_quat = rotvec_to_unitquat(eyes[..., 3:])
        eyes_proj = self.proj_eyes(th.cat([leye_quat, reye_quat], dim=1))
        rottrans_proj = self.proj_rottrans(th.cat([rotvec_to_unitquat(rot), trans], dim=-1))
        neck_proj = self.proj_neck(rotvec_to_unitquat(neck))

        all_expr_feats = th.cat([expr_proj, jaw_proj, eyes_proj, rottrans_proj, neck_proj], dim=1)
        expr_embs = self.mlp(all_expr_feats)  # [B, d_lat]

        return expr_embs


class MeshUNetPrimDecoder(nn.Module):
    def __init__(self, cfg, uv_size, device):
        super().__init__()
        self.cfg = cfg
        self.uv_size = uv_size
        self.device = device

        self.n_splats = uv_size ** 2
        self.gs_type = cfg['decoder']['gs_type']  # 2dgs or 3dgs
        self.learn_quat = cfg['decoder']['learn_quat']
        self.learn_stoffset = cfg['decoder']['learn_stoffset']
        try:
            self.learn_opacity = cfg['decoder']['learn_opacity']
        except KeyError:
            self.learn_opacity = False

        self.n_color = 3
        if self.gs_type == '3dgs':
            self.n_gs_param = 3 + 1  # (displace 3 + scale 1)
        elif self.gs_type == '2dgs':
            self.n_gs_param = 3 + 2  # (displace 3 + scale 2)

        if self.learn_quat:
            self.n_gs_param += 4

        if self.learn_stoffset:
            self.n_gs_param += 3

        if self.learn_opacity:
            self.n_gs_param += 1

        self.unet_in_channels = 0
        if cfg['decoder']['use_uv_rgb']:
            self.unet_in_channels += 3
        if cfg['decoder']['use_uv_geo']:
            self.unet_in_channels += 3

        self.unet = SongUNet(
            img_resolution=self.uv_size,
            in_channels=self.unet_in_channels,
            out_channels=self.n_color + self.n_gs_param,  # 3 + (3 + 2 + 4) = 12
            model_channels=cfg['decoder']['model_channels'],
            num_blocks=cfg['decoder']['num_blocks'],
            emb_dim_in=cfg['decoder']['lat_dim'],
            channel_mult=cfg['decoder']['channel_mult'],
            channel_mult_noise=0,
            attn_resolutions=cfg['decoder']['attn_resolutions']
        )

        self.geo_enhancenet = UNetWBConcat(
            in_channels=self.n_color + self.n_gs_param,
            out_channels=self.n_gs_param,
            n_init_ftrs=self.n_color + self.n_gs_param,
            size=uv_size
        )

        self.app_enhancenet = UNetWBConcat(
            in_channels=self.n_color + self.n_gs_param,
            out_channels=self.n_color,
            n_init_ftrs=self.n_color + self.n_gs_param,
            size=uv_size
        )

        # fixed 3DGS params.
        def_qvec_tbn = th.zeros((1, 1, 4), device=device)
        def_qvec_tbn[..., -1] = 1.0
        self.def_qvec_tbn = def_qvec_tbn
        self.def_opacity = th.ones((1, 1, 1), device=device)
        self.tf_resize_mask = transforms.Compose([
            transforms.Resize((uv_size, uv_size)),
            transforms.ToTensor()
        ])

        try:
            self.disp_min = cfg['decoder']['2dgs_disp_min']
            self.disp_max = cfg['decoder']['2dgs_disp_max']
        except KeyError:
            self.disp_min = -0.05
            self.disp_max = 0.05

    def forward(
            self,
            uv_input: th.Tensor,
            embs: th.Tensor,
            geom_posed: th.Tensor,
            tbn_quat_posed: th.Tensor,
            geom_canon=None,
            tbn_quat_canon=None,
            geo_fn=None,
            is_infer_suppress=False,
    ):
        P = self.n_splats
        device = embs.device
        uv_mask = geo_fn.valid_mask.reshape(-1, 1).to(device)

        preds = {}

        B = embs.shape[0]

        pred_unet = self.unet(uv_input, embs)
        geo_out = self.geo_enhancenet(pred_unet)  # [B, n_gs_param, uv, uv]
        app_out = self.app_enhancenet(pred_unet)  # [B, 3, uv, uv]

        postex_posed = geo_fn.to_uv(geom_posed)
        primposbase_posed = postex_posed.permute(0, 2, 3, 1).contiguous().view(B, -1, 3)

        color = (app_out + 0.5).permute(0, 2, 3, 1).contiguous().view(B, P, self.n_color)
        geo = geo_out.permute(0, 2, 3, 1).contiguous().view(B, P, self.n_gs_param)
        primscale = 0.1 * th.exp(-3.5 - F.softplus(-(geo[..., 3:5] - 5.0) - 3.5)) * uv_mask
        qvec_tbn = self.def_qvec_tbn.expand(B, P, -1)
        if self.learn_quat:
            if self.gs_type == '3dgs':
                dqvec_tbn = geo[..., 4:8]
            elif self.gs_type == '2dgs':
                dqvec_tbn = geo[..., 5:9]
            else:
                raise NotImplementedError
            qvec_tbn = quat_normalize(qvec_tbn + dqvec_tbn * uv_mask)

        if self.learn_stoffset:
            primdisp_coarse = th.tanh(geo[..., :3]) * 0.2  # large offset (mesh)
            primdisp_fine = th.tanh(geo[..., 9:12]) * 0.1  # small offset
            primdisp_tbn = primdisp_coarse + primdisp_fine
        else:
            primdisp_tbn = (geo[..., :3] * th.tensor([[0.015, 0.015, 0.15]], device=device)).clamp(min=self.disp_min, max=self.disp_max)

        if self.learn_opacity:
            opacity = (th.sigmoid(geo[..., -1:]) * (1 + 2 * 0.0005) - 0.0005) * uv_mask
        else:
            opacity = self.def_opacity.expand(B, P, -1) * uv_mask

        tbn_quat_uv_posed = geo_fn.quat_to_uv_face(tbn_quat_posed).permute(0, 2, 3, 1).contiguous().view(B, P, 4)
        primdisp_posed = quat_action(tbn_quat_uv_posed, primdisp_tbn, is_normalized=True)
        primpos_posed = (primposbase_posed + primdisp_posed) * uv_mask
        primqvec_posed = quat_xyzw_to_wxyz(quat_composition([tbn_quat_uv_posed, qvec_tbn], True))

        if geom_canon is not None and tbn_quat_canon is not None:
            postex_canon = geo_fn.to_uv(geom_canon)
            primposbase_canon = postex_canon.permute(0, 2, 3, 1).contiguous().view(B, -1, 3)
            tbn_quat_uv_canon = geo_fn.quat_to_uv_face(tbn_quat_canon).permute(0, 2, 3, 1).contiguous().view(B, P, 4)
            primdisp_canon = quat_action(tbn_quat_uv_canon, primdisp_tbn, is_normalized=True)
            primpos_canon = (primposbase_canon + primdisp_canon) * uv_mask
            primqvec_canon = quat_xyzw_to_wxyz(quat_composition([tbn_quat_uv_canon, qvec_tbn], True))
        else:
            primpos_canon = None
            primqvec_canon = None

        preds.update(
            color=color.clamp(min=0.0, max=1.0),
            opacity=opacity,
            primpos_posed=primpos_posed,
            primpos_canon=primpos_canon,
            primqvec_posed=primqvec_posed,
            primqvec_canon=primqvec_canon,
            primscale=primscale.clamp(*primscale_range),
            primdisp_tbn=primdisp_tbn,
        )
        if self.learn_stoffset:
            preds.update(
                primdisp_coarse=primdisp_coarse,
                primdisp_fine=primdisp_fine
            )
        return preds


class MeshUNetPriorModel2DGS(nn.Module):
    def __init__(
            self,
            cfg,
            device,
            image_height=512,
            image_width=512,
            bg_weight: float = 1.0,
            is_train: bool = True,
    ):
        super().__init__()
        mode = 'training' if is_train else 'inference'
        self.cfg = cfg
        self.encoder = ExprEncoder(
            n_embs=cfg['decoder']['lat_dim'],
        )

        self.model_type = cfg[mode]['model_type']
        self.height = image_height
        self.width = image_width
        self.bg_weight = bg_weight
        self.device = device
        self.uv_size = cfg[mode]['uv_size']

        self.decoder = MeshUNetPrimDecoder(
            cfg=cfg,
            uv_size=self.uv_size,
            device=self.device,
        )

    def forward(self, batch, posed_verts, canon_verts=None, geo_fn=None, is_infer_suppress=False) -> tuple[Any, Any]:
        tbn_rotmat_posed, _ = compute_tbn_rotmat(posed_verts, geo_fn.vt, geo_fn.vi, geo_fn.vti)
        tbn_quat_posed = rotmat_to_unitquat(
            tbn_rotmat_posed)  # Quaternion, xyzw representation : default in Roma library

        if canon_verts is not None:
            tbn_rotmat_canon, _ = compute_tbn_rotmat(canon_verts, geo_fn.vt, geo_fn.vi, geo_fn.vti)
            tbn_quat_canon = rotmat_to_unitquat(
                tbn_rotmat_canon)  # Quaternion, xyzw representation : default in Roma library
        else:
            tbn_quat_canon = None

        embs = self.encoder(batch)

        uv_input_list = []
        if self.cfg['decoder']['use_uv_rgb']:
            tex_uv = (2 * batch['tex_uv']) - 1.0
            uv_input_list.append(tex_uv)
        if self.cfg['decoder']['use_uv_geo']:
            geo_uv = batch['dgeo_uv']
            uv_input_list.append(geo_uv)
        uv_input = th.cat(uv_input_list, dim=1)  # [B, C, uv, uv]

        preds = self.decoder(
            embs=embs,
            uv_input=uv_input,
            geom_posed=posed_verts,
            tbn_quat_posed=tbn_quat_posed,
            geom_canon=canon_verts,
            tbn_quat_canon=tbn_quat_canon,
            geo_fn=geo_fn,
            is_infer_suppress=is_infer_suppress
        )
        return preds
