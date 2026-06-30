# Youwang
from __future__ import division
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import os
import numpy as np
import copy
import wandb
import yaml
import schedulefree
from torchmetrics.image import TotalVariation
from src.utils.utils import (
    to_device,
    process_flame,
    process_enc_flame,
    load_geo_fn_flame,
    psnr,
    resize_transform,
)
from src.config import NS_GEO_STATS_ROOT
from torch.utils.data import DataLoader
from src.utils.gaussian_renderer import render_2dgs
from src.utils.cameras import Camera
from src.utils.loss_utils import ssim, l1_loss
from src.utils.lpipsPyTorch.modules.lpips import LPIPS
from vhap.model.flame import FlameHead

def count_parameters(_model):
    return sum(p.numel() for p in _model.parameters() if p.requires_grad)

class FinetuneMeshUNetPriorModel2DGS(object):
    def __init__(self, model, cfg, device, train_dataset, val_dataset):
        self.model = model

        self.cfg = cfg['training']
        self.gs_type = cfg['decoder']['gs_type']
        self.train_batch_size = int(cfg['training']['batch_size'])
        self.val_batch_size = int(len(val_dataset))
        self.flame = FlameHead(
            cfg['flame']['n_shape'],
            cfg['flame']['n_expr'],
            add_teeth=cfg['flame']['add_teeth'],
            remove_lip_inside=cfg['flame']['remove_lip_inside'],
            face_clusters=("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises"),
        ).to(device)
        self.device = device

        # schedulefree optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            list(self.model.decoder.parameters()),
            lr=self.cfg['lr'] * 0.05)

        for p in self.model.encoder.parameters():
            p.requires_grad = False

        self.lr = self.cfg['lr']
        self.tgt_id = self.cfg['tgt_id']

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.l_lpips = LPIPS().to(device)
        self.l_tv = TotalVariation(reduction='mean').to(device)
        self.geo_fn = load_geo_fn_flame(self.flame, self.cfg, device)

        self.exp_name = f"{self.cfg['exp_name']}_personalize_{self.tgt_id}_st{self.cfg['stage']}"
        self.save_path = os.path.join(self.cfg['exp_path'], f"st{self.cfg['stage']}")
        os.makedirs(self.save_path, exist_ok=True)
        self.vis_save_path = os.path.join(self.save_path, 'vis_results')
        os.makedirs(self.vis_save_path, exist_ok=True)
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.finetune_logs_path = os.path.join(self.save_path, 'finetune_logs')
        os.makedirs(self.finetune_logs_path, exist_ok=True)

        self.val_min = None
        self.geouv_mean = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_mean.npy'))).to(device)
        self.geouv_std  = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_std.npy'))).to(device)

        self.cfg['log_freq'] = self.cfg['log_steps']
        self.cfg['val_freq'] = self.cfg['val_steps']
        self.cfg['ckpt_interval'] = self.cfg['ckpt_steps']

        if self.cfg['log_wandb']:
            self.use_wandb = True
            wandb.init(project='personalize_singleview', entity=self.cfg['wandb_id'], config=cfg, name=self.exp_name)
        else:
            self.use_wandb = False

        num_enc = count_parameters(self.model.encoder)
        num_dec = count_parameters(self.model.decoder)
        print('Number of total params : {}'.format(num_enc + num_dec))
        print('** Number of params in Encoder: {}'.format(num_enc))
        print('** Number of params in Decoder: {}'.format(num_dec))

        save_cfg_path = os.path.join(self.save_path, 'config.yaml')
        with open(save_cfg_path, 'w') as file:
            yaml.dump(cfg, file, default_flow_style=False, sort_keys=False)


    def train_epoch(self, epoch, train_data_loader, device):
        sum_loss_dict = {k: torch.tensor(0.0, device=device) for k in self.cfg['lambdas']}

        train_iter = tqdm(train_data_loader)
        for b_idx, _batch in enumerate(train_iter):
            _batch = to_device(_batch, device)
            batch = copy.deepcopy(_batch)

            R = batch['extr_R']
            t = batch['extr_t']
            mv_imgs = batch['mv_imgs']
            mv_masks = batch['mv_masks']
            cam_K = batch['cam_K']
            N, C, H, W = mv_imgs.shape

            batch['tex_uv'] = self.train_dataset.tex_uv[None].expand(N, -1, -1, -1).to(device)
            flame_verts_posed, flame_verts_cano, _ = process_flame(batch, self.flame)
            flame_verts_enc_cano = process_enc_flame(batch, self.flame)
            batch['geo_uv'] = self.geo_fn.to_uv(flame_verts_enc_cano)
            batch['dgeo_uv'] = (batch['geo_uv'] - self.geouv_mean) / self.geouv_std

            gs_params = self.model(
                batch=batch,
                posed_verts=flame_verts_posed,
                canon_verts=flame_verts_cano,
                geo_fn=self.geo_fn
            )

            gs_q = gs_params["primqvec_posed"]
            gs_s = gs_params["primscale"]
            gs_o = gs_params["opacity"]
            gs_c = gs_params["color"]
            gs_p = gs_params["primpos_posed"]

            rendered = {}
            rendered_rgb, rendered_mask, rendered_nrm, rendered_dist, surf_nrm = [], [], [], [], []
            for nid in range(N):
                posed_camera = Camera(R[nid].squeeze(), t[nid].squeeze(), H, W,
                                            K=cam_K[nid].squeeze(), data_device=device)
                render_i = render_2dgs(
                    viewpoint_camera=posed_camera,
                    pts_xyz=gs_p[nid],
                    pts_rgb=gs_c[nid],
                    rotations=gs_q[nid],
                    scales=gs_s[nid],
                    opacity=gs_o[nid],
                    bg_color=self.cfg['bg_color'],
                    device=device
                )
                rendered_rgb.append(render_i['rgb'])
                rendered_mask.append(render_i['rend_alpha'])
                rendered_nrm.append(render_i['rend_normal'])
                rendered_dist.append(render_i['rend_dist'])
                surf_nrm.append(render_i['surf_normal'])

            rendered['rgb'] = torch.stack(rendered_rgb, dim=0)
            rendered['mask'] = torch.stack(rendered_mask, dim=0)
            rendered['rend_nrm'] = torch.stack(rendered_nrm, dim=0)
            rendered['rend_dist'] = torch.stack(rendered_dist, dim=0)
            rendered['surf_nrm'] = torch.stack(surf_nrm, dim=0)

            if self.use_wandb and epoch % self.cfg['log_freq'] == 0 and b_idx == 0:
                with torch.no_grad():
                    N_log = min(4, N)
                    gt_grid, rdr_grid, nrm_grid = [], [], []
                    scale_factor = 0.5
                    # id_ = 0
                    for id_ in range(N_log):
                        gt_grid.append(F.interpolate((mv_imgs[[id_]].cpu()), scale_factor=scale_factor, mode='bilinear'))
                        rdr_grid.append(F.interpolate((rendered['rgb'].view(N, C, H, W)[[id_]].cpu()), scale_factor=scale_factor, mode='bilinear'))
                        nrm_grid.append(F.interpolate(((0.5 * rendered['rend_nrm'] + 0.5).clip(0.0, 1.0).view(N, C, H, W)[[id_]].cpu()), scale_factor=0.5, mode='bilinear'))

                    log_render = [
                        make_grid(torch.cat(gt_grid, dim=0), nrow=N_log, padding=0),
                        make_grid(torch.cat(rdr_grid, dim=0), nrow=N_log, padding=0),
                        make_grid(torch.cat(nrm_grid, dim=0), nrow=N_log, padding=0),
                    ]

                    renders = make_grid(torch.stack(log_render, dim=0), nrow=1, padding=0).clip(0.0, 1.0)
                    wandb_logs = dict(
                        train_render=wandb.Image(renders)
                    )
                    wandb.log(wandb_logs, step=epoch)
                    del wandb_logs, renders, log_render

            # compute losses
            l_photo = l1_loss(rendered['rgb'] * rendered['mask'], (mv_imgs * mv_masks).view(-1, C, H, W))
            l_mask = l1_loss(rendered['mask'], mv_masks.view(-1, 1, H, W))
            l_lpips = self.l_lpips(2 * resize_transform(rendered['rgb']) - 1, 2 * resize_transform(mv_imgs.view(-1, C, H, W)) - 1).squeeze() / N

            l_tv_disp = (self.l_tv(0.5 * (1 + gs_params["primdisp_tbn"]).view(N, self.cfg['uv_size'], self.cfg['uv_size'], C).permute(0, 3, 1, 2)) +
                         self.l_tv(gs_params['primdisp_tbn'].norm(dim=-1).view(N, 1, self.cfg['uv_size'], self.cfg['uv_size'])))
            l_tv_color = self.l_tv(gs_params["color"].view(N, self.cfg['uv_size'], self.cfg['uv_size'], C).permute(0, 3, 1, 2))

            l_photo *= self.cfg['lambdas']['photo']
            l_mask *= self.cfg['lambdas']['mask']
            l_lpips *= self.cfg['lambdas']['lpips']
            l_tv_disp *= (0.000001 * 10)
            l_tv_color *= (self.cfg['lambdas']['tv_color'] * 5)

            total_loss = (
                    l_photo + l_mask + l_lpips + l_tv_color + l_tv_disp
            )

            l_nrm = ((1 - (rendered['rend_nrm'] * rendered['surf_nrm']).sum(dim=1)) * rendered['mask'].squeeze()).mean()
            l_dist = rendered['rend_dist'].mean()
            l_nrm *= self.cfg['lambdas']['nrm']
            l_dist *= self.cfg['lambdas']['dist']
            total_loss += (l_nrm + l_dist)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=self.cfg['grad_clip'])

            with torch.no_grad():
                loss_dict = {
                    'photo': l_photo, 'mask': l_mask, 'lpips': l_lpips,  #  'ssim': l_ssim,
                    'nrm': l_nrm, 'dist': l_dist,
                    'tv_disp': l_tv_disp,
                    'tv_color': l_tv_color,
                }
                for k in loss_dict:
                    if k in sum_loss_dict.keys():
                        sum_loss_dict[k] += loss_dict[k]
                    else:
                        sum_loss_dict[k] = loss_dict[k]

            self.optimizer.step()
            self.optimizer.zero_grad()

        return sum_loss_dict

    def val_unseen_view(self, _batch, epoch, device):
        _batch = to_device(_batch, device)
        psnrs, lpipss, ssims = [], [], []

        batch = copy.deepcopy(_batch)

        R_posed = batch['extr_R']
        t = batch['extr_t']
        mv_imgs = batch['mv_imgs']
        mv_masks = batch['mv_masks']
        cam_K = batch['cam_K']
        N, C, H, W = batch['mv_imgs'].shape

        batch['tex_uv'] = self.val_dataset.tex_uv[None].expand(N, -1, -1, -1).to(device)
        flame_verts_posed, _, _ = process_flame(batch, self.flame)
        flame_verts_enc_cano = process_enc_flame(batch, self.flame)
        batch['geo_uv'] = self.geo_fn.to_uv(flame_verts_enc_cano)

        batch['dgeo_uv'] = (batch['geo_uv'] - self.geouv_mean) / self.geouv_std

        gs_params_t = self.model(
            batch=batch,
            posed_verts=flame_verts_posed,
            canon_verts=None,
            geo_fn=self.geo_fn,
        )
        gs_qp = gs_params_t["primqvec_posed"]
        gs_s = gs_params_t["primscale"]
        gs_o = gs_params_t["opacity"]
        gs_c = gs_params_t["color"]
        gs_pp = gs_params_t["primpos_posed"]

        rendered_posed = {}
        rendered_posed_rgb, rendered_posed_mask, rendered_posed_nrm = [], [], []

        for nid in range(N):
            posed_camera = Camera(R_posed[nid].squeeze(), t[nid].squeeze(), H, W,
                                        K=cam_K[nid].squeeze(), data_device=device)
            render_i = render_2dgs(
                viewpoint_camera=posed_camera,
                pts_xyz=gs_pp[nid],
                pts_rgb=gs_c[nid],
                rotations=gs_qp[nid],
                scales=gs_s[nid],
                opacity=gs_o[nid],
                bg_color=self.cfg['bg_color'],
                device=device
            )
            rendered_posed_rgb.append(render_i['rgb'])
            rendered_posed_mask.append(render_i['rend_alpha'])
            rendered_posed_nrm.append(render_i['rend_normal'])

        rendered_posed['rgb'] = torch.stack(rendered_posed_rgb, dim=0)
        rendered_posed['mask'] = torch.stack(rendered_posed_mask, dim=0)
        rendered_posed['rend_nrm'] = torch.stack(rendered_posed_nrm, dim=0)

        psnrs.append(psnr(rendered_posed['rgb'] * rendered_posed['mask'], (mv_imgs * mv_masks).view(-1, C, H, W)))
        ssims.append(ssim(rendered_posed['rgb'], mv_imgs.view(-1, C, H, W)))  # rendered['rgb'] : [NxV, 3, 512, 512]

        N_log = 1
        vis_grid = []
        for id_ in range(N_log):
            scale_factor = 0.5
            vis_grid.append(F.interpolate((mv_imgs[[id_]].cpu()), scale_factor=scale_factor, mode='bilinear').squeeze())
            vis_grid.append(F.interpolate((rendered_posed['rgb'].view(N, C, H, W)[[id_]].cpu()), scale_factor=scale_factor, mode='bilinear').squeeze())
            vis_grid.append(F.interpolate(((0.5 * rendered_posed['rend_nrm'] + 0.5).clip(0.0, 1.0).view(N, C, H, W)[[id_]].cpu()), scale_factor=0.5, mode='bilinear').squeeze())

        renders_ = make_grid(vis_grid, nrow=3, padding=0).clip(0.0, 1.0)

        metric_dict = {
            "val/SSIM": torch.stack(ssims).mean(),
            "val/PSNR": torch.stack(psnrs).mean(),
        }

        return metric_dict, renders_

    def save_vis_results(self, _batch, device):
        _batch = to_device(_batch, device)

        batch = copy.deepcopy(_batch)

        R_posed = batch['extr_R']
        t = batch['extr_t']
        mv_imgs = batch['mv_imgs']
        cam_K = batch['cam_K']
        N, C, H, W = batch['mv_imgs'].shape

        batch['tex_uv'] = self.val_dataset.tex_uv[None].expand(N, -1, -1, -1).to(device)
        flame_verts_posed, flame_verts_cano, _ = process_flame(batch, self.flame)
        flame_verts_enc_cano = process_enc_flame(batch, self.flame)
        batch['geo_uv'] = self.geo_fn.to_uv(flame_verts_enc_cano)
        batch['dgeo_uv'] = (batch['geo_uv'] - self.geouv_mean) / self.geouv_std

        gs_params_t = self.model(
            batch=batch,
            posed_verts=flame_verts_posed,
            canon_verts=None,
            geo_fn=self.geo_fn,
        )
        gs_qp = gs_params_t["primqvec_posed"]
        gs_s = gs_params_t["primscale"]
        gs_o = gs_params_t["opacity"]
        gs_c = gs_params_t["color"]
        gs_pp = gs_params_t["primpos_posed"]

        rendered_posed = {}
        rendered_posed_rgb, rendered_posed_mask, rendered_posed_nrm = [], [], []

        for nid in range(N):
            posed_camera = Camera(R_posed[nid].squeeze(), t[nid].squeeze(), H, W,
                                        K=cam_K[nid].squeeze(), data_device=device)
            render_i = render_2dgs(
                viewpoint_camera=posed_camera,
                pts_xyz=gs_pp[nid],
                pts_rgb=gs_c[nid],
                rotations=gs_qp[nid],
                scales=gs_s[nid],
                opacity=gs_o[nid],
                bg_color=self.cfg['bg_color'],
                device=device
            )
            rendered_posed_rgb.append(render_i['rgb'])
            rendered_posed_mask.append(render_i['rend_alpha'])
            rendered_posed_nrm.append(render_i['rend_normal'])

        rendered_posed['rgb'] = torch.stack(rendered_posed_rgb, dim=0)
        rendered_posed['mask'] = torch.stack(rendered_posed_mask, dim=0)
        rendered_posed['rend_nrm'] = torch.stack(rendered_posed_nrm, dim=0)

        for id_ in range(N):
            save_grid = [mv_imgs[[id_]].cpu(), rendered_posed['rgb'][[id_]].cpu(), (0.5 * rendered_posed['rend_nrm'] + 0.5).clip(0.0, 1.0)[[id_]].cpu()]
            renders = make_grid(torch.cat(save_grid, dim=0), nrow=3, padding=0).clip(0.0, 1.0)
            save_image(renders, os.path.join(self.vis_save_path, f"{batch['fids'][id_].item():06d}.png"))
        return None

    def train_model(self, epochs):
        start = 0  # Default start from the beginning

        train_data_loader = DataLoader(
            self.train_dataset,
            num_workers=0,
            pin_memory=True,
            batch_size=self.cfg['train_batch_size'],
            shuffle=True,
        )
        val_data_loader = DataLoader(
            self.val_dataset,
            num_workers=0,
            pin_memory=True,
            batch_size=self.cfg['train_batch_size']
        )

        for epoch in range(start, epochs):
            # validate unseen views
            if epoch % self.cfg['val_freq'] == 0:
                sum_metric_dict = {
                    "val/SSIM": 0.0,
                    "val/PSNR": 0.0,
                }
                with torch.no_grad():
                    self.model.eval()
                    # if using schedulefree optimizer
                    self.optimizer.eval()

                    val_grid = []
                    img_log_idx = [int(k) for k in np.linspace(0, len(val_data_loader) - 1, 4)]
                    for val_b_idx, val_batch in tqdm(enumerate(val_data_loader), desc=f"Epoch {epoch}: Evaluating"):
                        metric_logs, val_render = self.val_unseen_view(val_batch, epoch, self.device)
                        for k in metric_logs:
                            sum_metric_dict[k] += metric_logs[k]
                        if val_b_idx in img_log_idx:
                            val_grid.append(val_render)

                    val_renders = make_grid(val_grid, nrow=1, padding=0).clip(0.0, 1.0)
                    save_image(val_renders, os.path.join(self.finetune_logs_path, f"epoch_{epoch:05d}.png"))

                    n_val = len(val_data_loader)
                    for k in sum_metric_dict.keys():
                        sum_metric_dict[k] /= n_val

                    avg_metric_logs = sum_metric_dict
                    print(avg_metric_logs)
                    if self.use_wandb:
                        val_logs = dict(
                            val_render=wandb.Image(val_renders)
                        )
                        for k in avg_metric_logs:
                            val_logs[k] = avg_metric_logs[k]

                        wandb.log(val_logs, step=epoch)

                torch.cuda.empty_cache()

            self.model.train()
            # if using schedulefree optimizer
            self.optimizer.train()

            sum_loss_dict = self.train_epoch(epoch, train_data_loader, device=self.device)

            n_train = len(train_data_loader)
            for k in sum_loss_dict.keys():
                sum_loss_dict[k] /= n_train

            avg_sum_loss_dict = sum_loss_dict
            total_loss = 0.0
            print_str = f"[{self.exp_name}] \n [Epoch {epoch:05d}] "

            for k in avg_sum_loss_dict:
                total_loss += avg_sum_loss_dict[k]

            print_str += "L_total {:06.4f} |".format(total_loss)

            for k in avg_sum_loss_dict:
                print_str += " " + k + " {:06.4f}".format(avg_sum_loss_dict[k])
            print(print_str)
            if epoch % self.cfg['log_freq'] == 0 and self.use_wandb:
                wandb_logs = {}
                for k in avg_sum_loss_dict:
                    wandb_logs[k] = avg_sum_loss_dict[k]
                wandb.log(wandb_logs, step=epoch)

        self.save_checkpoint(epochs - 1)

        with torch.no_grad():
            self.model.eval()
            for val_b_idx, val_batch in tqdm(enumerate(val_data_loader), desc=f"Saving final video results ..."):
                self.save_vis_results(val_batch, self.device)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.checkpoint_path, f'st{self.cfg["stage"]}_final.pth')
        torch.save({'epoch': epoch,
                    'decoder_state_dict': self.model.decoder.state_dict(),
                    'encoder_state_dict': self.model.encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                   path)

