import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import pathlib
import torch
import yaml
import copy
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from src.utils.utils import to_device, imgsave, process_flame, load_geo_fn_flame
from src.dataloader.test_dataloader import TestMotionRenderDataset
from src.models.mesh_unet import MeshUNetPriorModel2DGS
from src.utils.gaussian_renderer import render_2dgs
from src.utils.cameras import Camera
from src.config import NS_GEO_STATS_ROOT
from vhap.model.flame import FlameHead


def process_pid_flame(batch, fm):
    with torch.no_grad():
        # for mesh geometry encoding
        _, enc_verts_cano = fm.forward(
            shape=batch['pid_flame_shape'],
            expr=torch.zeros_like(batch['pid_flame_expr']),
            rotation=torch.zeros_like(batch['pid_flame_rot']),
            neck=torch.zeros_like(batch['pid_flame_neck']),
            jaw=torch.zeros_like(batch['pid_flame_jaw']),
            eyes=torch.zeros_like(batch['pid_flame_eyes']),
            translation=torch.zeros_like(batch['pid_flame_trans']),
            zero_centered_at_root_node=True,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=batch['pid_flame_stoffset'],
            dynamic_offset=None,
        )
    return enc_verts_cano


def main(args, cfg):
    with torch.no_grad():
        ckpt_path = args.ckpt_file
        device = torch.device("cuda")
        ckpt = torch.load(ckpt_path)

        print(f"Loading Prior Model, Trained epoch: {ckpt['epoch']}...")
        model = MeshUNetPriorModel2DGS(cfg, device).to(device)
        model.decoder.load_state_dict(ckpt['decoder_state_dict'])
        model.encoder.load_state_dict(ckpt['encoder_state_dict'])
        model.eval()

        # get canonical body
        flame = FlameHead(
            cfg['flame']['n_shape'],
            cfg['flame']['n_expr'],
            add_teeth=cfg['flame']['add_teeth'],
            remove_lip_inside=cfg['flame']['remove_lip_inside'],
            face_clusters=("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises"),
        ).to(device)

        geo_fn = load_geo_fn_flame(flame, cfg['training'], device)

        if args.motion_samples_dir is not None:
            ds = TestMotionRenderDataset(
                pid=args.person_id,
                motion_dir=args.motion_samples_dir,
                num_frames=args.num_frames,
                uv_res=(512, 512),
                res=(512, 512),
                cam_path=args.cam_path,
                spherical_cycles=args.spherical_cycles,
            )
        else:
            raise NotImplementedError

        geouv_mean = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_mean.npy'))).to(device)
        geouv_std  = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_std.npy'))).to(device)

        if args.output_tag:
            output_tag = args.output_tag
        else:
            output_tag = pathlib.Path(args.motion_samples_dir).name
        save_path = os.path.join('/'.join(ckpt_path.split('/')[:-3]), "vis_motion")
        save_path_driver = os.path.join(save_path, 'vid_drive', output_tag)
        save_path_rgb = os.path.join(save_path, f'st{args.stage}_rgb/{output_tag}')
        save_path_nrm = os.path.join(save_path, f'st{args.stage}_nrm/{output_tag}')
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path_driver, exist_ok=True)
        os.makedirs(save_path_rgb, exist_ok=True)
        os.makedirs(save_path_nrm, exist_ok=True)

        dl = DataLoader(ds, num_workers=0, batch_size=1)

        for _batch in tqdm(dl, desc=f"[{args.person_id}]--[{args.motion_id}]"):
            _batch = to_device(_batch, device)

            N, _ = _batch['flame_trans'].shape
            batch = copy.deepcopy(_batch)

            R = batch['extr_R']
            t = batch['extr_t']
            mv_imgs = batch['mv_imgs']
            mv_masks = (batch['mv_masks'] >= 0.95).float()
            cam_K = batch['cam_K']
            N, V, C, H, W = mv_imgs.shape

            flame_verts_posed, _, _ = process_flame(batch, flame)

            # generate unet input data from PID mesh
            flame_verts_enc_cano = process_pid_flame(batch, flame)
            batch['tex_uv'] = ds.pid_tex_uv[None].expand(N, -1, -1, -1).to(device)
            batch['geo_uv'] = geo_fn.to_uv(flame_verts_enc_cano)
            batch['dgeo_uv'] = (batch['geo_uv'] - geouv_mean) / geouv_std

            # flame_verts_posed : [B, 5143, 3]
            # flame_verts_cano, flame_verts_enc_cano : [B, 5143, 3]
            gs_params_t = model.forward(
                batch=batch,
                posed_verts=flame_verts_posed,
                geo_fn=geo_fn,
            )

            gs_q = gs_params_t["primqvec_posed"]
            gs_s = gs_params_t["primscale"]
            gs_o = gs_params_t["opacity"]
            gs_c = gs_params_t["color"]
            gs_p = gs_params_t["primpos_posed"]

            nid, vid = 0, 0
            posed_camera = Camera(R[nid, vid].squeeze(), t[nid][vid].squeeze(), H, W,
                                        K=cam_K[nid][vid].squeeze(), data_device=device)
            render_i = render_2dgs(
                viewpoint_camera=posed_camera,
                pts_xyz=gs_p[nid],
                pts_rgb=gs_c[nid],
                rotations=gs_q[nid],
                scales=gs_s[nid],
                opacity=gs_o[nid],
                bg_color=[1.0, 1.0, 1.0],
                device=device
            )
            imgsave(
                input_data=render_i['rgb'].clip(0.0, 1.0),
                save_path=os.path.join(save_path_rgb, f"{(batch['fids'][nid]).item():05d}.png"),
                make_grid=False,
            )
            imgsave(
                input_data=(0.5*(render_i['rend_normal'] + 1)).clip(0.0, 1.0),
                save_path=os.path.join(save_path_nrm, f"{(batch['fids'][nid]).item():05d}.png"),
                make_grid=False,
            )
            imgsave(
                input_data=batch['mv_imgs'].clip(0.0, 1.0),
                save_path=os.path.join(save_path_driver, f"{(batch['fids'][nid]).item():05d}.png"),
                make_grid=False,
            )

        # Generate driver video
        driver_video_path = os.path.join(save_path_driver, f'../driver_{output_tag}.mp4')
        cmd = f'ffmpeg -y -framerate {args.save_fps} -i {save_path_driver}/%05d.png -vf "scale={W}:{H}" -c:v libx264 -pix_fmt yuv420p -crf 20 {driver_video_path}'
        os.system(cmd)

        print(f"Driver video saved to: {driver_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_id', type=str, required=True)
    parser.add_argument('--stage', type=str, default='2')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--spherical_cycles', type=int, default=1)
    parser.add_argument('--cam_path', type=str, default='motion_id', choices=['motion_id', 'spherical'])
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt_file', type=str, required=True)
    parser.add_argument("--save_fps", type=int, default=30)
    parser.add_argument('--motion_samples_dir', type=str, required=True)
    parser.add_argument('--output_tag', type=str, default=None)
    parser.add_argument('--motion_id', type=str, default='521')

    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg_file, 'r'))

    main(args, cfg)

