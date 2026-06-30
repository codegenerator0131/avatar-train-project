import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
import yaml
import copy
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.utils import load_geo_fn_flame, to_device, process_flame, imgsave
from src.dataloader.test_dataloader import MeshUNetRandomExprRenderDataset
from src.config import NS_GEO_STATS_ROOT
from src.models.mesh_unet import MeshUNetPriorModel2DGS
from src.utils.gaussian_renderer import render_2dgs
from src.utils.cameras import Camera
from vhap.model.flame import FlameHead


def resize_and_vertical_pad(img_tensor, target_w=550, target_h=802):
    """Resize to (target_w, target_w) square, then center-pad vertically to (target_h, target_w)."""
    if img_tensor.ndim != 3 or img_tensor.shape[0] not in [1, 3]:
        raise ValueError("expected [C, H, W] tensor")

    C, H, W = img_tensor.shape

    img_resized = F.interpolate(
        img_tensor.unsqueeze(0),  # [1, C, H, W]
        size=(target_w, target_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    new_h, new_w = img_resized.shape[1:]
    if new_w != target_w:
        raise ValueError("resized width does not match target_w")

    if new_h > target_h:
        raise ValueError(f"resized height {new_h} exceeds target_h {target_h}")

    padded = torch.ones((C, target_h, target_w), dtype=img_resized.dtype, device=img_resized.device)
    y_off = (target_h - new_h) // 2
    padded[:, y_off:y_off+new_h, :] = img_resized

    return padded


def process_pid_flame(batch, fm):
    with torch.no_grad():
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

        flame = FlameHead(
            cfg['flame']['n_shape'],
            cfg['flame']['n_expr'],
            add_teeth=cfg['flame']['add_teeth'],
            remove_lip_inside=cfg['flame']['remove_lip_inside'],
            face_clusters=("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises"),
        ).to(device)

        geo_fn = load_geo_fn_flame(flame, cfg['training'], device)

        res = (802, 550) if args.res == '802x550' else (512, 512)
        ds = MeshUNetRandomExprRenderDataset(
            pid=args.person_id,
            num_frames=args.num_frames,
            uv_res=(512, 512),
            res=res,
        )

        geouv_mean = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_mean.npy'))).to(device)
        geouv_std  = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_std.npy'))).to(device)

        save_path = os.path.join('/'.join(ckpt_path.split('/')[:-3]), f"rand_expr_dataset")
        os.makedirs(save_path, exist_ok=True)
        view_idx_list = []
        dl = DataLoader(ds, num_workers=0, batch_size=1)

        for bidx, _batch in tqdm(enumerate(dl), desc=f"[{args.person_id}]--[random_expr]"):
            _batch = to_device(_batch, device)

            N, _ = _batch['flame_trans'].shape
            batch = copy.deepcopy(_batch)
            view_idx_list.append(batch['view_idx'].item())

            R = batch['extr_R']
            t = batch['extr_t']
            mv_imgs = batch['mv_imgs']
            cam_K = batch['cam_K']
            N, V, C, H, W = mv_imgs.shape

            flame_verts_posed, _, _ = process_flame(batch, flame)

            # generate unet input data from PID mesh
            flame_verts_enc_cano = process_pid_flame(batch, flame)
            batch['tex_uv'] = ds.pid_tex_uv[None].expand(N, -1, -1, -1).to(device)
            batch['geo_uv'] = geo_fn.to_uv(flame_verts_enc_cano)
            batch['dgeo_uv'] = (batch['geo_uv'] - geouv_mean) / geouv_std

            gs_params_t = model.forward(
                batch=batch,
                posed_verts=flame_verts_posed,
                geo_fn=geo_fn,
                is_infer_suppress=True
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
            if args.res != '802x550':
                padded_rgb = resize_and_vertical_pad(render_i['rgb'].clip(0.0, 1.0), target_w=550, target_h=802)
                img_to_save = padded_rgb
            else:
                img_to_save = render_i['rgb'].clip(0.0, 1.0)
            imgsave(
                    input_data=img_to_save,
                    save_path=os.path.join(save_path, f"{bidx:05d}_rdr.png"),
                    make_grid=False,
                )

        np.save(os.path.join(save_path, 'view_idxs.npy'), np.array(view_idx_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_id', type=str, default='448')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--ckpt_file', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=1000)
    parser.add_argument('--res', default='512x512', choices=['512x512', '802x550'],
                        help='Input video resolution: 512x512 (default) or 802x550')

    args = parser.parse_args()

    assert args.cfg_file is not None
    cfg = yaml.safe_load(open(args.cfg_file, 'r'))

    main(args, cfg)

