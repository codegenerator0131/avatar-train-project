#!/usr/bin/env python3
"""Export ELITE trained avatar as 3D Gaussian Splatting PLY file."""
import os, sys, argparse
import numpy as np
import torch
import yaml

# Must be run from elite/ directory with PYTHONPATH=elite/
from src.models.mesh_unet import MeshUNetPriorModel2DGS
from src.utils.utils import load_geo_fn_flame, process_flame
from src.dataloader.test_dataloader import MeshUNetRandomExprRenderDataset
from src.config import NS_GEO_STATS_ROOT
from vhap.model.flame import FlameHead


def quat_wxyz_to_xyzw(q):
    """Convert wxyz → xyzw for PLY format."""
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def save_ply(path, positions, colors, scales, rotations, opacities):
    """Save 3D Gaussian Splatting PLY (standard Inria format)."""
    N = positions.shape[0]

    # Build PLY header
    props = [
        ('x', 'float'), ('y', 'float'), ('z', 'float'),
        ('nx', 'float'), ('ny', 'float'), ('nz', 'float'),
        ('f_dc_0', 'float'), ('f_dc_1', 'float'), ('f_dc_2', 'float'),
        ('opacity', 'float'),
        ('scale_0', 'float'), ('scale_1', 'float'), ('scale_2', 'float'),
        ('rot_0', 'float'), ('rot_1', 'float'), ('rot_2', 'float'), ('rot_3', 'float'),
    ]

    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {N}\n"
    for name, dtype in props:
        header += f"property {dtype} {name}\n"
    header += "end_header\n"

    # SH DC coefficient = color * C0 (C0 = 0.28209479177387814)
    C0 = 0.28209479177387814
    sh_dc = (colors - 0.5) / C0  # inverse of sigmoid-like activation

    # Build data array
    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        positions.astype(np.float32),           # x y z
        normals,                                 # nx ny nz
        sh_dc.astype(np.float32),               # f_dc_0 f_dc_1 f_dc_2
        np.log(opacities.clip(1e-6, 1-1e-6)).astype(np.float32).reshape(-1, 1),  # opacity (logit)
        np.log(scales.clip(1e-8)).astype(np.float32),  # scale (log)
        rotations.astype(np.float32),           # rot wxyz
    ], axis=1)

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    print(f"  Saved {N} Gaussians → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_id', required=True)
    parser.add_argument('--cfg_file', required=True)
    parser.add_argument('--ckpt_file', required=True)
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--res', default='512x512')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(open(args.cfg_file))

    print(f"  Loading model from {args.ckpt_file}...")
    ckpt = torch.load(args.ckpt_file, map_location=device)
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

    geouv_mean = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_mean.npy'))).to(device)
    geouv_std  = torch.from_numpy(np.load(str(NS_GEO_STATS_ROOT / 'ns_geo_uv_std.npy'))).to(device)

    res = (802, 550) if args.res == '802x550' else (512, 512)
    ds = MeshUNetRandomExprRenderDataset(
        pid=args.person_id,
        num_frames=1,
        uv_res=(512, 512),
        res=res,
    )

    with torch.no_grad():
        batch = ds[0]
        # Convert all arrays to tensors and add batch dimension
        def to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                return torch.from_numpy(v).unsqueeze(0).to(device)
            return v
        batch = {k: to_tensor(v) for k, v in batch.items()}

        flame_verts_posed, _, _ = process_flame(batch, flame)

        # Canonical FLAME verts
        _, enc_verts_cano = flame.forward(
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

        batch['tex_uv'] = ds.pid_tex_uv[None].to(device)
        batch['geo_uv'] = geo_fn.to_uv(enc_verts_cano)
        batch['dgeo_uv'] = (batch['geo_uv'] - geouv_mean) / geouv_std

        gs = model.forward(
            batch=batch,
            posed_verts=flame_verts_posed,
            geo_fn=geo_fn,
            is_infer_suppress=True,
        )

    positions  = gs['primpos_posed'][0].cpu().numpy()   # (P, 3)
    rotations  = gs['primqvec_posed'][0].cpu().numpy()  # (P, 4) wxyz
    scales     = gs['primscale'][0].cpu().numpy()        # (P, 3)
    opacities  = gs['opacity'][0].cpu().numpy()          # (P, 1) or (P,)
    colors     = gs['color'][0].cpu().numpy()            # (P, 3)

    if opacities.ndim == 2:
        opacities = opacities[:, 0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    save_ply(args.out_path, positions, colors, scales, rotations, opacities)


if __name__ == '__main__':
    main()
