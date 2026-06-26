import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import torch
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
import argparse
import yaml
import random
import numpy as np
from src.dataloader.finetune_dataloader import (
    MeshUNetSVVideoTrainDataset,
    MeshUNetSVVideoTestDataset,
    MeshUNetSVVideoRandExprTrainDataset,
)
from src.models.mesh_unet import MeshUNetPriorModel2DGS
from src.pipeline.finetuning import FinetuneMeshUNetPriorModel2DGS
from src.config import SINGLEVIEW_TRACKED_ROOT


def main(args, cfg, train_dataset, val_dataset):
    device = torch.device("cuda")
    seed = cfg['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = MeshUNetPriorModel2DGS(cfg, device).to(device)

    if model is not None:
        ckpt = torch.load(args.ckpt_path)
        print(f"Model: {type(model).__name__} // Trained epoch: {ckpt['epoch']}...")
        model.decoder.load_state_dict(ckpt['decoder_state_dict'])
        model.encoder.load_state_dict(ckpt['encoder_state_dict'])
        cfg['training']['ckpt_epoch'] = ckpt['epoch']

    trainer = FinetuneMeshUNetPriorModel2DGS(model, cfg, device, train_dataset, val_dataset)
    print(f"Trainer: {type(trainer).__name__}")
    trainer.train_model(args.n_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, type=int, choices=[1, 2],
                        help='1: finetune prior on real frames, 2: difix with real+synth frames')
    parser.add_argument('--exp_path', required=True, type=str)
    parser.add_argument('--tgt_id', required=True, type=str)
    # n_steps/val_steps default to None; stage-specific defaults applied after parsing
    parser.add_argument('--n_steps', default=None, type=int)
    parser.add_argument('--val_steps', default=None, type=int)
    parser.add_argument('--log_steps', default=10, type=int)
    parser.add_argument('--log_wandb', default=True, type=eval)
    parser.add_argument('--singleview_bs', default=8, type=int)
    parser.add_argument('--start_frame', default=0, type=int)
    parser.add_argument('--num_real_frames', default=64, type=int,
                        help='Number of real training frames (used in both stages)')
    parser.add_argument('--num_synth_frames', default=None, type=int,
                        help='Number of synthetic training frames (stage 2 only; default 64)')
    parser.add_argument('--eval_last_n_frames', default=32, type=int)
    parser.add_argument('--merged_ft_dataset', default=False, type=eval,
                        help='Stage 1 only: use merged finetune dataset')
    parser.add_argument('--prior_cfg', default='checkpoints/ELITE/config.yaml', type=str,
                        help='Path to the pretrained ELITE prior config.yaml')
    parser.add_argument('--prior_ckpt', default='checkpoints/ELITE/3d_prior.pth', type=str,
                        help='Path to the pretrained ELITE prior checkpoint file')
    parser.add_argument('--res', default='512x512', choices=['512x512', '802x550'],
                        help='Input video resolution: 512x512 (default) or 802x550')
    args = parser.parse_args()

    if args.n_steps is None:
        args.n_steps = 21 if args.stage == 1 else 51
    if args.val_steps is None:
        args.val_steps = 10 if args.stage == 1 else 20
    if args.num_synth_frames is None:
        args.num_synth_frames = 0 if args.stage == 1 else 64

    args.exp_name = args.exp_path.split('/')[-1]
    args.cfg_path = args.prior_cfg
    args.ckpt_path = args.prior_ckpt

    res = (802, 550) if args.res == '802x550' else (512, 512)
    ds_rate = 1

    cfg = yaml.safe_load(open(args.cfg_path, 'r'))

    cfg['training']['tgt_id'] = args.tgt_id
    cfg['training']['exp_name'] = args.exp_name
    cfg['training']['exp_path'] = args.exp_path
    cfg['training']['log_wandb'] = args.log_wandb
    cfg['training']['val_steps'] = args.val_steps
    cfg['training']['log_steps'] = args.log_steps
    cfg['training']['ckpt_steps'] = args.n_steps - 1
    cfg['training']['train_batch_size'] = args.singleview_bs
    cfg['training']['stage'] = str(args.stage)
    cfg['training']['merged_ft_dataset'] = args.merged_ft_dataset if args.stage == 1 else False

    # Clamp num_real_frames to the actual available training frames.
    pid_flame_dir = f"{SINGLEVIEW_TRACKED_ROOT}/{args.tgt_id}_whiteBg_staticOffset"
    pid_flame_path = os.path.join(pid_flame_dir, sorted(os.listdir(pid_flame_dir))[-1], 'tracked_flame_params_30.npz')
    _flame_params = np.load(pid_flame_path)
    total_frames = len(_flame_params['timestep_id'])
    available_frames = total_frames - args.start_frame - args.eval_last_n_frames
    if args.num_real_frames > available_frames:
        print(f"[WARNING] num_real_frames ({args.num_real_frames}) exceeds available train frames ({available_frames}). Overriding.")
        args.num_real_frames = available_frames

    if args.stage == 1:
        train_dataset = MeshUNetSVVideoTrainDataset(
            pid=args.tgt_id,
            res=res,
            uv_res=(512, 512),
            ds_rate=ds_rate,
            num_frames=args.num_real_frames,
            eval_last_n_frames=args.eval_last_n_frames,
        )
    else:
        train_dataset = MeshUNetSVVideoRandExprTrainDataset(
            pid=args.tgt_id,
            res=res,
            uv_res=(512, 512),
            ds_rate=ds_rate,
            start_frame=args.start_frame,
            eval_last_n_frames=args.eval_last_n_frames,
            num_real_frames=args.num_real_frames,
            num_synth_frames=args.num_synth_frames,
            exp_path=args.exp_path,
        )

    val_dataset = MeshUNetSVVideoTestDataset(
        pid=args.tgt_id,
        res=res,
        uv_res=(512, 512),
        ds_rate=ds_rate,
        train_num_frames=args.num_real_frames,
        eval_num_frames=args.eval_last_n_frames,
    )

    main(args, cfg, train_dataset, val_dataset)
