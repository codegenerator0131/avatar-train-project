"""Chunked HuFix post-processing — avoids OOM by processing frames in small chunks."""
from hufix.src.model import Difix
import torch
import argparse
from glob import glob
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
import numpy as np

SEED = 42


def resize_and_vertical_pad(img_tensor, target_w=550, target_h=802):
    B, C, H, W = img_tensor.shape
    img_resized = F.interpolate(img_tensor, size=(target_w, target_w), mode="bilinear", align_corners=False)
    new_h, new_w = img_resized.shape[-2:]
    padded = torch.ones((B, C, target_h, target_w), dtype=img_resized.dtype, device=img_resized.device)
    y_off = (target_h - new_h) // 2
    padded[:, :, y_off:y_off+new_h, :] = img_resized
    return padded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/2d_prior.pth")
    parser.add_argument('--input_rgb', type=str, required=True)
    parser.add_argument('--input_nrm', type=str, required=True)
    parser.add_argument('--ref_image', type=str, default=None)
    parser.add_argument('--save_dir_rgb', type=str, default=None)
    parser.add_argument('--save_dir_nrm', type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--save_fps", type=int, default=25)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    os.makedirs(args.save_dir_rgb, exist_ok=True)
    os.makedirs(args.save_dir_nrm, exist_ok=True)

    # Load model once
    print("Loading HuFix model...")
    model = Difix(pretrained_path=args.model_path, timestep=199, mv_unet=True)
    model.set_eval()

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Collect file lists
    input_rgb_files = sorted(glob(os.path.join(args.input_rgb, "*.png")))
    input_nrm_files = sorted(glob(os.path.join(args.input_nrm, "*.png")))
    print(f"Found {len(input_rgb_files)} input frames")

    # Load reference image
    ref_tensor = None
    if args.ref_image:
        ref_img = to_tensor(Image.open(args.ref_image).convert("RGB")).unsqueeze(0)
        ref_tensor = ref_img

    # Get output size from first frame
    first = to_tensor(Image.open(input_rgb_files[0]).convert("RGB"))
    _, H, W = first.shape

    # Resize ref if needed
    if ref_tensor is not None:
        _, _, H_ref, W_ref = ref_tensor.shape
        if (H_ref != H) or (W_ref != W):
            ref_tensor = resize_and_vertical_pad(ref_tensor, target_w=W, target_h=H)

    # Process in chunks
    total = len(input_rgb_files)
    chunk_size = args.chunk_size
    print(f"Processing {total} frames in chunks of {chunk_size}...")

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_rgb = input_rgb_files[start:end]
        chunk_nrm = input_nrm_files[start:end]

        # Check if already processed (skip logic)
        all_done = all(
            os.path.isfile(os.path.join(args.save_dir_rgb, os.path.basename(f)))
            for f in chunk_rgb
        )
        if all_done:
            print(f"  [skip] frames {start:05d}-{end-1:05d} already processed")
            continue

        rgb_batch = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in chunk_rgb])
        nrm_batch = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in chunk_nrm])

        # Expand ref to match batch
        if ref_tensor is not None:
            ref_batch = ref_tensor.expand(len(chunk_rgb), -1, -1, -1)
        else:
            ref_batch = None

        with torch.no_grad():
            outputs = model.sample_batch_multi_tensor(
                image=rgb_batch,
                ref_image=ref_batch,
                batch_size=len(chunk_rgb),
            )

        # Save outputs
        for idx in range(outputs.shape[0]):
            img = to_pil(outputs[idx].cpu().clamp(0, 1))
            save_name = os.path.basename(chunk_rgb[idx])
            img.save(os.path.join(args.save_dir_rgb, save_name))

        for idx in range(nrm_batch.shape[0]):
            img = to_pil(nrm_batch[idx].cpu().clamp(0, 1))
            save_name = os.path.basename(chunk_nrm[idx])
            img.save(os.path.join(args.save_dir_nrm, save_name))

        print(f"  Processed frames {start:05d}-{end-1:05d} / {total}")

        # Free VRAM between chunks
        del rgb_batch, nrm_batch, outputs
        torch.cuda.empty_cache()

    # Generate videos
    print("Generating output videos...")
    SAVE_NAME = os.path.basename(args.save_dir_rgb)
    rgb_video = os.path.join(args.save_dir_rgb, f'../../final_rgb_{SAVE_NAME}.mp4')
    nrm_video = os.path.join(args.save_dir_nrm, f'../../final_nrm_{SAVE_NAME}.mp4')
    os.system(f'ffmpeg -y -framerate {args.save_fps} -i {args.save_dir_rgb}/%05d.png -vf "scale={W}:{H}" -c:v libx264 -pix_fmt yuv420p -crf 20 {rgb_video}')
    os.system(f'ffmpeg -y -framerate {args.save_fps} -i {args.save_dir_nrm}/%05d.png -vf "scale={W}:{H}" -c:v libx264 -pix_fmt yuv420p -crf 20 {nrm_video}')
    print("HuFix post-processing done.")
