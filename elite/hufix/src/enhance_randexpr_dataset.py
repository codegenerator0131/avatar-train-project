from model import Difix
import torch
import argparse
from glob import glob
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from torchvision.utils import save_image
SEED = 42


def resize_and_vertical_pad(img_tensor, target_w=550, target_h=802):
    """
    1) 이미지를 (target_w, target_w) 정방형으로 리사이즈
    2) 세로 방향만 padding 하여 (target_h, target_w)로 맞춤
    - img_tensor: torch.Tensor, [B, C, H, W] (0~1 범위)
    """

    B, C, H, W = img_tensor.shape

    # 1) 정방형 리사이즈 (512->550)
    img_resized = F.interpolate(
        img_tensor,  # [B, C, H, W]
        size=(target_w, target_w),
        mode="bilinear",
        align_corners=False
    )  # [B, C, 550, 550]

    # 2) 세로 padding
    new_h, new_w = img_resized.shape[-2:]
    if new_w != target_w:
        raise ValueError("리사이즈 결과 가로 길이가 target_w와 다릅니다.")

    if new_h > target_h:
        raise ValueError(f"리사이즈된 높이 {new_h}가 target_h {target_h}보다 큽니다.")

    padded = torch.ones((B, C, target_h, target_w), dtype=img_resized.dtype, device=img_resized.device)
    y_off = (target_h - new_h) // 2
    padded[:, :, y_off:y_off+new_h, :] = img_resized

    return padded


def restore_to_512_batch(padded_batch, orig_size=512, resized_size=550, target_h=802):
    """
    550x802 (padding 된 이미지들) -> 512x512 복원 (batch version)
    1) 세로 방향 중앙 crop (802 -> 550)
    2) 550x550 -> 512x512 resize

    Args:
        padded_batch: torch.Tensor, [B, C, H, W]
    Returns:
        restored_batch: torch.Tensor, [B, C, 512, 512]
    """
    if padded_batch.ndim != 4 or padded_batch.shape[1] not in [1, 3]:
        raise ValueError("입력은 [B, C, H, W] 배치 텐서여야 합니다.")

    B, C, H, W = padded_batch.shape
    if (H, W) != (target_h, resized_size):
        raise ValueError(f"입력 크기가 {target_h}x{resized_size}이어야 합니다. 현재: {H}x{W}")

    # 1) 중앙 crop (802 -> 550)
    y_off = (target_h - resized_size) // 2  # (802-550)//2 = 126
    cropped = padded_batch[:, :, y_off:y_off + resized_size, :]  # [B, C, 550, 550]

    # 2) 550 -> 512 resize
    restored = F.interpolate(
        cropped,
        size=(orig_size, orig_size),
        mode="bilinear",
        align_corners=False
    )  # [B, C, 512, 512]

    return restored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/2d_prior.pth")
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


    model = Difix(
        pretrained_path=args.model_path,
        timestep=199,
        mv_unet=True
    )

    model.set_eval()

    to_tensor = transforms.ToTensor()

    # Load input images
    if os.path.isdir(args.input_image):
        print("Searching in:", os.path.join(args.input_image, "*rdr.png"))
        input_files = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
    else:
        input_files = [args.input_image]

    print(f"Found {len(input_files)} input images")

    input_tensor = [to_tensor(Image.open(f).convert("RGB")) for f in input_files]
    input_tensor = torch.stack(input_tensor)

    _, _, H, W = input_tensor.shape
    # Load reference images if provided
    ref_tensor = []
    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_files = sorted(glob(os.path.join(args.ref_image, "*ref.png")))
        else:
            ref_files = [args.ref_image]
        print(f"Found {len(ref_files)} reference images")
        ref_tensor = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in ref_files])

    _, _, H_ref, W_ref = ref_tensor.shape
    is_need_resize = False
    if (H_ref != H) or (W_ref != W):
        if H_ref == 512 and W_ref == 512:
            is_need_resize = True
            ref_tensor = resize_and_vertical_pad(ref_tensor, target_w=W, target_h=H)

    print("input shape: ", input_tensor.shape, ref_tensor.shape)

    # input: [B C H W], ref: [1(V), C, H, W]
    outputs = model.sample_batch_multi_tensor(
        image=input_tensor,
        ref_image=ref_tensor,
        batch_size=args.batch_size,
    )

    # output: [B C H W]
    print(f"Output shape: {outputs.shape}")  # [B, C, H, W]
    os.makedirs(args.save_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()

    for idx in range(outputs.shape[0]):
        img = to_pil(outputs[idx].cpu().clamp(0, 1))
        save_name = os.path.basename(input_files[idx]).replace('rdr', 'fix')
        save_path = os.path.join(args.save_dir, save_name)
        img.save(save_path)

    print(f"Saved {outputs.shape[0]} images to {args.save_dir}")