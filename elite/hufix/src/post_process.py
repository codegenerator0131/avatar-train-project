# from inference_human_difix import inference_on_tensors
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
# from torchvision.utils import save_image
SEED = 42


def resize_and_vertical_pad(img_tensor, target_w=550, target_h=802):
    """
    1) 이미지를 (target_w, target_w) 정방형으로 리사이즈
    2) 세로 방향만 padding 하여 (target_h, target_w)로 맞춤
    - img_tensor: torch.Tensor, [B, C, H, W] (0~1 범위)
    """
    # if img_tensor.ndim != 3 or img_tensor.shape[0] not in [1, 3]:
    #     raise ValueError("입력은 [C, H, W] 텐서여야 합니다.")

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


def restore_from_vertical_pad(
    padded_batch: torch.Tensor,
    orig_h: int, orig_w: int,     # 복원하고 싶은 원본 크기 (예: 512, 512)
    target_h: int, target_w: int  # 패딩/리사이즈 후의 현재 크기 (outputs의 H, W)
):
    """
    (target_w x target_h) 패딩 이미지를
    1) 세로 중앙에서 target_w 높이만큼 crop -> (target_w x target_w)
    2) (target_w x target_w) -> (orig_h x orig_w)로 리사이즈
    """
    if padded_batch.ndim != 4:
        raise ValueError("입력은 [B, C, H, W] 배치 텐서여야 합니다.")
    B, C, H, W = padded_batch.shape
    if (H, W) != (target_h, target_w):
        raise ValueError(f"입력 크기는 {target_h}x{target_w}이어야 합니다. 현재: {H}x{W}")

    if target_h < target_w:
        raise ValueError("세로 패딩을 가정하므로 target_h가 target_w 이상이어야 합니다.")

    # 1) 세로 중앙 crop: H -> target_w
    y_off = (target_h - target_w) // 2
    cropped = padded_batch[:, :, y_off:y_off + target_w, :]  # [B, C, target_w, target_w]

    # 2) 원래 크기로 리사이즈
    restored = F.interpolate(
        cropped, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    return restored



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/2d_prior.pth")
    parser.add_argument('--input_rgb', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--input_nrm', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument('--save_dir_rgb', type=str, default=None)
    parser.add_argument('--save_dir_nrm', type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_fps", type=int, default=30)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    SAVE_NAME = args.save_dir_rgb.split('/')[-1]
    model = Difix(
        pretrained_path=args.model_path,
        timestep=199,
        mv_unet=True
    )

    model.set_eval()

    to_tensor = transforms.ToTensor()

    # Load input images
    if os.path.isdir(args.input_rgb):
        print("Searching in:", os.path.join(args.input_rgb, "*.png"))
        input_rgb_files = sorted(glob(os.path.join(args.input_rgb, "*.png")))
        input_nrm_files = sorted(glob(os.path.join(args.input_nrm, "*.png")))
    else:
        input_rgb_files = [args.input_rgb]
        input_nrm_files = [args.input_nrm]

    print(f"Found {len(input_rgb_files)} input images")

    input_tensor_rgb = [to_tensor(Image.open(f).convert("RGB")) for f in input_rgb_files]
    input_tensor_rgb = torch.stack(input_tensor_rgb)

    input_tensor_nrm = [to_tensor(Image.open(f).convert("RGB")) for f in input_nrm_files]
    input_tensor_nrm = torch.stack(input_tensor_nrm)

    _, _, H, W = input_tensor_rgb.shape
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

    print("input shape: ", input_tensor_rgb.shape, ref_tensor.shape)

    # input: [B C H W], ref: [1(V), C, H, W]
    outputs = model.sample_batch_multi_tensor(
        image=input_tensor_rgb,
        ref_image=ref_tensor,
        batch_size=args.batch_size,
    )

    outputs_nrm = input_tensor_nrm

    # output: [B C H W]
    print(f"Output shape: {outputs.shape}")  # [B, C, H, W]
    os.makedirs(args.save_dir_rgb, exist_ok=True)
    os.makedirs(args.save_dir_nrm, exist_ok=True)

    to_pil = transforms.ToPILImage()

    for idx in range(outputs.shape[0]):
        img = to_pil(outputs[idx].cpu().clamp(0, 1))
        save_name = os.path.basename(input_rgb_files[idx])
        save_path = os.path.join(args.save_dir_rgb, save_name)
        img.save(save_path)

    for idx in range(outputs_nrm.shape[0]):
        img = to_pil(outputs_nrm[idx].cpu().clamp(0, 1))
        save_name = os.path.basename(input_nrm_files[idx])
        save_path = os.path.join(args.save_dir_nrm, save_name)
        img.save(save_path)

    rgb_video_path = os.path.join(args.save_dir_rgb, f'../../final_rgb_{SAVE_NAME}.mp4')
    cmd = f'ffmpeg -y -framerate {args.save_fps} -i {args.save_dir_rgb}/%05d.png -vf "scale={W}:{H}" -c:v libx264 -pix_fmt yuv420p -crf 20 {rgb_video_path}'
    os.system(cmd)

    nrm_video_path = os.path.join(args.save_dir_nrm, f'../../final_nrm_{SAVE_NAME}.mp4')
    cmd = f'ffmpeg -y -framerate {args.save_fps} -i {args.save_dir_nrm}/%05d.png -vf "scale={W}:{H}" -c:v libx264 -pix_fmt yuv420p -crf 20 {nrm_video_path}'
    os.system(cmd)

    print(f"HUFIX post-processing done ...")