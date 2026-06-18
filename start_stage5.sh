#!/usr/bin/env bash
# Stage 5: Render talking-head video using trained ELITE avatar
set -euo pipefail
trap 'echo ""; echo "ERROR: Stage 5 failed at line $LINENO. Command: $BASH_COMMAND"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"
ELITE_PYTHON="$HOME/miniconda3/envs/ELITE/bin/python"

echo "================================================"
echo "  Stage 5: Render Avatar Video (ELITE)"
echo "================================================"
echo ""

# --- Inputs
DEFAULT_VIDEO=""
for f in "$SCRIPT_DIR/data/capture/IMG_9625.mov" "$SCRIPT_DIR/data/capture/IMG_9625.MOV"; do
    [ -f "$f" ] && DEFAULT_VIDEO="$f" && break
done
read -e -i "${DEFAULT_VIDEO:-$SCRIPT_DIR/data/capture/IMG_9625.mov}" -p "Input video path (driver): " VIDEO
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

read -e -i "0" -p "GPU index: " GPU
export CUDA_VISIBLE_DEVICES=$GPU

ID="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
echo ""
echo "Person ID: $ID"
echo ""

# --- Paths
PROCESSED_DIR="$ELITE_DIR/data/source/processed"
TRACKED_DIR="$ELITE_DIR/data/source/tracked"
PROCESSED_SUFFIX="_whiteBg_staticOffset_maskBelowLine"
PROCESSED_TARGET="$PROCESSED_DIR/${ID}${PROCESSED_SUFFIX}"
TRACKED_TARGET="$TRACKED_DIR/${ID}_whiteBg_staticOffset"
EXP_PATH="$ELITE_DIR/outputs/$ID"
ST2_CKPT="$EXP_PATH/st2/checkpoints/st2_final.pth"
DRIVE_DIR="$ELITE_DIR/data/drive/${ID}"
VIS_DIR="$EXP_PATH/vis_motion"

export CONDA_ENV="ELITE"
export CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export ELITE_CFG="$ELITE_DIR/configs/3d_prior.yaml"
export HUFIX_CKPT="$ELITE_DIR/checkpoints/2d_prior.pth"
export SINGLEVIEW_PRC_ROOT="$PROCESSED_DIR"
export SINGLEVIEW_TRACKED_ROOT="$TRACKED_DIR"
export EXP_ROOT="$ELITE_DIR/outputs"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PYTHONPATH="$ELITE_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled

# --- Check Stage 3 complete
if [ ! -f "$ST2_CKPT" ]; then
    echo "ERROR: Stage 3 not complete. Missing: $ST2_CKPT"
    echo "Run bash start_stage3.sh first."
    exit 1
fi

echo "Step 1/3 — Preparing motion drive data from Stage 2 VHAP output..."
echo ""

# Build drive directory from VHAP tracked output
mkdir -p "$DRIVE_DIR/flame_param"

"$ELITE_PYTHON" - <<PYEOF
import os, json, numpy as np, glob, shutil
from PIL import Image
import torchvision.transforms as T
import torch

processed = "$PROCESSED_TARGET"
tracked   = "$TRACKED_TARGET"
drive_dir = "$DRIVE_DIR"

# Find tracked_flame_params npz
last_run   = sorted(os.listdir(tracked))[-1]
npz_path   = os.path.join(tracked, last_run, 'tracked_flame_params_30.npz')
params     = np.load(npz_path, allow_pickle=True)
timestep_ids = params['timestep_id']  # e.g. ['f000000', ...]

print(f"  Found {len(timestep_ids)} frames in {npz_path}")

# Load transforms.json from processed dir
tf_path = os.path.join(processed, 'transforms.json')
with open(tf_path) as f:
    transforms_data = json.load(f)

frames_by_ts = {}
for frame in transforms_data['frames']:
    ts = frame.get('timestep_id', '')
    if ts not in frames_by_ts:
        frames_by_ts[ts] = frame

print(f"  transforms.json has {len(frames_by_ts)} unique timesteps")

# Build per-frame npz files and new transforms.json for drive dir
to_tensor = T.ToTensor()
new_frames = []
written = 0

for i, ts in enumerate(timestep_ids):
    ts_str = str(ts)
    if ts_str not in frames_by_ts:
        continue

    frame_entry = frames_by_ts[ts_str]
    img_fname   = frame_entry.get('file_path', '')   # e.g. images/000000_00.png
    mask_fname  = frame_entry.get('fg_mask_path', frame_entry.get('fg_masks', ''))

    img_src  = os.path.join(processed, img_fname)
    if not mask_fname:
        # try to guess
        mask_src = img_src.replace('/images/', '/fg_masks/')
    else:
        mask_src = os.path.join(processed, mask_fname)

    if not os.path.isfile(img_src):
        continue

    # Write per-frame flame_param npz
    npz_out = os.path.join(drive_dir, 'flame_param', f'{written:05d}.npz')
    np.savez(npz_out,
        shape       = params['shape'],
        rotation    = params['rotation'][i:i+1],
        translation = params['translation'][i:i+1],
        neck_pose   = params['neck_pose'][i:i+1],
        jaw_pose    = params['jaw_pose'][i:i+1],
        eyes_pose   = params['eyes_pose'][i:i+1],
        expr        = params['expr'][i:i+1],
        static_offset = params['static_offset'][i:i+1] if 'static_offset' in params else np.zeros((1, params['shape'].shape[-1] if 'shape' in params else 5023, 3), dtype=np.float32),
    )

    # Build frame entry for drive transforms.json
    new_entry = {k: v for k, v in frame_entry.items()}
    new_entry['timestep_index']  = written
    new_entry['file_path']       = img_fname
    new_entry['fg_mask_path']    = mask_fname if mask_fname else img_fname.replace('images/', 'fg_masks/'),
    new_entry['flame_param_path'] = f'flame_param/{written:05d}.npz'
    new_entry['camera_id']       = '222200037'
    new_frames.append(new_entry)
    written += 1

# Write drive transforms.json
drive_tf = {k: v for k, v in transforms_data.items() if k != 'frames'}
drive_tf['frames'] = new_frames
with open(os.path.join(drive_dir, 'transforms.json'), 'w') as f:
    json.dump(drive_tf, f, indent=2)

# Symlink images and fg_masks into drive_dir
for subdir in ['images', 'fg_masks']:
    src = os.path.join(processed, subdir)
    dst = os.path.join(drive_dir, subdir)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)

print(f"  Drive data prepared: {written} frames → {drive_dir}")
PYEOF

echo ""
echo "Step 2/3 — Rendering avatar with ELITE renderer..."
echo ""

cd "$ELITE_DIR"
mkdir -p "$VIS_DIR"

MOTION_NAME="$ID"
SAVE_PATH_RGB="$VIS_DIR/st2_rgb/${MOTION_NAME}"
SAVE_PATH_NRM="$VIS_DIR/st2_nrm/${MOTION_NAME}"
mkdir -p "$SAVE_PATH_RGB" "$SAVE_PATH_NRM" "$VIS_DIR/vid_drive/${MOTION_NAME}"

"$ELITE_PYTHON" src/render.py \
    --cfg_file "$EXP_PATH/st2/config.yaml" \
    --ckpt_file "$ST2_CKPT" \
    --person_id "$ID" \
    --motion_samples_dir "$DRIVE_DIR" \
    --stage '2' \
    --cam_path 'motion_id' \
    --save_fps 25 \
    --motion_id "$MOTION_NAME"

echo ""
echo "Step 3/3 — Post-processing with HuFix enhancement..."
echo ""

REF_IMAGE="$PROCESSED_TARGET/images/00000_00.png"
# fallback ref image if naming differs
if [ ! -f "$REF_IMAGE" ]; then
    REF_IMAGE=$(find "$PROCESSED_TARGET/images" -name "*.png" | sort | head -1)
fi

"$ELITE_PYTHON" hufix/src/post_process.py \
    --ref_image "$REF_IMAGE" \
    --input_rgb "$SAVE_PATH_RGB" \
    --input_nrm "$SAVE_PATH_NRM" \
    --save_dir_rgb "${SAVE_PATH_RGB}_difix" \
    --save_dir_nrm "${SAVE_PATH_NRM}_difix" \
    --save_fps 25 \
    --model_path "$HUFIX_CKPT"

# Rename final outputs
FINAL_RGB="$VIS_DIR/${ID}_rgb_${MOTION_NAME}.mp4"
FINAL_NRM="$VIS_DIR/${ID}_nrm_${MOTION_NAME}.mp4"
[ -f "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" "$FINAL_RGB"
[ -f "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" "$FINAL_NRM"

# Cleanup intermediate frames
rm -rf "$SAVE_PATH_RGB" "$SAVE_PATH_NRM" "$VIS_DIR/vid_drive/${MOTION_NAME}"

echo ""
echo "================================================"
echo "  Stage 5 complete!"
echo "================================================"
echo ""
echo "Output videos:"
echo "  RGB: $FINAL_RGB"
echo "  Normal: $FINAL_NRM"
echo ""
echo "Next: bash start.sh  (choose stage 6 or deliver to client)"
