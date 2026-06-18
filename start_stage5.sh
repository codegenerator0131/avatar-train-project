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

# Remove old broken drive dir if it has 0 frames
if [ -f "$DRIVE_DIR/transforms.json" ]; then
    FRAME_COUNT=$("$ELITE_PYTHON" -c "import json; d=json.load(open('$DRIVE_DIR/transforms.json')); print(len(d['frames']))" 2>/dev/null || echo "0")
    if [ "$FRAME_COUNT" = "0" ]; then
        echo "  Removing empty drive dir..."
        rm -rf "$DRIVE_DIR"
    fi
fi
mkdir -p "$DRIVE_DIR"

"$ELITE_PYTHON" - <<PYEOF
import os, json, numpy as np

processed = "$PROCESSED_TARGET"
drive_dir = "$DRIVE_DIR"

# Load transforms.json — already has flame_param_path per frame
tf_path = os.path.join(processed, 'transforms.json')
with open(tf_path) as f:
    transforms_data = json.load(f)

frames = transforms_data['frames']
print(f"  transforms.json has {len(frames)} frames")

# Patch camera_id to the string TestMotionRenderDataset expects
# Our data uses integer 0; we remap to our own cam id string
CAM_ID = 'our_cam'
for frame in frames:
    frame['camera_id'] = CAM_ID
    # ensure flame_param_path is absolute or resolvable from drive_dir
    # it should already point to flame_param/ relative path

# Write patched transforms.json into drive_dir
drive_tf = dict(transforms_data)
drive_tf['frames'] = frames
out_path = os.path.join(drive_dir, 'transforms.json')
with open(out_path, 'w') as f:
    json.dump(drive_tf, f, indent=2)

# Symlink images, fg_masks, flame_param into drive_dir
for subdir in ['images', 'fg_masks', 'flame_param']:
    src = os.path.join(processed, subdir)
    dst = os.path.join(drive_dir, subdir)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"  Symlinked {subdir}")
    elif os.path.exists(dst):
        print(f"  {subdir} already linked")
    else:
        print(f"  WARNING: {src} not found")

print(f"  Drive data ready: {len(frames)} frames → {drive_dir}")
print(f"  Camera ID set to: {CAM_ID}")
PYEOF

# Pass the cam_id we set above to render.py via cam_ids arg
CAM_ID="our_cam"

echo ""
echo "Step 2/3 — Rendering avatar with ELITE renderer..."
echo ""

cd "$ELITE_DIR"
mkdir -p "$VIS_DIR"

MOTION_NAME="$ID"
SAVE_PATH_RGB="$VIS_DIR/st2_rgb/${MOTION_NAME}"
SAVE_PATH_NRM="$VIS_DIR/st2_nrm/${MOTION_NAME}"
mkdir -p "$SAVE_PATH_RGB" "$SAVE_PATH_NRM" "$VIS_DIR/vid_drive/${MOTION_NAME}"

# Patch render.py to use our cam_id (default is '222200037' which doesn't match our data)
"$ELITE_PYTHON" - <<PYEOF2
path = "$ELITE_DIR/src/render.py"
with open(path) as f:
    code = f.read()
patched = code.replace(
    "cam_ids=None,",
    "cam_ids=['$CAM_ID'],",
)
if patched != code:
    with open(path, 'w') as f:
        f.write(patched)
    print("  Patched render.py: cam_ids=['$CAM_ID']")
else:
    print("  render.py already patched or cam_ids not found — checking...")
    # Also try patching the default in TestMotionRenderDataset call
    import re
    patched2 = re.sub(
        r"cam_ids=\[.*?\]",
        "cam_ids=['$CAM_ID']",
        code,
    )
    if patched2 != code:
        with open(path, 'w') as f:
            f.write(patched2)
        print("  Patched render.py via regex")
PYEOF2

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
