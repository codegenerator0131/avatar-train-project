#!/usr/bin/env bash
# Stage 3: 3D Gaussian Avatar — ELITE personalization + rendering
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"
ELITE_PYTHON="$HOME/miniconda3/envs/ELITE/bin/python"

# --- Run setup if ELITE env is missing
if [ ! -f "$ELITE_PYTHON" ] || [ ! -f "$ELITE_DIR/src/render.py" ]; then
    echo "Setup incomplete — running setup first..."
    bash "$SCRIPT_DIR/setup_linux.sh"
    echo ""
fi

echo "================================================"
echo "  Stage 3: 3D Gaussian Avatar (ELITE)"
echo "================================================"
echo ""

# --- Check checkpoints
if [ ! -f "$ELITE_DIR/checkpoints/3d_prior.pth" ] || [ ! -f "$ELITE_DIR/checkpoints/2d_prior.pth" ]; then
    echo ""
    echo "ERROR: ELITE checkpoints missing."
    echo "Download from:"
    echo "  https://drive.google.com/drive/folders/1GKVymlwRi9shK0G2Qi5JrOFfkIdyUaHM"
    echo "Place at:"
    echo "  $ELITE_DIR/checkpoints/3d_prior.pth"
    echo "  $ELITE_DIR/checkpoints/2d_prior.pth"
    echo ""
    exit 1
fi

# --- Inputs
DEFAULT_VIDEO=""
for f in "$SCRIPT_DIR/data/capture/IMG_9625.mov" "$SCRIPT_DIR/data/capture/IMG_9625.MOV"; do
    [ -f "$f" ] && DEFAULT_VIDEO="$f" && break
done
read -e -i "${DEFAULT_VIDEO:-$SCRIPT_DIR/data/capture/IMG_9625.mov}" -p "Input video path: " VIDEO
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

read -e -i "0" -p "GPU index: " GPU
export CUDA_VISIBLE_DEVICES=$GPU

# Derive ID from video filename stem (same as Stage 2)
ID="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
echo ""
echo "Person ID: $ID"
echo ""

# --- Paths (mirrors ELITE's configs/paths.sh layout)
SOURCE_VIDEO_DIR="$ELITE_DIR/data/source/input_videos"
TRACKED_DIR="$ELITE_DIR/data/source/tracked"
PROCESSED_DIR="$ELITE_DIR/data/source/processed"
PROCESSED_SUFFIX="_whiteBg_staticOffset_maskBelowLine"
STAGE2_OUTPUT="$SCRIPT_DIR/data/processed/${ID}_nerf"

mkdir -p "$SOURCE_VIDEO_DIR" "$TRACKED_DIR" "$PROCESSED_DIR" \
         "$ELITE_DIR/checkpoints" "$ELITE_DIR/outputs" "$ELITE_DIR/logs"

# Export env vars that ELITE scripts expect
export CONDA_ENV="ELITE"
export CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export ELITE_CFG="$ELITE_DIR/configs/3d_prior.yaml"
export ELITE_CKPT="$ELITE_DIR/checkpoints/3d_prior.pth"
export HUFIX_CKPT="$ELITE_DIR/checkpoints/2d_prior.pth"
export SINGLEVIEW_PRC_ROOT="$PROCESSED_DIR"
export SINGLEVIEW_TRACKED_ROOT="$TRACKED_DIR"
export EXP_ROOT="$ELITE_DIR/outputs"
export LOG_ROOT="$ELITE_DIR/logs"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PYTHONPATH="$ELITE_DIR:${PYTHONPATH:-}"

echo "Step 1/3 — Preparing video input for ELITE..."
echo ""
# Symlink video into ELITE's expected location
mkdir -p "$SOURCE_VIDEO_DIR"
TARGET_VIDEO="$SOURCE_VIDEO_DIR/${ID}.mp4"
if [ ! -f "$TARGET_VIDEO" ]; then
    # Convert MOV → MP4 if needed, otherwise symlink
    EXT="${VIDEO##*.}"
    if [ "${EXT,,}" = "mp4" ]; then
        ln -sf "$VIDEO" "$TARGET_VIDEO"
        echo "  Symlinked $VIDEO → $TARGET_VIDEO"
    else
        echo "  Converting $VIDEO → $TARGET_VIDEO..."
        ffmpeg -i "$VIDEO" -c:v copy -c:a aac "$TARGET_VIDEO" -y
    fi
else
    echo "  [skip] Video already at $TARGET_VIDEO"
fi

# --- Reuse Stage 2 VHAP output (both tracked + processed)
PROCESSED_TARGET="$PROCESSED_DIR/${ID}${PROCESSED_SUFFIX}"
TRACKED_TARGET="$TRACKED_DIR/${ID}_whiteBg_staticOffset"
STAGE2_TRACKED="$SCRIPT_DIR/data/processed/${ID}_vhap"

mkdir -p "$PROCESSED_DIR" "$TRACKED_DIR"

if [ -d "$STAGE2_OUTPUT" ] && [ ! -d "$PROCESSED_TARGET" ]; then
    echo "  Reusing Stage 2 processed output..."
    ln -sfn "$STAGE2_OUTPUT" "$PROCESSED_TARGET"
    echo "  Symlinked $STAGE2_OUTPUT → $PROCESSED_TARGET"
fi

if [ -d "$STAGE2_TRACKED" ] && [ ! -d "$TRACKED_TARGET" ]; then
    echo "  Reusing Stage 2 tracked output..."
    ln -sfn "$STAGE2_TRACKED" "$TRACKED_TARGET"
    echo "  Symlinked $STAGE2_TRACKED → $TRACKED_TARGET"
fi

echo ""
echo "Step 2/3 — VHAP preprocessing + tracking (skipped if already done)..."
echo ""
TRACKED_TARGET="$TRACKED_DIR/${ID}_whiteBg_staticOffset"
if [ -d "$PROCESSED_TARGET" ]; then
    echo "  [skip] Processed data already exists at $PROCESSED_TARGET"
else
    cd "$ELITE_DIR"
    "$ELITE_PYTHON" vhap/vhap/preprocess_video.py \
        --input "$TARGET_VIDEO" \
        --matting_method robust_video_matting

    "$ELITE_PYTHON" vhap/vhap/track.py \
        --data.root_folder "$SOURCE_VIDEO_DIR" \
        --exp.output_folder "$TRACKED_TARGET" \
        --data.sequence "$ID" \
        --data.scale-factor 0.5 \
        --batch-size 4

    "$ELITE_PYTHON" vhap/vhap/export_as_nerf_dataset.py \
        --src_folder "$TRACKED_TARGET" \
        --tgt_folder "$PROCESSED_TARGET"
fi

echo ""
echo "Step 3/3 — Personalizing ELITE avatar (stage 1)..."
echo ""
cd "$ELITE_DIR"

# Ensure elite/asset/flame/ has the FLAME files
mkdir -p "$ELITE_DIR/asset/flame"
ln -sf "$SCRIPT_DIR/asset/flame/flame2023.pkl" "$ELITE_DIR/asset/flame/flame2023.pkl"
ln -sf "$SCRIPT_DIR/asset/flame/FLAME_masks.pkl" "$ELITE_DIR/asset/flame/FLAME_masks.pkl"
ln -sf "$SCRIPT_DIR/asset/flame/mediapipe_landmark_embedding.npz" "$ELITE_DIR/asset/flame/mediapipe_landmark_embedding.npz" 2>/dev/null || true
# Copy any other files from asset/flame that ELITE may need
for f in "$SCRIPT_DIR/asset/flame/"*; do
    fname="$(basename "$f")"
    [ ! -e "$ELITE_DIR/asset/flame/$fname" ] && ln -sf "$f" "$ELITE_DIR/asset/flame/$fname" || true
done
EXP_PATH="$EXP_ROOT/$ID"
mkdir -p "$EXP_PATH"

# ELITE only supports 512x512 or 802x550 — auto-resize if needed
IMG_RES="512x512"
echo "  Checking image resolution..."
"$ELITE_PYTHON" - <<PYEOF
import os, struct
from PIL import Image

processed = "$PROCESSED_TARGET"
for subdir in ["images", "fg_masks"]:
    folder = os.path.join(processed, subdir)
    if not os.path.exists(folder):
        continue
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if not files:
        continue
    sample = Image.open(os.path.join(folder, files[0]))
    w, h = sample.size
    if (w, h) == (512, 512) or (w, h) == (802, 550):
        print(f"  [{subdir}] {w}x{h} — OK")
        continue
    print(f"  [{subdir}] {w}x{h} — resizing to 512x512...")
    for fname in files:
        path = os.path.join(folder, fname)
        img = Image.open(path).resize((512, 512), Image.LANCZOS)
        img.save(path)
    print(f"  [{subdir}] resize done ({len(files)} files)")
PYEOF
echo "  Using resolution: $IMG_RES"

# Patch transforms.json: add "f" prefix to timestep_id so ELITE's ts[1:] gives correct filename
echo "  Patching transforms.json timestep_id format..."
"$ELITE_PYTHON" - <<PYEOF
import json, os

for fname in ["transforms.json", "transforms_train.json", "transforms_val.json"]:
    path = os.path.join("$PROCESSED_TARGET", fname)
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    changed = False
    for frame in data.get("frames", []):
        ts = frame.get("timestep_id", "")
        if ts and not ts.startswith("f"):
            frame["timestep_id"] = "f" + ts
            changed = True
    if changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Patched {fname}")
    else:
        print(f"  {fname} already OK")
PYEOF

# Normalize image filenames to match transforms.json timestep_id format (6-digit, no prefix)
echo "  Checking image filename format..."
"$ELITE_PYTHON" - <<PYEOF
import os, re

processed = "$PROCESSED_TARGET"
for subdir in ["images", "fg_masks"]:
    folder = os.path.join(processed, subdir)
    if not os.path.exists(folder):
        continue
    files = sorted(os.listdir(folder))
    for fname in files:
        # Revert f000000_00.png → 000000_00.png
        m = re.match(r'^f(\d{6})(_\d{2}\.png)$', fname)
        if m:
            new_name = f"{m.group(1)}{m.group(2)}"
            os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))
            continue
        # Rename 00000_00.png → 000000_00.png (5-digit to 6-digit)
        m = re.match(r'^(\d{5})(_\d{2}\.png)$', fname)
        if m:
            new_name = f"{int(m.group(1)):06d}{m.group(2)}"
            os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))
    print(f"  [{subdir}] filenames normalized")
PYEOF

# Stage 1: personalize with real images
"$ELITE_PYTHON" src/personalize.py \
    --stage 1 \
    --exp_path "$EXP_PATH" \
    --tgt_id "$ID" \
    --prior_cfg "$ELITE_CFG" \
    --prior_ckpt "$ELITE_CKPT" \
    --res "$IMG_RES" \
    --singleview_bs 1

echo ""
echo "Step 3/3 — Personalizing ELITE avatar (stage 2 fine-tuning)..."
echo ""
# Stage 2: joint finetuning with synthetic images
"$ELITE_PYTHON" src/personalize.py \
    --stage 2 \
    --exp_path "$EXP_PATH" \
    --tgt_id "$ID" \
    --prior_cfg "$ELITE_CFG" \
    --prior_ckpt "$ELITE_CKPT" \
    --res "$IMG_RES" \
    --singleview_bs 1

echo ""
echo "================================================"
echo "  Stage 3 complete!"
echo "================================================"
echo ""
echo "Avatar trained for: $ID"
echo "Output: $ELITE_DIR/outputs/$ID/"
echo ""
echo "To render with a motion sequence:"
echo "  cd $ELITE_DIR"
echo "  bash scripts/render_videos.sh $ID <motion_name>"
echo ""
echo "Next: bash start.sh  (choose stage 4)"
