#!/usr/bin/env bash
# Stage 2: FLAME Tracking via VHAP (photometric fitting → NeRF/3DGS dataset)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Run setup if venv or VHAP is missing
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ] || [ ! -f "$SCRIPT_DIR/vhap/vhap/preprocess_video.py" ]; then
    echo "Setup incomplete — running setup first..."
    bash "$SCRIPT_DIR/setup_linux.sh"
    echo ""
fi

# --- Initialize conda in this shell session
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found. Install Miniconda then re-run: bash setup_linux.sh"
    exit 1
}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate vhap

echo "================================================"
echo "  Stage 2: FLAME Tracking (VHAP)"
echo "================================================"
echo ""

# --- Inputs
read -e -i "data/capture/IMG_9625.MOV" -p "Input video path: " VIDEO
[ -f "$VIDEO" ] || VIDEO="$SCRIPT_DIR/$VIDEO"
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

read -e -i "take1" -p "Dataset name (ID): " ID
read -e -i "0"     -p "GPU index: " GPU
export CUDA_VISIBLE_DEVICES=$GPU

VHAP_DIR="$SCRIPT_DIR/vhap"
TRACKED_DIR="$SCRIPT_DIR/data/processed/${ID}_vhap"
NERF_DIR="$SCRIPT_DIR/data/processed/${ID}_nerf"

echo ""
echo "Step 1/3 — Preprocessing video (frame extraction + background matting)..."
echo ""
python "$VHAP_DIR/vhap/preprocess_video.py" \
    --input "$VIDEO" \
    --matting_method robust_video_matting

echo ""
echo "Step 2/3 — Photometric FLAME tracking..."
echo ""
python "$VHAP_DIR/vhap/track.py" \
    --data.root_folder "$SCRIPT_DIR/data/source/input_videos" \
    --exp.output_folder "$TRACKED_DIR" \
    --data.sequence "$ID"

echo ""
echo "Step 3/3 — Exporting as NeRF/3DGS dataset..."
echo ""
python "$VHAP_DIR/vhap/export_as_nerf_dataset.py" \
    --src_folder "$TRACKED_DIR" \
    --tgt_folder "$NERF_DIR"

echo ""
echo "================================================"
echo "  Stage 2 complete!"
echo "================================================"
echo ""
echo "Output: $NERF_DIR"
echo "  transforms.json     — camera poses per frame"
echo "  images/             — matted frames (white background)"
echo "  fg_masks/           — foreground alpha masks"
echo "  flame_param/        — per-frame FLAME parameters (.npz)"
echo "  canonical.obj       — FLAME neutral mesh"
echo ""
echo "Next: bash start.sh  (choose stage 3)"
