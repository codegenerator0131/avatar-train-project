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

# --- Use vhap env python directly (conda activate unreliable in subshells)
VHAP_PYTHON="$HOME/miniconda3/envs/vhap/bin/python"
if [ ! -f "$VHAP_PYTHON" ]; then
    echo "ERROR: vhap conda env not found. Run: bash setup_linux.sh"
    exit 1
fi

echo "================================================"
echo "  Stage 2: FLAME Tracking (VHAP)"
echo "================================================"
echo ""

# --- Inputs
# Find whichever capitalisation exists
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

VHAP_DIR="$SCRIPT_DIR/vhap"
# Sequence name = video filename without extension (VHAP preprocess creates a dir with this name)
SEQUENCE="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
CAPTURE_DIR="$(dirname "$VIDEO")"
TRACKED_DIR="$SCRIPT_DIR/data/processed/${SEQUENCE}_vhap"
NERF_DIR="$SCRIPT_DIR/data/processed/${SEQUENCE}_nerf"

echo ""
echo "Step 1/3 — Preprocessing video (frame extraction + background matting)..."
echo ""
if [ -d "$CAPTURE_DIR/$SEQUENCE/alpha_maps" ]; then
    echo "  [skip] Already preprocessed: $CAPTURE_DIR/$SEQUENCE/"
else
    "$VHAP_PYTHON" "$VHAP_DIR/vhap/preprocess_video.py" \
        --input "$VIDEO" \
        --matting_method robust_video_matting
fi

echo ""
echo "Step 2/3 — Photometric FLAME tracking..."
echo ""
"$VHAP_PYTHON" "$VHAP_DIR/vhap/track.py" \
    --data.root_folder "$CAPTURE_DIR" \
    --exp.output_folder "$TRACKED_DIR" \
    --data.sequence "$SEQUENCE" \
    --data.scale-factor 0.5 \
    --batch-size 4

echo ""
echo "Step 3/3 — Exporting as NeRF/3DGS dataset..."
echo ""
"$VHAP_PYTHON" "$VHAP_DIR/vhap/export_as_nerf_dataset.py" \
    --src_folder "$TRACKED_DIR" \
    --tgt_folder "$NERF_DIR"

# --- Extract audio from original video
AUDIO_OUT="$SCRIPT_DIR/data/capture/${SEQUENCE}.wav"
echo ""
echo "Extracting audio from video..."
if [ -f "$AUDIO_OUT" ]; then
    echo "  [skip] Audio already extracted: $AUDIO_OUT"
else
    ffmpeg -i "$VIDEO" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_OUT" -y
    echo "  Audio saved: $AUDIO_OUT"
fi

echo ""
echo "================================================"
echo "  Stage 2 complete!"
echo "================================================"
echo ""
echo "Sequence: $SEQUENCE"
echo "Output: $NERF_DIR"
echo "  transforms.json     — camera poses per frame"
echo "  images/             — matted frames (white background)"
echo "  fg_masks/           — foreground alpha masks"
echo "  flame_param/        — per-frame FLAME parameters (.npz)"
echo "  canonical.obj       — FLAME neutral mesh"
echo "Audio: $AUDIO_OUT (16kHz mono WAV for Stage 4)"
echo ""
echo "Next: bash start.sh  (choose stage 3)"
