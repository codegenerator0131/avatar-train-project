#!/usr/bin/env bash
# Stage 2: Tracking — frames → FLAME params per frame
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"

if [ ! -f "$VENV" ]; then
    echo "Venv not found — running setup first..."
    bash "$SCRIPT_DIR/setup_linux.sh"
fi
source "$VENV"

echo "================================================"
echo "  Stage 2: FLAME Tracking"
echo "================================================"
echo ""

# --- Dataset
read -e -i "data/processed/take1" -p "Dataset path: " DATASET
[ -d "$DATASET" ] || DATASET="$SCRIPT_DIR/$DATASET"
if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    echo "Run Stage 1 first: bash start_stage1.sh"
    exit 1
fi

# --- FLAME model
FLAME_DEFAULT="$SCRIPT_DIR/data/flame/flame2023.pkl"
read -e -i "$FLAME_DEFAULT" -p "FLAME model path: " FLAME_PATH
if [ ! -f "$FLAME_PATH" ]; then
    echo ""
    echo "ERROR: FLAME model not found at: $FLAME_PATH"
    echo ""
    echo "Download free from: https://flame.is.tue.mpg.de"
    echo "  1. Register / log in"
    echo "  2. Download FLAME 2023"
    echo "  3. Place flame2023.pkl at: $FLAME_DEFAULT"
    echo "     mkdir -p $SCRIPT_DIR/data/flame"
    exit 1
fi

LM_EMBED="$(dirname "$FLAME_PATH")/mediapipe_landmark_embedding.npz"
if [ ! -f "$LM_EMBED" ]; then
    echo ""
    echo "WARNING: mediapipe_landmark_embedding.npz not found at: $LM_EMBED"
    echo "Tracking will fall back to built-in landmarks (less accurate)."
    echo ""
    LM_EMBED=""
fi

# --- Options
read -e -i "300" -p "Optimizer iterations per frame: " ITERS
read -e -i "50"  -p "Max frames to process (0 = all): " MAX_FRAMES

echo ""
if [ "$MAX_FRAMES" != "0" ]; then
    echo "Running Stage 2 in TEST mode (first $MAX_FRAMES frames)..."
    echo "Check reprojection error at the end. If median < 10px, re-run with 0 for all frames."
else
    echo "Running Stage 2 on ALL frames..."
fi
echo ""

LM_ARGS=""
[ -n "$LM_EMBED" ] && LM_ARGS="--lm-embed $LM_EMBED"

python "$SCRIPT_DIR/track.py" \
    --dataset    "$DATASET" \
    --flame      "$FLAME_PATH" \
    $LM_ARGS \
    --device     cuda \
    --iters      "$ITERS" \
    --max-frames "$MAX_FRAMES"

echo ""
echo "Done! Tracking saved to: $DATASET/tracking_smoothed.json"
echo "This file feeds into Stage 3 (splat training): bash start_stage3.sh"
