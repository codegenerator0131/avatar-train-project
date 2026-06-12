#!/usr/bin/env bash
# Avatar Pipeline — interactive launcher (Stage 1 + Stage 2)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"

echo "================================================"
echo "  Avatar Pipeline"
echo "================================================"
echo ""

# --- Run setup if venv doesn't exist yet
if [ ! -f "$VENV" ]; then
    echo "First run detected — running setup first..."
    echo ""
    bash "$SCRIPT_DIR/setup_linux.sh"
    echo ""
fi

source "$VENV"

# --- Choose stage
echo "Which stage do you want to run?"
echo "  1) Stage 1 — Capture   (video → frames + audio)"
echo "  2) Stage 2 — Tracking  (frames → FLAME params per frame)"
echo ""
read -rp "Enter stage [1]: " STAGE
STAGE="${STAGE:-1}"

echo ""

# ===========================================================================
if [ "$STAGE" = "1" ]; then
# ===========================================================================

    # --- video file
    while true; do
        read -rp "Video file path (e.g. data/capture/IMG_9625.MOV): " VIDEO
        [ -f "$VIDEO" ] && break
        [ -f "$SCRIPT_DIR/$VIDEO" ] && VIDEO="$SCRIPT_DIR/$VIDEO" && break
        echo "  ERROR: File not found: $VIDEO — try again."
    done

    read -rp "Dataset name [take1]: " NAME
    NAME="${NAME:-take1}"

    OUT="$SCRIPT_DIR/data/processed"

    read -rp "Crop size in pixels [512]: " SIZE
    SIZE="${SIZE:-512}"

    read -rp "FPS [30]: " FPS
    FPS="${FPS:-30}"

    echo ""
    echo "Running Stage 1: Capture..."
    echo ""

    python "$SCRIPT_DIR/capture.py" \
        --video "$VIDEO" \
        --name  "$NAME" \
        --out   "$OUT" \
        --size  "$SIZE" \
        --fps   "$FPS"

    echo ""
    echo "Done! Output saved to: $OUT/$NAME/"
    echo ""
    echo "Next: open a few frames in $OUT/$NAME/frames/ and confirm"
    echo "the full head (hair, chin, ears) is inside every crop."
    echo "Then run this script again and choose Stage 2."

# ===========================================================================
elif [ "$STAGE" = "2" ]; then
# ===========================================================================

    # --- dataset
    read -rp "Dataset path [data/processed/take1]: " DATASET
    DATASET="${DATASET:-data/processed/take1}"
    if [ ! -d "$DATASET" ]; then
        DATASET="$SCRIPT_DIR/$DATASET"
    fi
    if [ ! -d "$DATASET" ]; then
        echo "ERROR: Dataset not found: $DATASET"
        echo "Run Stage 1 first."
        exit 1
    fi

    # --- FLAME model
    FLAME_DEFAULT="$SCRIPT_DIR/data/flame/flame2023.pkl"
    read -rp "FLAME model path [$FLAME_DEFAULT]: " FLAME_PATH
    FLAME_PATH="${FLAME_PATH:-$FLAME_DEFAULT}"
    if [ ! -f "$FLAME_PATH" ]; then
        echo ""
        echo "ERROR: FLAME model not found at: $FLAME_PATH"
        echo ""
        echo "Download it free from: https://flame.is.tue.mpg.de"
        echo "  1. Register / log in"
        echo "  2. Download FLAME 2023"
        echo "  3. Place flame2023.pkl at: $FLAME_DEFAULT"
        echo "     mkdir -p $SCRIPT_DIR/data/flame"
        exit 1
    fi

    read -rp "Optimizer iterations per frame [300]: " ITERS
    ITERS="${ITERS:-300}"

    read -rp "Max frames to process (0 = all) [0]: " MAX_FRAMES
    MAX_FRAMES="${MAX_FRAMES:-0}"

    echo ""
    echo "Running Stage 2: Tracking..."
    echo ""

    python "$SCRIPT_DIR/track.py" \
        --dataset     "$DATASET" \
        --flame       "$FLAME_PATH" \
        --device      cuda \
        --iters       "$ITERS" \
        --max-frames  "$MAX_FRAMES"

    echo ""
    echo "Done! Tracking saved to: $DATASET/tracking_smoothed.json"
    echo "This file feeds into Stage 3 (splat training)."

# ===========================================================================
else
    echo "Unknown stage: $STAGE. Enter 1 or 2."
    exit 1
fi
