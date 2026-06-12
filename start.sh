#!/usr/bin/env bash
# Avatar Pipeline — one-shot launcher
# Usage: bash start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/activate"

echo "================================================"
echo "  Avatar Pipeline — Stage 1: Capture"
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

# --- video file
while true; do
    read -rp "Video file path (e.g. data/capture/IMG_9625.MOV): " VIDEO
    if [ -f "$VIDEO" ]; then
        break
    fi
    # Try relative to script dir
    if [ -f "$SCRIPT_DIR/$VIDEO" ]; then
        VIDEO="$SCRIPT_DIR/$VIDEO"
        break
    fi
    echo "  ERROR: File not found: $VIDEO — try again."
done

# --- dataset name
read -rp "Dataset name [take1]: " NAME
NAME="${NAME:-take1}"

# --- output dir (fixed default, no need to change usually)
OUT="$SCRIPT_DIR/data/processed"

# --- size
read -rp "Crop size in pixels [512]: " SIZE
SIZE="${SIZE:-512}"

# --- fps
read -rp "FPS [30]: " FPS
FPS="${FPS:-30}"

echo ""
echo "Starting capture..."
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
echo "Next steps:"
echo "  1. Open a few frames in $OUT/$NAME/frames/ and check the full"
echo "     head (hair, chin, ears) is inside every crop."
echo "  2. If anything is clipped, re-run and increase the crop size or"
echo "     edit capture.py --margin (default 0.45, try 0.55)."
echo "  3. Hand the $OUT/$NAME/ folder to Stage 2 when ready."
