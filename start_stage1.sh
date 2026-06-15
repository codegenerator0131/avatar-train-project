#!/usr/bin/env bash
# Stage 1: Capture — video → frames + audio
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"

if [ ! -f "$VENV" ]; then
    echo "venv not found — running setup first..."
    bash "$SCRIPT_DIR/setup_linux.sh"
fi
source "$VENV"

echo "================================================"
echo "  Stage 1: Capture"
echo "================================================"
echo ""

while true; do
    read -e -p "Video file path (e.g. data/capture/IMG_9625.MOV): " VIDEO
    [ -f "$VIDEO" ] && break
    [ -f "$SCRIPT_DIR/$VIDEO" ] && VIDEO="$SCRIPT_DIR/$VIDEO" && break
    echo "  ERROR: File not found: $VIDEO — try again."
done

read -e -i "take1" -p "Dataset name: " NAME
read -e -i "512"   -p "Crop size in pixels: " SIZE
read -e -i "30"    -p "FPS: " FPS

OUT="$SCRIPT_DIR/data/processed"

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
echo "Next: check a few frames in $OUT/$NAME/frames/ — full head should be"
echo "visible (hair, chin, ears). Then run: bash start.sh  (choose stage 2)"
