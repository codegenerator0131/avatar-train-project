#!/usr/bin/env bash
# Avatar Pipeline — main launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Run setup if this is a new machine (venv missing)
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "First run detected — running setup..."
    echo ""
    bash "$SCRIPT_DIR/setup_linux.sh"
    echo ""
fi

echo "================================================"
echo "  Avatar Pipeline"
echo "================================================"
echo ""
echo "Which stage do you want to run?"
echo "  1) Stage 1 — Capture    (video → frames + audio)"
echo "  2) Stage 2 — Tracking   (video → FLAME params + camera poses)"
echo ""
read -rp "Enter stage [1]: " STAGE
STAGE="${STAGE:-1}"
echo ""

case "$STAGE" in
    1) bash "$SCRIPT_DIR/start_stage1.sh" ;;
    2) bash "$SCRIPT_DIR/start_stage2.sh" ;;
    *)
        echo "Unknown stage: $STAGE. Enter 1 or 2."
        exit 1
        ;;
esac
