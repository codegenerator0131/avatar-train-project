#!/usr/bin/env bash
# Avatar Pipeline — main launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "  Avatar Pipeline"
echo "================================================"
echo ""
echo "Which stage do you want to run?"
echo "  1) Stage 1 — Capture   (video → frames + audio)"
echo "  2) Stage 2 — Tracking  (frames → FLAME params per frame)"
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
