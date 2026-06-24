#!/usr/bin/env bash
# Avatar pipeline: Vast.ai setup (PyTorch Vast template, root user, miniforge pre-installed)
# Run inside the Vast.ai instance after uploading project files.
# Assumes: miniforge3 at /opt/miniforge3, CUDA 12.x driver, root user
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "  Avatar Pipeline — Vast.ai Setup"
echo "================================================"
echo ""

# ── FLAME file check ──────────────────────────────────────────────────────────
MISSING_FILES=0
check_file() {
    local f="$1" label="$2"
    if [ ! -f "$f" ]; then
        echo "  MISSING: $label → $f"
        MISSING_FILES=1
    fi
}
check_file "$SCRIPT_DIR/data/flame/flame2023.pkl"  "FLAME 2023 model"
check_file "$SCRIPT_DIR/data/flame/FLAME_masks.pkl" "FLAME Vertex Masks"
if [ "$MISSING_FILES" = "1" ]; then
    echo ""
    echo "  Upload FLAME files first (see DOWNLOADS.md), then re-run."
    exit 1
fi
echo "  [ok] FLAME files present"

# ── Use existing miniforge conda ──────────────────────────────────────────────
CONDA_DIR="/opt/miniforge3"
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "ERROR: conda not found at $CONDA_DIR"
    echo "This script is for Vast.ai PyTorch template only."
    exit 1
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
echo "  [ok] conda found at $CONDA_DIR"

# ── System packages (minimal — most already in Vast.ai image) ─────────────────
echo ""
echo "== [1/4] System packages =="
apt-get update -qq || true
PKGS=(build-essential git git-lfs ffmpeg ninja-build cmake pkg-config wget unzip gcc-11 g++-11)
TO_INSTALL=()
for p in "${PKGS[@]}"; do
    if dpkg -s "$p" &>/dev/null; then
        echo "  [skip] $p"
    else
        TO_INSTALL+=("$p")
    fi
done
if [ ${#TO_INSTALL[@]} -gt 0 ]; then
    apt-get install -y "${TO_INSTALL[@]}"
fi

# ── CUDA PATH ─────────────────────────────────────────────────────────────────
echo ""
echo "== [2/4] CUDA PATH =="
CUDA_PATH=""
for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
    if [ -d "$candidate/bin" ]; then
        CUDA_PATH="$candidate"
        break
    fi
done
if [ -n "$CUDA_PATH" ]; then
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    echo "  [ok] CUDA at $CUDA_PATH"
else
    echo "  WARNING: CUDA dir not found, nvcc may not work"
fi

# ── ELITE conda env ───────────────────────────────────────────────────────────
echo ""
echo "== [3/4] ELITE conda env =="
ELITE_DIR="$SCRIPT_DIR/elite"
ELITE_CONDA="$CONDA_DIR/bin/conda"
ELITE_PIP="$CONDA_DIR/envs/ELITE/bin/pip"
ELITE_PYTHON="$CONDA_DIR/envs/ELITE/bin/python"

# Ensure pip exists in ELITE env before using it
if [ ! -f "$ELITE_PIP" ] && conda env list | grep -q "^ELITE "; then
    "$ELITE_CONDA" install -y -n ELITE pip
fi

if [ -f "$ELITE_PYTHON" ] && "$ELITE_PYTHON" -c "import torch; import diff_surfel_rasterization; import vhap" &>/dev/null 2>&1; then
    echo "  [skip] ELITE env already fully installed"
else
    if ! conda env list | grep -q "^ELITE "; then
        echo "  Creating ELITE conda env (Python 3.10)..."
        "$ELITE_CONDA" create --name ELITE -y python=3.10
    fi

    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

    "$ELITE_CONDA" install -y -n ELITE pip
    "$ELITE_PIP" install --upgrade pip "setuptools==69.5.1" wheel hatchling editables

    # ELITE requirements (skip chumpy + torch — handled separately)
    grep -v "^chumpy\|^torch=\|^torchvision=\|^torchaudio=\|^triton=" \
        "$ELITE_DIR/requirements.txt" > /tmp/elite_req.txt || true
    "$ELITE_PIP" install -r /tmp/elite_req.txt

    "$ELITE_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git

    # PyTorch cu121
    "$ELITE_PIP" install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121

    # pytorch3d pre-built wheel
    "$ELITE_PIP" install \
        https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

    # diff-surfel-rasterization
    "$ELITE_CONDA" install -y -n ELITE -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake
    CUDA_HOME="$CONDA_DIR/envs/ELITE" CC=gcc-11 CXX=g++-11 \
        "$ELITE_PIP" install --no-build-isolation \
        git+https://github.com/hbb1/diff-surfel-rasterization.git

    # VHAP (needed by ELITE for FLAME)
    "$ELITE_PIP" install --no-build-isolation "$ELITE_DIR/vhap/" --no-deps

    # Fix conda path in configs
    sed -i "s|anaconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true
    sed -i "s|miniconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true

    echo "  ELITE env ready."
fi

# ── TTS conda env ─────────────────────────────────────────────────────────────
echo ""
echo "== [4/4] TTS conda env (XTTS v2) =="
TTS_PYTHON="$CONDA_DIR/envs/tts/bin/python"
TTS_PIP="$CONDA_DIR/envs/tts/bin/pip"

if [ -f "$TTS_PYTHON" ] && "$TTS_PYTHON" -c "import TTS" &>/dev/null 2>&1; then
    echo "  [skip] tts env already installed"
else
    if ! conda env list | grep -q "^tts "; then
        "$ELITE_CONDA" create --name tts -y python=3.10
    fi
    "$TTS_PIP" install --upgrade pip
    "$TTS_PIP" install transformers==4.39.3
    "$TTS_PIP" install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    "$TTS_PIP" install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    "$TTS_PIP" install TTS gdown
    echo "  tts env ready."
fi

# ── FLAME asset symlinks ──────────────────────────────────────────────────────
mkdir -p "$ELITE_DIR/asset/flame"
ln -sf "$SCRIPT_DIR/data/flame/flame2023.pkl"  "$ELITE_DIR/asset/flame/flame2023.pkl"
ln -sf "$SCRIPT_DIR/data/flame/FLAME_masks.pkl" "$ELITE_DIR/asset/flame/FLAME_masks.pkl"
echo "  FLAME assets symlinked."

# ── Processed data symlinks ───────────────────────────────────────────────────
echo ""
echo "== Fixing data symlinks =="
PROCESSED_SRC="$SCRIPT_DIR/data/processed/IMG_9625_nerf"
mkdir -p "$ELITE_DIR/data/source/processed" "$ELITE_DIR/data/source/tracked"

if [ -d "$PROCESSED_SRC" ]; then
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/processed/IMG_9625_whiteBg_staticOffset_maskBelowLine"
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/tracked/IMG_9625_whiteBg_staticOffset"
    echo "  [ok] Data symlinks created"
else
    echo "  WARNING: $PROCESSED_SRC not found — rsync may not be complete"
fi

# ── ELITE checkpoints ─────────────────────────────────────────────────────────
mkdir -p "$ELITE_DIR/checkpoints"
if [ -f "$ELITE_DIR/checkpoints/3d_prior.pth" ] && [ -f "$ELITE_DIR/checkpoints/2d_prior.pth" ]; then
    echo "  [ok] ELITE checkpoints present"
else
    echo ""
    echo "  ACTION REQUIRED: ELITE checkpoints missing."
    echo "  Upload from your server:"
    echo "    rsync -avz --progress -e 'ssh -p PORT' \\"
    echo "      ~/Documents/avatar-train-project/elite/checkpoints/ \\"
    echo "      root@HOST:/workspace/avatar-train-project/elite/checkpoints/"
    echo ""
fi

echo ""
echo "================================================"
echo "  Setup complete!"
echo "================================================"
echo ""
echo "  Run Stage 5:"
echo "    bash start_stage5.sh"
echo ""
