#!/usr/bin/env bash
# Avatar pipeline: Vast.ai setup
# Works on any CUDA driver version — uses conda cuda-toolkit 12.1 for compilation
# Same approach as the working Linux server setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"

echo "================================================"
echo "  Avatar Pipeline — Vast.ai Setup"
echo "================================================"
echo ""

# ── FLAME file check ──────────────────────────────────────────────────────────
for f in "$SCRIPT_DIR/data/flame/flame2023.pkl" "$SCRIPT_DIR/data/flame/FLAME_masks.pkl"; do
    [ -f "$f" ] || { echo "MISSING: $f — upload FLAME files first"; exit 1; }
done
echo "  [ok] FLAME files present"

# ── conda ─────────────────────────────────────────────────────────────────────
CONDA_DIR="/opt/miniforge3"
[ -f "$CONDA_DIR/bin/conda" ] || { echo "ERROR: conda not found at $CONDA_DIR"; exit 1; }
source "$CONDA_DIR/etc/profile.d/conda.sh"
CONDA="$CONDA_DIR/bin/conda"
echo "  [ok] conda at $CONDA_DIR"

# ── system packages ───────────────────────────────────────────────────────────
echo ""
echo "== [1/3] System packages =="
apt-get update -qq || true
for p in build-essential git ffmpeg ninja-build cmake gcc-11 g++-11; do
    dpkg -s "$p" &>/dev/null && echo "  [skip] $p" || apt-get install -y "$p"
done

# ── ELITE env ─────────────────────────────────────────────────────────────────
echo ""
echo "== [2/3] ELITE conda env =="

# Detect env path — Vast.ai uses /venv/<name>
if [ -d "/venv/ELITE" ]; then
    ELITE_ENV="/venv/ELITE"
else
    ELITE_ENV="$CONDA_DIR/envs/ELITE"
fi
ELITE_PIP="$ELITE_ENV/bin/pip"
ELITE_PYTHON="$ELITE_ENV/bin/python"

if "$ELITE_PYTHON" -c "import torch; import diff_surfel_rasterization; import vhap" &>/dev/null 2>&1; then
    echo "  [skip] ELITE env already complete"
else
    # Create env if missing
    [ -f "$ELITE_PYTHON" ] || "$CONDA" create --prefix "$ELITE_ENV" -y python=3.10

    # Install cuda-toolkit 12.1 via conda INTO the env — same as working server
    echo "  Installing cuda-toolkit 12.1 into ELITE env..."
    "$CONDA" install -y --prefix "$ELITE_ENV" -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake

    # Set CUDA_HOME to the env (has nvcc 12.1 now)
    export CUDA_HOME="$ELITE_ENV"
    export PATH="$ELITE_ENV/bin:$PATH"
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

    # Verify nvcc
    "$ELITE_ENV/bin/nvcc" --version

    # pip + base tools
    "$CONDA" install -y --prefix "$ELITE_ENV" pip
    "$ELITE_PIP" install --upgrade pip "setuptools==69.5.1" wheel hatchling editables

    # ELITE requirements (skip torch/chumpy/triton)
    grep -v "^chumpy\|^torch=\|^torchvision=\|^torchaudio=\|^triton=" \
        "$ELITE_DIR/requirements.txt" > /tmp/elite_req.txt
    "$ELITE_PIP" install -r /tmp/elite_req.txt

    "$ELITE_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git

    # PyTorch cu121 — matches cuda-toolkit 12.1 in the env
    echo "  Installing torch 2.1.0+cu121..."
    "$ELITE_PIP" install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121

    # pytorch3d pre-built wheel (py310 + cu121 + pyt210)
    "$ELITE_PIP" install \
        https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

    # diff-surfel-rasterization — compiled against cu121 in env
    echo "  Building diff_surfel_rasterization..."
    CUDA_HOME="$ELITE_ENV" CC=gcc-11 CXX=g++-11 \
        "$ELITE_PIP" install --no-build-isolation \
        git+https://github.com/hbb1/diff-surfel-rasterization.git

    # VHAP
    "$ELITE_PIP" install --no-build-isolation "$ELITE_DIR/vhap/" --no-deps

    # Fix paths.sh
    sed -i "s|anaconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true
    sed -i "s|miniconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true

    echo "  ELITE env ready."
fi

# ── TTS env ───────────────────────────────────────────────────────────────────
echo ""
echo "== [3/3] TTS conda env =="
if [ -d "/venv/tts" ]; then TTS_ENV="/venv/tts"; else TTS_ENV="$CONDA_DIR/envs/tts"; fi
TTS_PIP="$TTS_ENV/bin/pip"
TTS_PYTHON="$TTS_ENV/bin/python"

if "$TTS_PYTHON" -c "import TTS" &>/dev/null 2>&1; then
    echo "  [skip] tts env already complete"
else
    [ -f "$TTS_PYTHON" ] || "$CONDA" create --prefix "$TTS_ENV" -y python=3.10
    "$TTS_PIP" install --upgrade pip
    "$TTS_PIP" install transformers==4.39.3
    "$TTS_PIP" install torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121
    "$TTS_PIP" install TTS gdown
    echo "  tts env ready."
fi

# ── symlinks ──────────────────────────────────────────────────────────────────
mkdir -p "$ELITE_DIR/asset/flame"
ln -sf "$SCRIPT_DIR/data/flame/flame2023.pkl"  "$ELITE_DIR/asset/flame/flame2023.pkl"
ln -sf "$SCRIPT_DIR/data/flame/FLAME_masks.pkl" "$ELITE_DIR/asset/flame/FLAME_masks.pkl"

PROCESSED_SRC="$SCRIPT_DIR/data/processed/IMG_9625_nerf"
mkdir -p "$ELITE_DIR/data/source/processed" "$ELITE_DIR/data/source/tracked"
if [ -d "$PROCESSED_SRC" ]; then
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/processed/IMG_9625_whiteBg_staticOffset_maskBelowLine"
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/tracked/IMG_9625_whiteBg_staticOffset"
    echo "  [ok] data symlinks created"
else
    echo "  WARNING: $PROCESSED_SRC not found"
fi

echo ""
echo "================================================"
echo "  Setup complete! Run: bash start_stage5.sh"
echo "================================================"
