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

# Vast.ai stores envs in /venv/<name> not $CONDA_DIR/envs/<name>
# Detect the correct path
if [ -d "/venv/ELITE" ]; then
    ELITE_ENV_DIR="/venv/ELITE"
elif [ -d "$CONDA_DIR/envs/ELITE" ]; then
    ELITE_ENV_DIR="$CONDA_DIR/envs/ELITE"
else
    ELITE_ENV_DIR="$CONDA_DIR/envs/ELITE"
fi
ELITE_PIP="$ELITE_ENV_DIR/bin/pip"
ELITE_PYTHON="$ELITE_ENV_DIR/bin/python"

# Find system nvcc FIRST — needed for both torch selection and build
SYS_CUDA=""
for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8 /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
    if [ -f "$candidate/bin/nvcc" ]; then
        SYS_CUDA="$candidate"
        break
    fi
done
if [ -z "$SYS_CUDA" ]; then
    echo "ERROR: nvcc not found"
    exit 1
fi
echo "  nvcc found at: $SYS_CUDA"

# Detect CUDA version
NVCC_VER=$("$SYS_CUDA/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "12.1")
CUDA_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$NVCC_VER" | cut -d. -f2)
if [ "$CUDA_MAJOR" -ge 13 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; }; then
    TORCH_CUDA="cu126"
    TORCH_VER="2.6.0"
    TORCHVISION_VER="0.21.0"
    PYTORCH3D_WHEEL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu126_pyt260/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl"
else
    TORCH_CUDA="cu121"
    TORCH_VER="2.1.0"
    TORCHVISION_VER="0.16.0"
    PYTORCH3D_WHEEL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl"
fi
echo "  CUDA $NVCC_VER → will use torch==$TORCH_VER+$TORCH_CUDA"

if [ -f "$ELITE_PYTHON" ] && "$ELITE_PYTHON" -c "import torch; import diff_surfel_rasterization; import vhap" &>/dev/null 2>&1; then
    echo "  [skip] ELITE env already fully installed"
else
    if [ ! -f "$ELITE_PYTHON" ]; then
        echo "  Creating ELITE conda env (Python 3.10)..."
        "$ELITE_CONDA" create --prefix "$ELITE_ENV_DIR" -y python=3.10
    fi

    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    export CUDA_HOME="$SYS_CUDA"
    export PATH="$SYS_CUDA/bin:$PATH"

    "$ELITE_CONDA" install -y --prefix "$ELITE_ENV_DIR" pip
    "$ELITE_PIP" install --upgrade pip "setuptools==69.5.1" wheel hatchling editables

    # ELITE requirements (skip chumpy + torch — handled separately)
    grep -v "^chumpy\|^torch=\|^torchvision=\|^torchaudio=\|^triton=" \
        "$ELITE_DIR/requirements.txt" > /tmp/elite_req.txt || true
    "$ELITE_PIP" install -r /tmp/elite_req.txt

    "$ELITE_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git

    echo "  Installing torch==$TORCH_VER+$TORCH_CUDA..."
    "$ELITE_PIP" install torch==$TORCH_VER torchvision==$TORCHVISION_VER \
        --index-url https://download.pytorch.org/whl/$TORCH_CUDA

    # pytorch3d pre-built wheel
    "$ELITE_PIP" install "$PYTORCH3D_WHEEL" || \
        echo "  WARNING: pytorch3d wheel not found, skipping"

    # Symlink nvcc into ELITE env bin so build tools find it
    mkdir -p "$ELITE_ENV_DIR/bin"
    ln -sf "$SYS_CUDA/bin/nvcc" "$ELITE_ENV_DIR/bin/nvcc"

    # diff-surfel-rasterization
    echo "  Building diff_surfel_rasterization with CUDA_HOME=$SYS_CUDA..."
    CUDA_HOME="$SYS_CUDA" CC=gcc-11 CXX=g++-11 \
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
# Detect TTS env path
if [ -d "/venv/tts" ]; then
    TTS_ENV_DIR="/venv/tts"
else
    TTS_ENV_DIR="$CONDA_DIR/envs/tts"
fi
TTS_PYTHON="$TTS_ENV_DIR/bin/python"
TTS_PIP="$TTS_ENV_DIR/bin/pip"

if [ -f "$TTS_PYTHON" ] && "$TTS_PYTHON" -c "import TTS" &>/dev/null 2>&1; then
    echo "  [skip] tts env already installed"
else
    if [ ! -f "$TTS_PYTHON" ]; then
        "$ELITE_CONDA" create --prefix "$TTS_ENV_DIR" -y python=3.10
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
