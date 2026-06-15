#!/usr/bin/env bash
# Avatar pipeline: native Ubuntu setup (tested target: Ubuntu 22.04/24.04, RTX 3070/4080)
# Run as a normal user with sudo rights. NVIDIA driver must already be installed
# (check with: nvidia-smi). Install driver via "Additional Drivers" or:
#   sudo ubuntu-drivers autoinstall && reboot
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Helper: check if a dpkg package is installed
pkg_installed() { dpkg -s "$1" &>/dev/null; }

echo "== [1/4] System packages =="
sudo apt update || true

PKGS=(build-essential git git-lfs ffmpeg ninja-build cmake pkg-config wget unzip python3.11 python3.11-venv python3.11-dev)
TO_INSTALL=()
for p in "${PKGS[@]}"; do
    if pkg_installed "$p"; then
        echo "  [skip] $p already installed"
    else
        TO_INSTALL+=("$p")
    fi
done
if [ ${#TO_INSTALL[@]} -gt 0 ]; then
    sudo apt install -y "${TO_INSTALL[@]}"
else
    echo "  All system packages already installed."
fi

echo "== [2/4] CUDA nvcc compiler =="
if command -v nvcc >/dev/null 2>&1; then
    echo "  [skip] nvcc already present: $(nvcc --version | grep release)"
else
    echo "  nvcc not found — adding CUDA repo and installing..."
    sudo rm -f /usr/share/keyrings/cuda-archive-keyring.gpg
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
            | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    fi

    sudo apt update || true

    if sudo apt install -y cuda-nvcc-12-4 2>/dev/null; then
        echo "  Installed cuda-nvcc-12-4"
    else
        echo "  cuda-nvcc-12-4 not found, installing latest available..."
        sudo apt install -y cuda-nvcc-13-2 || sudo apt install -y cuda-nvcc-13-1 || sudo apt install -y cuda-nvcc-13-0
    fi
fi

echo "== [2b/4] CUDA PATH =="
# Find the actual CUDA installation directory
CUDA_PATH=""
for candidate in /usr/local/cuda /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0 /usr/local/cuda-12.4; do
    if [ -d "$candidate/bin" ]; then
        CUDA_PATH="$candidate"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    # Try finding nvcc directly
    NVCC_PATH="$(command -v nvcc 2>/dev/null || true)"
    if [ -n "$NVCC_PATH" ]; then
        CUDA_PATH="$(dirname "$(dirname "$NVCC_PATH")")"
    fi
fi

if [ -n "$CUDA_PATH" ]; then
    echo "  Found CUDA at: $CUDA_PATH"
    # Add to current session
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

    # Add to .bashrc permanently if not already there
    if ! grep -q "$CUDA_PATH/bin" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA" >> ~/.bashrc
        echo "export PATH=$CUDA_PATH/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\${LD_LIBRARY_PATH:-}" >> ~/.bashrc
        echo "  Added CUDA to ~/.bashrc"
    else
        echo "  [skip] CUDA already in ~/.bashrc"
    fi
else
    echo "  WARNING: CUDA directory not found. nvcc may not work."
fi

echo "== [3/4] Project venv =="
if [ -d "$VENV_DIR" ]; then
    echo "  [skip] venv already exists at $VENV_DIR"
else
    python3.11 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

echo "== [4/4] Python packages =="
# PyTorch: install from custom index URL (not in requirements.txt)
if python -c "import torch" &>/dev/null 2>&1; then
    echo "  [skip] torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch (cu126 wheels)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu126
fi

# All other dependencies from requirements.txt
echo "  Installing packages from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet

# chumpy (FLAME pkl dependency) — pip's build is broken on Python 3.11,
# install directly from source instead
if python -c "import chumpy" &>/dev/null 2>&1; then
    echo "  [skip] chumpy already installed"
else
    echo "  Installing chumpy from source..."
    pip install --no-build-isolation git+https://github.com/mattloper/chumpy.git --quiet
fi

mkdir -p "$SCRIPT_DIR/data/capture" "$SCRIPT_DIR/data/flame" "$SCRIPT_DIR/data/source/input_videos" "$SCRIPT_DIR/output"

echo "== [5/6] Miniconda =="
MINICONDA_DIR="$HOME/miniconda3"
if [ ! -f "$MINICONDA_DIR/bin/conda" ]; then
    echo "  Installing Miniconda..."
    MINICONDA_SH="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_SH"
    bash "$MINICONDA_SH" -b -p "$MINICONDA_DIR"
    rm -f "$MINICONDA_SH"
    echo "  Miniconda installed at $MINICONDA_DIR"
else
    echo "  [skip] Miniconda already installed at $MINICONDA_DIR"
fi

# Always source conda into this shell session
# shellcheck disable=SC1090
source "$MINICONDA_DIR/etc/profile.d/conda.sh"
conda init bash >/dev/null 2>&1 || true

# Always accept TOS (safe to run multiple times, no-op if already accepted)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "== [6/6] VHAP submodule (Stage 2 tracker) =="
VHAP_DIR="$SCRIPT_DIR/vhap"
if [ -f "$VHAP_DIR/vhap/preprocess_video.py" ]; then
    echo "  [skip] VHAP submodule already present"
else
    echo "  Adding VHAP submodule..."
    cd "$SCRIPT_DIR"
    git submodule add https://github.com/ShenhanQian/VHAP.git vhap || true
    git submodule update --init --recursive
fi

if conda env list | grep -q "^vhap "; then
    echo "  [skip] conda env 'vhap' already exists"
else
    echo "  Creating conda env 'vhap'..."
    conda create --name vhap -y python=3.10
    # Use full path to vhap env pip/conda — conda activate doesn't persist in subshells
    VHAP_PIP="$HOME/miniconda3/envs/vhap/bin/pip"
    VHAP_CONDA="$HOME/miniconda3/bin/conda"
    VHAP_PYTHON="$HOME/miniconda3/envs/vhap/bin/python"
    VHAP_ENV_DIR="$HOME/miniconda3/envs/vhap"

    "$VHAP_PIP" install --upgrade pip setuptools wheel editables hatchling
    # Auto-detect CUDA version and pick matching torch + conda cuda-toolkit wheels
    CUDA_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')" || CUDA_VER=""
    CUDA_MAJOR="$(echo "$CUDA_VER" | cut -d. -f1)"
    CUDA_MINOR="$(echo "$CUDA_VER" | cut -d. -f2)"
    if [ -z "$CUDA_MAJOR" ]; then
        echo "  WARNING: nvcc not found, defaulting to cu126 wheels"
        TORCH_CUDA="cu126"
        CONDA_CUDA_CHANNEL="nvidia/label/cuda-12.6.0"
    elif [ "$CUDA_MAJOR" -ge 13 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; }; then
        TORCH_CUDA="cu126"
        CONDA_CUDA_CHANNEL="nvidia/label/cuda-12.6.0"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
        TORCH_CUDA="cu121"
        CONDA_CUDA_CHANNEL="nvidia/label/cuda-12.1.1"
    else
        TORCH_CUDA="cu118"
        CONDA_CUDA_CHANNEL="nvidia/label/cuda-11.8.0"
    fi
    echo "  Detected CUDA $CUDA_VER — using torch $TORCH_CUDA wheels"
    "$VHAP_CONDA" install -y -n vhap -c "$CONDA_CUDA_CHANNEL" cuda-toolkit ninja cmake
    ln -sf "$VHAP_ENV_DIR/lib" "$VHAP_ENV_DIR/lib64" 2>/dev/null || true
    "$VHAP_PIP" install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    "$VHAP_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git
    "$VHAP_PIP" install "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling" --force-reinstall
    rm -rf ~/.cache/torch_extensions/*/nvdiffrast* 2>/dev/null || true
    # Install VHAP deps manually (skip pytorch3d — not needed, uses nvdiffrast instead)
    "$VHAP_PIP" install tyro pyyaml "numpy==1.22.3" "matplotlib==3.8.0" scipy pillow \
        opencv-python ffmpeg-python colour-science tensorboard trimesh \
        dlib pandas gdown face-alignment joblib dearpygui
    # BackgroundMattingV2 — clone ShenhanQian's fork and pip install it
    if [ ! -d "$VHAP_DIR/BackgroundMattingV2" ]; then
        git clone https://github.com/ShenhanQian/BackgroundMattingV2.git "$VHAP_DIR/BackgroundMattingV2"
    fi
    "$VHAP_PIP" install "$VHAP_DIR/BackgroundMattingV2/"
    "$VHAP_PIP" install --no-build-isolation -e "$VHAP_DIR/" --no-deps
    conda deactivate
    echo "  conda env 'vhap' ready."
fi

echo ""
echo "Setup complete. Now run:"
echo "  source venv/bin/activate && python verify_env.py"
echo ""
echo "To run Stage 2 (VHAP tracking):"
echo "  bash start_stage2.sh"
