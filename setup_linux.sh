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

echo "== [5/5] VHAP submodule (Stage 2 tracker) =="
VHAP_DIR="$SCRIPT_DIR/vhap"
if [ -f "$VHAP_DIR/vhap/preprocess_video.py" ]; then
    echo "  [skip] VHAP submodule already present"
else
    echo "  Adding VHAP submodule..."
    cd "$SCRIPT_DIR"
    git submodule add https://github.com/ShenhanQian/VHAP.git vhap || true
    git submodule update --init --recursive
fi

if conda env list 2>/dev/null | grep -q "^vhap "; then
    echo "  [skip] conda env 'vhap' already exists"
else
    if command -v conda >/dev/null 2>&1; then
        echo "  Creating conda env 'vhap'..."
        # shellcheck disable=SC1090
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda create --name vhap -y python=3.10
        conda activate vhap
        conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake
        ln -sf "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64" 2>/dev/null || true
        conda env config vars set CUDA_HOME="$CONDA_PREFIX"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        pip install git+https://github.com/ShenhanQian/nvdiffrast.git
        pip install -e "$VHAP_DIR/"
        conda deactivate
        echo "  conda env 'vhap' ready."
    else
        echo "  WARNING: conda not found — skipping VHAP env setup."
        echo "  Install Miniconda then re-run setup, or follow manual steps in README."
    fi
fi

echo ""
echo "Setup complete. Now run:"
echo "  source venv/bin/activate && python verify_env.py"
echo ""
echo "To run Stage 2 (VHAP tracking):"
echo "  bash start_stage2.sh"
