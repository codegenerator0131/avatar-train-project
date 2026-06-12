#!/usr/bin/env bash
# Avatar pipeline: native Ubuntu setup (tested target: Ubuntu 22.04/24.04, RTX 3070/4080)
# Run as a normal user with sudo rights. NVIDIA driver must already be installed
# (check with: nvidia-smi). Install driver via "Additional Drivers" or:
#   sudo ubuntu-drivers autoinstall && reboot
set -euo pipefail

# Helper: check if a dpkg package is installed
pkg_installed() { dpkg -s "$1" &>/dev/null; }

# Helper: check if a Python package is installed in the active venv
py_installed() { python -c "import $1" &>/dev/null 2>&1; }

echo "== [1/5] System packages =="
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

echo "== [2/5] CUDA nvcc compiler =="
if command -v nvcc >/dev/null 2>&1; then
    echo "  [skip] nvcc already present: $(nvcc --version | grep release)"
else
    echo "  nvcc not found — adding CUDA repo and installing..."
    sudo rm -f /usr/share/keyrings/cuda-archive-keyring.gpg
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    # Manually add source list if dpkg didn't create it
    if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
            | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    fi

    sudo apt update || true

    # Install whichever nvcc version is available (prefer 12-4, fall back to latest)
    if sudo apt install -y cuda-nvcc-12-4 2>/dev/null; then
        echo "  Installed cuda-nvcc-12-4"
    else
        echo "  cuda-nvcc-12-4 not found, installing latest available..."
        sudo apt install -y cuda-nvcc-13-2 || sudo apt install -y cuda-nvcc-13-1 || sudo apt install -y cuda-nvcc-13-0
    fi
fi

# Ensure CUDA is on PATH
if ! grep -q 'cuda/bin' ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "== [3/5] Project venv =="
VENV_DIR="$(pwd)/.venv"
if [ -d "$VENV_DIR" ]; then
    echo "  [skip] venv already exists at $VENV_DIR"
else
    python3.11 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

echo "== [4/5] PyTorch =="
if py_installed torch; then
    echo "  [skip] torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch (cu126 wheels, compatible with CUDA 12.x and 13.x)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

echo "== [5/5] Pipeline dependencies =="
PY_PKGS=(numpy cv2 PIL imageio scipy matplotlib tqdm tensorboard einops trimesh ninja mediapipe)
PKG_NAMES=(numpy opencv-python-headless pillow imageio imageio-ffmpeg scipy matplotlib tqdm tensorboard einops trimesh ninja mediapipe)

TO_PIP=()
for i in "${!PY_PKGS[@]}"; do
    if py_installed "${PY_PKGS[$i]}"; then
        echo "  [skip] ${PY_PKGS[$i]} already installed"
    else
        TO_PIP+=("${PKG_NAMES[$i]}")
    fi
done
if [ ${#TO_PIP[@]} -gt 0 ]; then
    pip install "${TO_PIP[@]}"
else
    echo "  All pipeline dependencies already installed."
fi

mkdir -p data/capture data/flame output

echo ""
echo "Setup complete. Now run:"
echo "  source .venv/bin/activate && python verify_env.py"
