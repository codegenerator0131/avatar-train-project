#!/usr/bin/env bash
# Avatar pipeline: native Ubuntu setup (tested target: Ubuntu 22.04/24.04, RTX 3070/4080)
# Run as a normal user with sudo rights. NVIDIA driver must already be installed
# (check with: nvidia-smi). Install driver via "Additional Drivers" or:
#   sudo ubuntu-drivers autoinstall && reboot
set -euo pipefail

echo "== [1/5] System packages =="
# Ignore errors from broken third-party repos (AnyDesk, PlasticSCM, etc.)
sudo apt update || true
sudo apt install -y build-essential git git-lfs ffmpeg ninja-build cmake pkg-config \
    wget unzip python3.11 python3.11-venv python3.11-dev

echo "== [2/5] CUDA Toolkit 12.4 (compiler needed for custom splat kernels) =="
if ! command -v nvcc >/dev/null 2>&1; then
    # Remove stale keyring if present, then re-download
    sudo rm -f /usr/share/keyrings/cuda-archive-keyring.gpg
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    # Update only the CUDA repo to avoid failures from broken third-party repos
    sudo apt update -o Dir::Etc::sourcelist="sources.list.d/cuda-ubuntu2404-x86_64.list" \
                    -o Dir::Etc::sourceparts="-" \
                    -o APT::Get::List-Cleanup="0" || sudo apt update || true

    sudo apt install -y cuda-toolkit-12-4

    # Add CUDA to PATH (skip if already present)
    if ! grep -q 'cuda/bin' ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
else
    echo "nvcc already present: $(nvcc --version | tail -1)"
    # Still ensure PATH is set in .bashrc
    if ! grep -q 'cuda/bin' ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
fi

echo "== [3/5] Project venv =="
mkdir -p ~/avatar
cd ~/avatar
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "== [4/5] PyTorch (CUDA 12.4 wheels) =="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "== [5/5] Pipeline dependencies =="
pip install numpy opencv-python-headless pillow imageio imageio-ffmpeg \
    scipy matplotlib tqdm tensorboard einops trimesh ninja mediapipe

mkdir -p ~/avatar/data/capture ~/avatar/data/flame ~/avatar/output

echo ""
echo "Setup complete. Now run:"
echo "  source ~/avatar/.venv/bin/activate && python verify_env.py"
