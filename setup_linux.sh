#!/usr/bin/env bash
# Avatar pipeline: native Ubuntu setup (tested target: Ubuntu 22.04/24.04, RTX 4080 Laptop)
# Run as a normal user with sudo rights. NVIDIA driver must already be installed
# (check with: nvidia-smi). Install driver via "Additional Drivers" or:
#   sudo ubuntu-drivers autoinstall && reboot
set -euo pipefail

echo "== [1/5] System packages =="
sudo apt update
sudo apt install -y build-essential git git-lfs ffmpeg ninja-build cmake pkg-config \
    wget unzip python3.11 python3.11-venv python3.11-dev

echo "== [2/5] CUDA Toolkit 12.4 (compiler needed for custom splat kernels) =="
if ! command -v nvcc >/dev/null 2>&1; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-4
    rm -f cuda-keyring_1.1-1_all.deb
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
else
    echo "nvcc already present: $(nvcc --version | tail -1)"
fi

echo "== [3/5] Project venv =="
mkdir -p ~/avatar && cd ~/avatar
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
echo "Setup complete. Now run:  source ~/avatar/.venv/bin/activate && python verify_env.py"
