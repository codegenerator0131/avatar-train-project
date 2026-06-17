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

    "$VHAP_PIP" install pip "setuptools==69.5.1" wheel editables hatchling
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
    # Pin numpy first to prevent later packages from upgrading it to 2.x
    "$VHAP_PIP" install "numpy==1.22.3"
    # Pin to 2.1.0 to match the pytorch3d pre-built wheel (py310_cu121_pyt210)
    "$VHAP_PIP" install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    "$VHAP_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git
    # pytorch3d pre-built wheel (py310 + cu121 + pyt210) — avoids compiling from source
    "$VHAP_PIP" install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
    "$VHAP_PIP" install "git+https://github.com/ShenhanQian/STAR/" --no-build-isolation
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
    # VHAP looks for asset/ relative to working directory (project root)
    # Symlink vhap/asset → asset so all VHAP assets are found automatically
    ln -sfn "$VHAP_DIR/asset" "$SCRIPT_DIR/asset"
    # Override the FLAME files with our own copies (flame2023.pkl, FLAME_masks.pkl)
    ln -sf "$SCRIPT_DIR/data/flame/flame2023.pkl"  "$SCRIPT_DIR/asset/flame/flame2023.pkl"
    ln -sf "$SCRIPT_DIR/data/flame/FLAME_masks.pkl" "$SCRIPT_DIR/asset/flame/FLAME_masks.pkl"
    conda deactivate
    echo "  conda env 'vhap' ready."
fi

echo "== [7/7] ELITE submodule (Stage 3 Gaussian avatar) =="
ELITE_DIR="$SCRIPT_DIR/elite"
if [ -f "$ELITE_DIR/src/render.py" ]; then
    echo "  [skip] ELITE submodule already present"
else
    echo "  Cloning ELITE..."
    cd "$SCRIPT_DIR"
    rm -rf "$ELITE_DIR"
    git clone https://github.com/kaist-ami/ELITE.git "$ELITE_DIR"
    cd "$ELITE_DIR"
    git submodule update --init --recursive
    cd "$SCRIPT_DIR"
fi

ELITE_CONDA="$HOME/miniconda3/bin/conda"
ELITE_PIP="$HOME/miniconda3/envs/ELITE/bin/pip"
ELITE_PYTHON="$HOME/miniconda3/envs/ELITE/bin/python"

if "$HOME/miniconda3/envs/ELITE/bin/python" -c "import torch; import diff_surfel_rasterization; import vhap" &>/dev/null 2>&1; then
    echo "  [skip] conda env 'ELITE' already installed"
else
    echo "  Installing ELITE conda env packages..."
    # Create env if it doesn't exist yet
    if ! conda env list | grep -q "^ELITE "; then
        "$ELITE_CONDA" create --name ELITE -y python=3.10
    fi

    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    # Install gcc-11 if missing
    if ! command -v gcc-11 >/dev/null 2>&1; then
        sudo apt install -y gcc-11 g++-11 || true
    fi

    # Core deps — install pip into env first, then upgrade
    "$ELITE_CONDA" install -y -n ELITE pip
    "$ELITE_PIP" install --upgrade pip setuptools==69.5.1 wheel hatchling editables
    # Install requirements excluding chumpy and torch (we pin specific versions below)
    grep -v "^chumpy\|^torch=\|^torchvision=\|^torchaudio=\|^triton=" "$ELITE_DIR/requirements.txt" > /tmp/elite_requirements.txt || true
    "$ELITE_PIP" install -r /tmp/elite_requirements.txt
    "$ELITE_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git

    # Use cu121 torch — matches pytorch3d pre-built wheel AND works with CUDA 13.2 driver
    "$ELITE_PIP" install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    # pytorch3d pre-built wheel (py310 + cu121 + pyt210)
    "$ELITE_PIP" install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

    # diff-surfel-rasterization — needs CUDA_HOME pointing to conda cuda-toolkit
    "$ELITE_CONDA" install -y -n ELITE -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake
    export CUDA_HOME="$HOME/miniconda3/envs/ELITE"
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    CUDA_HOME="$HOME/miniconda3/envs/ELITE" CC=gcc-11 CXX=g++-11 \
        "$ELITE_PIP" install --no-build-isolation \
        git+https://github.com/hbb1/diff-surfel-rasterization.git

    # Install VHAP from elite's submodule
    "$ELITE_PIP" install --no-build-isolation "$ELITE_DIR/vhap/" --no-deps

    # Patch configs/paths.sh with correct conda path
    sed -i "s|anaconda3/etc/profile.d/conda.sh|miniconda3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh"

    echo "  conda env 'ELITE' ready."
fi

# Always ensure FLAME assets are symlinked (runs every time, idempotent)
mkdir -p "$ELITE_DIR/asset/flame"
ln -sf "$SCRIPT_DIR/data/flame/flame2023.pkl"  "$ELITE_DIR/asset/flame/flame2023.pkl"
ln -sf "$SCRIPT_DIR/data/flame/FLAME_masks.pkl" "$ELITE_DIR/asset/flame/FLAME_masks.pkl"
echo "  FLAME assets symlinked into elite/asset/flame/"

# --- ELITE checkpoints (must be placed manually)
mkdir -p "$ELITE_DIR/checkpoints"
if [ -f "$ELITE_DIR/checkpoints/3d_prior.pth" ] && [ -f "$ELITE_DIR/checkpoints/2d_prior.pth" ]; then
    echo "  [skip] ELITE checkpoints already present"
else
    echo ""
    echo "  -------------------------------------------------------"
    echo "  ACTION REQUIRED: ELITE checkpoints not found."
    echo "  Download from:"
    echo "    https://drive.google.com/drive/folders/1GKVymlwRi9shK0G2Qi5JrOFfkIdyUaHM"
    echo "  Place at:"
    echo "    $ELITE_DIR/checkpoints/3d_prior.pth"
    echo "    $ELITE_DIR/checkpoints/2d_prior.pth"
    echo "  Then re-run: bash start_stage3.sh"
    echo "  -------------------------------------------------------"
    echo ""
fi

echo ""
echo "Setup complete. Now run:"
echo "  source venv/bin/activate && python verify_env.py"
echo ""
echo "To run Stage 2 (VHAP tracking):"
echo "  bash start_stage2.sh"
echo ""
echo "To run Stage 3 (Gaussian avatar):"
echo "  bash start_stage3.sh"
