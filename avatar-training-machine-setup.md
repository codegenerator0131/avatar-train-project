# Avatar Training Machine Setup Brief

**Target machine:** Windows laptop with NVIDIA RTX 4080 Laptop GPU (12GB VRAM)
**Goal:** Ubuntu environment via WSL2 with CUDA-enabled PyTorch for 3D Gaussian Splatting avatar training.
**Estimated time:** 45 to 90 minutes including downloads.

-----

## Part 1: Windows host

### 1.1 Update NVIDIA driver

- Install the latest **Game Ready** or **Studio** driver from nvidia.com (or via GeForce Experience).
- This single Windows driver also serves WSL2. **Do NOT install any NVIDIA driver inside Linux later.** Doing so breaks GPU passthrough.

### 1.2 Install WSL2 with Ubuntu 24.04

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot when prompted. On first launch of Ubuntu, create a username and password and give these credentials to the project owner.

If WSL was already installed previously, update it first:

```powershell
wsl --update
wsl --set-default-version 2
```

### 1.3 Recommended tools (Windows side)

- **Windows Terminal** (Microsoft Store)
- **VS Code** + the **“WSL”** extension (for remote editing inside the Linux environment)

### 1.4 Power settings

- Windows power mode: **Best performance**
- NVIDIA Control Panel > Manage 3D settings > Power management mode: **Prefer maximum performance**
- Training runs take hours at 100% GPU load. The laptop must stay **plugged in** with good airflow (hard surface, ideally a cooling pad).

-----

## Part 2: Inside Ubuntu (WSL2)

Open the Ubuntu terminal for all steps below.

### 2.1 Base packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git git-lfs ffmpeg ninja-build \
    python3.11 python3.11-venv python3.11-dev \
    cmake pkg-config wget unzip
```

### 2.2 Verify GPU is visible to WSL

```bash
nvidia-smi
```

Expected: a table showing “NVIDIA GeForce RTX 4080 Laptop GPU”. If this fails, the Windows driver from step 1.1 is missing or outdated. Do not proceed until this works. Reminder: never `apt install` any nvidia driver inside WSL.

### 2.3 CUDA Toolkit 12.x (WSL-Ubuntu variant, toolkit only)

Follow NVIDIA’s official “CUDA on WSL” instructions for the **WSL-Ubuntu** package, which installs the toolkit WITHOUT a driver:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Then `source ~/.bashrc` and verify:

```bash
nvcc --version
```

Expected: release 12.4 (or the 12.x version installed). The nvcc compiler is required because the project compiles custom CUDA extensions (gaussian splat rasterizer).

### 2.4 Project directory and Python environment

**Important:** the project must live in the Linux filesystem (`~/`), NOT under `/mnt/c/`. Cross-filesystem I/O in WSL is 10x+ slower and will bottleneck training.

```bash
mkdir -p ~/avatar && cd ~/avatar
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2.5 PyTorch with CUDA 12 support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2.6 Supporting Python packages

```bash
pip install numpy opencv-python-headless pillow imageio imageio-ffmpeg \
    scipy matplotlib tqdm tensorboard einops trimesh chumpy ninja
```

-----

## Part 3: Verification (send these outputs back)

Run each command inside Ubuntu with the venv activated and capture the output:

```bash
# 1. GPU visible
nvidia-smi

# 2. CUDA compiler present
nvcc --version

# 3. PyTorch sees the GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4. Can compile a CUDA extension (the real test)
python -c "from torch.utils.cpp_extension import CUDA_HOME; print('CUDA_HOME:', CUDA_HOME)"

# 5. ffmpeg works
ffmpeg -version | head -1
```

**Pass criteria:** test 3 must print `True` and `NVIDIA GeForce RTX 4080 Laptop GPU`; test 4 must print a valid path (not None).

-----

## Part 4: Notes for the project owner (not devops)

- FLAME model files: register at flame.is.tue.mpg.de and download FLAME 2023. Place the files in `~/avatar/data/flame/` (these are license-gated, the account must be the owner’s).
- Training video transfer: copy the video into WSL, e.g. `cp /mnt/c/Users/<name>/Videos/capture.mov ~/avatar/data/capture/`. Original resolution, no re-encoding or messaging-app compression (do not send the video through chat apps, transfer the original file).
- VRAM expectation: per-user avatar training fits within 12GB at our planned settings (up to ~300k gaussians at 512 to 800px crops). If out-of-memory errors appear, reduce splat cap or crop size, do not raise WSL memory limits expecting it to help (VRAM is separate from system RAM).
- Optional: create `%UserProfile%\.wslconfig` on Windows to give WSL more system RAM if the machine has 32GB+:

```
[wsl2]
memory=24GB
```

## Troubleshooting quick hits

- `nvidia-smi` fails in WSL: update the Windows NVIDIA driver, then `wsl --shutdown` in PowerShell and reopen Ubuntu.
- CUDA extension compile errors mentioning gcc version: install `gcc-12 g++-12` and set `export CC=gcc-12 CXX=g++-12`.
- Slow file access: confirm the project is in `~/avatar`, not `/mnt/c/...`.
- WSL clock drift after laptop sleep: `sudo hwclock -s` fixes certificate/pip errors.