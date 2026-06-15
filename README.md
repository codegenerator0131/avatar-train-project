# Avatar Train

An in-house pipeline for building fully animatable 3D talking-head avatars from a single short video. Records a person speaking, fits a 3D head model to every frame, trains a 3D Gaussian avatar, then drives it with audio or text to produce talking-head video.

Inspired by the [Tavus Phoenix](https://www.tavus.io) model. Uses [FLAME 2023](https://flame.is.tue.mpg.de) for face parametrization and [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) for rendering.

---

## Pipeline overview

```
Video
  │
  ▼
Stage 1 ── Capture ──────────── frames + audio + meta.json
  │
  ▼
Stage 2 ── FLAME Tracking ───── per-frame FLAME params + camera poses (transforms.json)
  │
  ▼
Stage 3 ── Splat Training ────── trained 3D Gaussian avatar
  │
  ▼
Stage 4 ── Audio-to-Motion ───── blendshape curves from speech audio
  │
  ▼
Stage 5 ── Inference ──────────── talking-head video from text or audio
```

| Stage | Status |
|-------|--------|
| 1 — Capture | Done |
| 2 — FLAME Tracking (VHAP) | Ready to run |
| 3 — Splat Training | Not built |
| 4 — Audio-to-Motion | Not built |
| 5 — Inference | Not built |

---

## Hardware requirements

| Component | Minimum | Tested on |
|-----------|---------|-----------|
| GPU | 8 GB VRAM | RTX 3070 Laptop 8 GB |
| RAM | 16 GB | 15 GB |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |
| CUDA | 11.8+ | 13.2 (nvcc 13.2.78) |
| Python | 3.10 | 3.10 |

> The pipeline was designed for 12 GB VRAM (RTX 4080 Laptop). On 8 GB, use `--size 512` for capture and limit Gaussians to ~150k in Stage 3.

---

## Setup

```bash
git clone <this-repo>
cd avatar-train-project
bash setup_linux.sh
bash start.sh
```

That's it. On a new machine, `start.sh` detects missing setup and runs `setup_linux.sh` automatically if you forget.

`setup_linux.sh` does everything in one shot:
1. System packages (`build-essential`, `ffmpeg`, `cmake`, etc.)
2. CUDA nvcc compiler
3. Main Python venv + all dependencies (PyTorch cu126, chumpy, mediapipe, etc.)
4. VHAP submodule (`git submodule add`) + dedicated `vhap` conda env with nvdiffrast

**Prerequisite:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) must be installed before running setup (needed for the VHAP conda env). NVIDIA driver must also be installed (`nvidia-smi` should work).

### Required FLAME assets

Register for free at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) and download:

| File | Place at |
|------|----------|
| `flame2023.pkl` | `data/flame/flame2023.pkl` |
| `FLAME_masks.pkl` | `data/flame/FLAME_masks.pkl` |
| `mediapipe_landmark_embedding.npz` | `data/flame/mediapipe_landmark_embedding.npz` |

---

## Quickstart

```bash
bash start.sh
```

This opens a menu to run any stage. Or run stages directly:

```bash
bash start_stage1.sh   # Capture
bash start_stage2.sh   # FLAME Tracking
```

---

## Stage 1 — Capture

**Script:** [capture.py](capture.py)  
**Launcher:** [start_stage1.sh](start_stage1.sh)

Preprocesses a raw talking-head video into training-ready data:

- Detects the face in every frame using MediaPipe
- Computes a single fixed square crop that covers the face in all frames (keeps camera intrinsics constant — critical for 3D geometry)
- Exports cropped frames as PNGs at a fixed resolution
- Extracts audio as 16 kHz mono WAV

```bash
python capture.py \
  --video data/capture/IMG_9625.MOV \
  --name  take1 \
  --out   data/processed \
  --size  512 \
  --fps   30
```

**Output** at `data/processed/take1/`:
```
frames/000000.png ...   # square face crops
audio.wav               # 16 kHz mono
audio_full.wav          # original sample rate
meta.json               # fps, crop box, image size, per-frame face boxes
```

**Recording protocol for best results:**
- 2 minutes of natural talking
- 30 seconds of slow head rotation (provides multi-view signal)
- 30 seconds of exaggerated mouth movements (trains expression range)

---

## Stage 2 — FLAME Tracking (VHAP)

**Tracker:** [VHAP](https://github.com/ShenhanQian/VHAP) — Versatile Head Alignment with Adaptive Appearance Priors  
**Launcher:** [start_stage2.sh](start_stage2.sh)

Fits the FLAME 2023 head model to every frame using **photometric optimization** — minimizing the difference between a rendered mesh and the actual pixel colors, not just landmark positions. This produces accurate camera poses and geometry that Stage 3 (3DGS training) requires.

Three-step process:

1. **Preprocess** — extract frames from video + separate foreground with `robust_video_matting`
2. **Track** — photometric FLAME fitting per frame (full perspective camera, 30 global optimization epochs)
3. **Export** — write `transforms.json` + per-frame FLAME params in NeRF/3DGS format

### Install (one-time)

Handled automatically by `bash setup_linux.sh` — it adds the VHAP submodule and creates the `vhap` conda env. If you need to do it manually:

```bash
git submodule add https://github.com/ShenhanQian/VHAP.git vhap
git submodule update --init --recursive

conda create --name vhap -y python=3.10
conda activate vhap
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
conda env config vars set CUDA_HOME=$CONDA_PREFIX
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/ShenhanQian/nvdiffrast.git
pip install -e vhap/
```

### Run

```bash
bash start_stage2.sh
```

Or manually:

```bash
conda activate vhap

python vhap/vhap/preprocess_video.py \
  --input data/capture/IMG_9625.MOV \
  --matting_method robust_video_matting

python vhap/vhap/track.py \
  --data.root_folder data/source/input_videos \
  --exp.output_folder data/processed/take1_vhap \
  --data.sequence take1

python vhap/vhap/export_as_nerf_dataset.py \
  --src_folder data/processed/take1_vhap \
  --tgt_folder data/processed/take1_nerf
```

**Output** at `data/processed/take1_nerf/`:
```
transforms.json           # camera intrinsics + extrinsics per frame (NeRF/3DGS format)
transforms_train.json
transforms_val.json
images/                   # matted frames (white background)
fg_masks/                 # foreground alpha masks
flame_param/              # per-frame .npz (shape, expr, rotation, pose, translation)
canonical.obj             # FLAME neutral mesh
canonical_flame_param.npz
```

---

## Stage 3 — Splat Training (not built)

Trains a 3D Gaussian Splatting avatar from the tracked frames and FLAME geometry.

**Planned approach** (based on [ELITE](https://kim-youwang.github.io/elite), CVPR 2026):
- Initialize Gaussians from FLAME mesh via a Mesh2Gaussian Prior Model
- Fine-tune with `diff-surfel-rasterization` (surface-based, better normals than volume-based 3DGS)
- Input: `transforms.json` + `flame_param/` from Stage 2 (VHAP output)
- Output: trained `.ply` Gaussian avatar

---

## Stage 4 — Audio-to-Motion (not built)

Converts speech audio into per-frame blendshape curves (expression + jaw parameters).

- Input: audio file (WAV/MP3)
- Output: same parameter format as Stage 2 — composable with Stage 5

---

## Stage 5 — Inference (not built)

Drives the trained Gaussian avatar with audio or text to render a talking-head video.

- Input: text or audio + trained avatar from Stage 3
- Output: MP4 talking-head video

---

## File reference

| File | Purpose |
|------|---------|
| [capture.py](capture.py) | Stage 1 — video preprocessing |
| [vhap/](vhap/) | Stage 2 — VHAP photometric FLAME tracker (submodule) |
| [track.py](track.py) | Legacy landmark-only tracker (superseded by VHAP) |
| [flame.py](flame.py) | FLAME 2023 differentiable forward model |
| [landmarks.py](landmarks.py) | MediaPipe Face Landmarker wrapper (468-pt) |
| [verify_env.py](verify_env.py) | Environment check (CUDA, deps, assets) |
| [setup_linux.sh](setup_linux.sh) | Automated environment setup |
| [start.sh](start.sh) | Main menu launcher |
| [start_stage1.sh](start_stage1.sh) | Stage 1 launcher |
| [start_stage2.sh](start_stage2.sh) | Stage 2 launcher |
| [requirements.txt](requirements.txt) | Python dependencies |
| [avatar-project-handover.md](avatar-project-handover.md) | Full technical spec |
| [avatar-training-machine-setup.md](avatar-training-machine-setup.md) | Machine setup guide |

---

## Data directory layout

```
data/
  capture/
    IMG_9625.MOV          # original recording
  flame/
    flame2023.pkl         # FLAME 2023 model (download required)
    FLAME_masks.pkl       # vertex region masks (download required)
    mediapipe_landmark_embedding.npz
  processed/
    take1/
      frames/             # Stage 1 output — cropped PNG frames
      audio.wav
      audio_full.wav
      meta.json
    take1_vhap/           # Stage 2 VHAP intermediate (tracked params)
    take1_nerf/           # Stage 2 final output (NeRF/3DGS format)
      transforms.json
      transforms_train.json
      transforms_val.json
      images/
      fg_masks/
      flame_param/
      canonical.obj
      canonical_flame_param.npz
```

---

## References

- [FLAME 2023](https://flame.is.tue.mpg.de) — parametric head model
- [VHAP](https://github.com/ShenhanQian/VHAP) — photometric FLAME tracker
- [ELITE](https://kim-youwang.github.io/elite) — Efficient Gaussian Head Avatar (CVPR 2026)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — real-time radiance field rendering
- [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) — surface-based 3DGS rasterizer
