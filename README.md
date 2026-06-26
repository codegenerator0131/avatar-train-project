# Avatar Train

An in-house pipeline for building fully animatable 3D talking-head avatars from a single short video. Records a person speaking, fits a 3D head model to every frame, trains a 3D Gaussian avatar, then drives it with audio or text to produce a talking-head video.

Inspired by the [Tavus Phoenix](https://www.tavus.io) model. Uses [FLAME 2023](https://flame.is.tue.mpg.de) for face parametrization and [ELITE](https://kim-youwang.github.io/elite) (CVPR 2026) for 3D Gaussian avatar training and rendering.

---

## Pipeline overview

```
Video
  │
  ▼
Stage 1 ── Capture ──────────── frames + audio + meta.json
  │
  ▼
Stage 2 ── FLAME Tracking ───── per-frame FLAME params + transforms.json
  │
  ▼
Stage 3 ── 3DGS Avatar ──────── trained 3D Gaussian avatar (st2_final.pth)
  │
  ▼
Stage 4 ── Audio-to-Motion ───── speech.wav + per-frame FLAME .npz (lip sync)
  │
  ▼
Stage 5 ── Render ─────────────── talking-head video
```

| Stage | Status |
|-------|--------|
| 1 — Capture | ✅ Done |
| 2 — FLAME Tracking (VHAP) | ✅ Done |
| 3 — 3DGS Avatar (ELITE) | ✅ Done |
| 4 — Audio-to-Motion (CodeTalker + XTTS) | ✅ Done |
| 5 — Render (ELITE + HuFix) | ✅ Done |

---

## Hardware requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| GPU | 8 GB VRAM | 24 GB (RTX 4090) | Stage 3/5 train+render on 8GB, HuFix needs 24GB |
| RAM | 16 GB | 32 GB | |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 | |
| CUDA Driver | 12.x+ | 13.x | Pipeline uses conda cuda-toolkit 12.1 internally |
| Python | 3.10 | 3.10 | |

> **HuFix post-processing** (Stage 5 Step 3) requires 24GB VRAM minimum. Use Vast.ai RTX 4090 (~$0.40/hr) for this step.

---

## Setup

### Linux server (local)

```bash
git clone <this-repo>
cd avatar-train-project
git submodule update --init --recursive
bash setup_linux.sh
bash start.sh
```

### Vast.ai (cloud GPU)

```bash
# On Vast.ai instance (use PyTorch 2.1.0 + CUDA 12.1 template)
cd /workspace/avatar-train-project
bash setup_vastai.sh
bash start.sh
```

> Use `setup_vastai.sh` on Vast.ai — NOT `setup_linux.sh`. The linux setup tries to install python3.11 via apt which fails on Ubuntu 24.04 Vast.ai images.

---

## Required downloads (one-time)

### FLAME model files
Register free at [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) and place:

| File | Path |
|------|------|
| `flame2023.pkl` | `data/flame/flame2023.pkl` |
| `FLAME_masks.pkl` | `data/flame/FLAME_masks.pkl` |
| `mediapipe_landmark_embedding.npz` | `data/flame/mediapipe_landmark_embedding.npz` |

### ELITE checkpoints
Download from [Google Drive](https://drive.google.com/drive/folders/1GKVymlwRi9shK0G2Qi5JrOFfkIdyUaHM) and place:

| File | Path | Size |
|------|------|------|
| `3d_prior.pth` | `elite/checkpoints/3d_prior.pth` | 4.9 GB |
| `2d_prior.pth` | `elite/checkpoints/2d_prior.pth` | 415 MB |

---

## Quickstart

```bash
bash start.sh
```

Opens a menu to run any stage. Or run stages directly:

```bash
bash start_stage1.sh   # Capture
bash start_stage2.sh   # FLAME Tracking
bash start_stage3.sh   # 3DGS Avatar Training
bash start_stage4.sh   # Audio-to-Motion
bash start_stage5.sh   # Render
```

---

## Stage 1 — Capture

**Script:** [capture.py](capture.py) | **Launcher:** [start_stage1.sh](start_stage1.sh)

Preprocesses a raw talking-head video:
- Detects face in every frame using MediaPipe
- Computes a fixed square crop (keeps camera intrinsics constant)
- Exports cropped frames as PNGs
- Extracts 16kHz mono WAV audio

**Output:** `data/processed/take1/` — frames/, audio.wav, meta.json

**Recording tips for best results:**
- 2 minutes of natural talking
- 30 seconds of slow head rotation
- 30 seconds of exaggerated mouth movements

---

## Stage 2 — FLAME Tracking (VHAP)

**Launcher:** [start_stage2.sh](start_stage2.sh)

Fits FLAME 2023 head model to every frame via photometric optimization.

**Output:** `data/processed/IMG_9625_nerf/`
```
transforms.json           # camera intrinsics + extrinsics
images/                   # matted frames (white background), 6-digit filenames: 000000_00.png
fg_masks/                 # foreground alpha masks
flame_param/              # per-frame .npz, 5-digit filenames: 00000.npz
```

**Run flags for 8GB VRAM:** `--data.scale-factor 0.5 --batch-size 4`

---

## Stage 3 — 3DGS Avatar Training (ELITE)

**Launcher:** [start_stage3.sh](start_stage3.sh)

Trains a 3D Gaussian Splatting avatar using ELITE (CVPR 2026).

**Recommended steps:** `ST1_N_STEPS=2000`, `ST2_N_STEPS=4000` (~12 hrs on RTX 4090 with `singleview_bs=4`)

**Output:** `elite/outputs/IMG_9625/st2/checkpoints/st2_final.pth`

> **Important:** After training on Vast.ai, download `st2_final.pth` BEFORE destroying the instance. It's ~4-5GB.

### Known issues & fixes (already applied in start_stage3.sh)
- `finetuning.py`: `N_log = 4` → `N_log = min(4, N)` (index OOB with small batch)
- `singleview_bs 4` is optimal for RTX 4090 — bs=6/8 are actually slower
- ELITE dataloaders use `sorted(os.listdir())[-1]` to find tracked run folder — rsync copies stray files (transforms*.json, images/, fg_masks/) into tracked dir which breaks this → script cleans them before running

### Vast.ai extra steps (handled in start_stage3.sh)
- ELITE_PYTHON path: `/opt/miniforge3/envs/ELITE/bin/python`
- FLAME files must be **copied** (not symlinked) to `elite/asset/flame/` — symlinks break when running from inside `elite/` dir
- `vhap/model/flame.py` uses relative paths — must patch to absolute paths:
  ```bash
  sed -i 's|FLAME_MODEL_PATH = "asset/flame/flame2023.pkl"|FLAME_MODEL_PATH = "/workspace/avatar-train-project/elite/asset/flame/flame2023.pkl"|' \
    /opt/miniforge3/envs/ELITE/lib/python3.10/site-packages/vhap/model/flame.py
  sed -i 's|FLAME_PARTS_PATH = "asset/flame/FLAME_masks.pkl"|FLAME_PARTS_PATH = "/workspace/avatar-train-project/elite/asset/flame/FLAME_masks.pkl"|' \
    /opt/miniforge3/envs/ELITE/lib/python3.10/site-packages/vhap/model/flame.py
  sed -i 's|FLAME_MESH_PATH = "asset/flame/head_template_mesh.obj"|FLAME_MESH_PATH = "/workspace/avatar-train-project/elite/asset/flame/head_template_mesh.obj"|' \
    /opt/miniforge3/envs/ELITE/lib/python3.10/site-packages/vhap/model/flame.py
  sed -i 's|FLAME_LMK_PATH = "asset/flame/landmark_embedding_with_eyes.npy"|FLAME_LMK_PATH = "/workspace/avatar-train-project/elite/asset/flame/landmark_embedding_with_eyes.npy"|' \
    /opt/miniforge3/envs/ELITE/lib/python3.10/site-packages/vhap/model/flame.py
  ```

---

## Stage 4 — Audio-to-Motion

**Launcher:** [start_stage4.sh](start_stage4.sh)

Converts text → speech (XTTS v2 voice cloning) → lip-sync FLAME params (CodeTalker).

**Pipeline:** Text → XTTS v2 → speech.wav → CodeTalker VOCASET → mesh vertices → FLAME fitting → per-frame .npz

**Output:** `data/stage4/IMG_9625/{text}/`
```
speech.wav          # cloned voice audio
transforms.json     # camera params
flame_param/        # per-frame .npz: shape(1,300), expr(1,100), jaw_pose(1,3), ...
images/             # reference images (symlinked from processed)
fg_masks/           # reference masks (symlinked from processed)
```

### Known issues (already fixed in start_stage4.sh)
- Stage 4 `expr` values can be 3x out of range vs Stage 2 — causes Gaussians to explode in render → `start_stage5.sh` auto-scales expr to match Stage 2 distribution
- Stage 4 `jaw_pose` can be 25x out of range → `start_stage5.sh` auto-scales jaw to match Stage 2 distribution
- Stage 4 npz `shape` was saving `(1,)` instead of `(1,300)` → fixed in start_stage4.sh

---

## Stage 5 — Render

**Launcher:** [start_stage5.sh](start_stage5.sh)

Renders the trained avatar with Stage 4 lip-sync motion. Optionally applies HuFix (Stable Diffusion-based enhancement).

**Output:** `elite/outputs/IMG_9625/vis_motion/IMG_9625_rgb_stage4_{text}.mp4`

### HuFix
- Requires **24GB VRAM** minimum (RTX 4090 recommended)
- On 8GB GPU: automatically skipped, raw render used instead
- Raw render quality is acceptable without HuFix

### Known issues (already fixed in start_stage5.sh)
- Drive images 6-digit vs transforms.json 5-digit filenames → auto-renamed
- Stage 4 npz `shape (1,)` → auto-fixed to `(1,300)`
- Stage 4 `expr` and `jaw_pose` out of training distribution → auto-scaled

### Manual patch required on each new machine
`elite/src/dataloader/test_dataloader.py` line 395 must be patched:
```bash
sed -i "s/        flame_shape = fp\['shape'\]$/        flame_shape = fp['shape'][0]/" \
  ~/Documents/avatar-train-project/elite/src/dataloader/test_dataloader.py
```
**Why:** DataLoader batches `shape(1,300)` → `(B,1,300)` = 3D tensor, causing `torch.cat([shape, expr], dim=1)` to fail with dimension mismatch.

---

## Vast.ai guide

See [VASTAI_SETUP.md](VASTAI_SETUP.md) for full step-by-step instructions.

**Quick summary:**
1. Rent RTX 4090 24GB instance (~$0.40/hr)
2. rsync project files from Linux server
3. `bash setup_vastai.sh`
4. Patch vhap/model/flame.py (absolute paths — see Stage 3 section above)
5. `bash start_stage3.sh` or `bash start_stage5.sh`
6. Download `st2_final.pth` before destroying instance

---

## File reference

| File | Purpose |
|------|---------|
| [setup_linux.sh](setup_linux.sh) | Full env setup for Linux server |
| [setup_vastai.sh](setup_vastai.sh) | Env setup for Vast.ai cloud GPU |
| [start.sh](start.sh) | Main menu launcher |
| [start_stage1.sh](start_stage1.sh) | Stage 1 — Capture |
| [start_stage2.sh](start_stage2.sh) | Stage 2 — FLAME Tracking |
| [start_stage3.sh](start_stage3.sh) | Stage 3 — 3DGS Avatar Training |
| [start_stage4.sh](start_stage4.sh) | Stage 4 — Audio-to-Motion |
| [start_stage5.sh](start_stage5.sh) | Stage 5 — Render |
| [elite_export_ply.py](elite_export_ply.py) | Export trained avatar as PLY |
| [hufix_chunked.py](hufix_chunked.py) | Chunked HuFix post-processing |
| [capture.py](capture.py) | Stage 1 video preprocessing |
| [VASTAI_SETUP.md](VASTAI_SETUP.md) | Vast.ai step-by-step guide |
| [DOWNLOADS.md](DOWNLOADS.md) | Manual downloads required |

---

## Data directory layout

```
data/
  capture/
    IMG_9625.mov            # original recording
  flame/
    flame2023.pkl           # FLAME 2023 model (download required)
    FLAME_masks.pkl         # vertex region masks (download required)
    mediapipe_landmark_embedding.npz
  processed/
    IMG_9625_nerf/          # Stage 2 output (NeRF/3DGS format)
      transforms.json
      images/               # 6-digit filenames: 000000_00.png
      fg_masks/
      flame_param/          # 5-digit filenames: 00000.npz
    IMG_9625_vhap/          # Stage 2 tracked intermediate (date subfolders)
  stage4/
    IMG_9625/
      Hello_World/          # Stage 4 output for text "Hello, world"
        speech.wav
        transforms.json
        flame_param/        # per-frame FLAME params
        images/             # symlink to processed images
        fg_masks/           # symlink to processed masks

elite/
  checkpoints/
    3d_prior.pth            # ELITE 3DGS prior (4.9GB, download required)
    2d_prior.pth            # HuFix prior (415MB, download required)
  outputs/
    IMG_9625/
      st1/checkpoints/st1_final.pth   # Stage 3 step 1 checkpoint
      st2/checkpoints/st2_final.pth   # Stage 3 step 2 checkpoint (main avatar)
      vis_motion/                      # Stage 5 render output
```

---

## References

- [FLAME 2023](https://flame.is.tue.mpg.de) — parametric head model
- [VHAP](https://github.com/ShenhanQian/VHAP) — photometric FLAME tracker
- [ELITE](https://kim-youwang.github.io/elite) — Efficient Gaussian Head Avatar (CVPR 2026)
- [CodeTalker](https://github.com/Doubiiu/CodeTalker) — speech-driven 3D facial animation
- [XTTS v2](https://github.com/coqui-ai/TTS) — voice cloning TTS
- [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) — surface-based 3DGS rasterizer
