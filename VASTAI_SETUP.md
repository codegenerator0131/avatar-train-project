# Vast.ai Setup Guide — Stage 5 HuFix

Run the final talking-head video render on a rented GPU.
Estimated cost: **< $1** (1–2 hrs on RTX 4090 @ $0.35/hr)

---

## 1. Rent a GPU on Vast.ai

1. Go to https://vast.ai and log in
2. Click **Search** → filter:
   - **GPU**: RTX 4090
   - **VRAM**: 24 GB
   - **Disk**: 50 GB minimum
   - **Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` (or Ubuntu 22.04 + CUDA 12.1)
3. Click **Rent** on the cheapest available instance (~$0.35/hr)
4. Wait for status to show **Running**, then click **Connect** to get SSH command

---

## 2. Connect via SSH

Vast.ai gives you a command like:
```bash
ssh -p 12345 root@ssh.vast.ai
```
Run that in your terminal.

---

## 3. Upload project files to the instance

From your **local machine** (or the Linux server), run:

```bash
# Set your Vast.ai SSH details
VAST_HOST="ssh.vast.ai"
VAST_PORT="12345"   # ← replace with your instance port
VAST_USER="root"
PROJECT="~/Documents/avatar-train-project"

# Upload entire project (excluding large checkpoints we'll re-download)
rsync -avz --progress \
  --exclude='elite/outputs/*/st1/' \
  --exclude='elite/outputs/*/st2/' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -e "ssh -p $VAST_PORT" \
  "$PROJECT/" \
  "$VAST_USER@$VAST_HOST:/workspace/avatar-train-project/"
```

Then upload the trained avatar checkpoint separately (it's 4.9 GB):
```bash
# Upload Stage 3 trained avatar (required for Stage 5)
rsync -avz --progress \
  -e "ssh -p $VAST_PORT" \
  "$PROJECT/elite/outputs/IMG_9625/st2/checkpoints/st2_final.pth" \
  "$VAST_USER@$VAST_HOST:/workspace/avatar-train-project/elite/outputs/IMG_9625/st2/checkpoints/"

# Upload Stage 4 motion output
rsync -avz --progress \
  -e "ssh -p $VAST_PORT" \
  "$PROJECT/data/stage4/IMG_9625/" \
  "$VAST_USER@$VAST_HOST:/workspace/avatar-train-project/data/stage4/IMG_9625/"
```

---

## 4. Files you must manually place (one-time)

Before running setup, upload the FLAME files (require free registration):

```bash
scp -P $VAST_PORT \
  "$PROJECT/data/flame/flame2023.pkl" \
  "$PROJECT/data/flame/FLAME_masks.pkl" \
  "$VAST_USER@$VAST_HOST:/workspace/avatar-train-project/data/flame/"
```

If you don't have them yet, see `DOWNLOADS.md` → Setup section.

---

## 5. Run setup on the Vast.ai instance

SSH into the instance, then:

```bash
cd /workspace/avatar-train-project

# Install NVIDIA driver check (Vast.ai instances have drivers pre-installed)
nvidia-smi   # should show RTX 4090, 24GB

# Run full environment setup (~15–20 min first time)
bash setup_linux.sh
```

> **Note:** Vast.ai instances use `root` by default. If `sudo` prompts fail, just run without `sudo` or prefix commands with `sudo -E`.

---

## 6. Run Stage 5

```bash
cd /workspace/avatar-train-project
bash start_stage5.sh
```

When prompted:
- **Input video path**: press Enter (uses default `IMG_9625.mov`)
- **GPU index**: `0`
- **Stage 4 output dir**: `/workspace/avatar-train-project/data/stage4/IMG_9625/Hello`

Stage 5 has 3 steps:
1. **Prepare drive data** — ~1 min
2. **Render frames** — ~14 min (3071 frames)
3. **HuFix post-processing** — ~30–60 min (this is why we need 24GB)

---

## 7. Download the final video

After Stage 5 completes, the output video is at:
```
elite/outputs/IMG_9625/vis_motion/IMG_9625_rgb_stage4_Hello.mp4
```

Download it:
```bash
scp -P $VAST_PORT \
  "$VAST_USER@$VAST_HOST:/workspace/avatar-train-project/elite/outputs/IMG_9625/vis_motion/IMG_9625_rgb_stage4_Hello.mp4" \
  ~/Desktop/
```

---

## 8. Stop the instance

**Important — stop the instance when done or you keep paying.**

In Vast.ai dashboard → your instance → **Destroy** (or **Stop** to pause).

---

## Summary of what gets uploaded

| What | Size | Why |
|------|------|-----|
| Full project code + scripts | ~50 MB | Scripts, configs, patches |
| `data/flame/flame2023.pkl` | 103 MB | FLAME model (manual download) |
| `data/flame/FLAME_masks.pkl` | 1.1 MB | FLAME masks (manual download) |
| `elite/outputs/IMG_9625/st2/checkpoints/st2_final.pth` | ~4.9 GB | Trained avatar (Stage 3 output) |
| `data/stage4/IMG_9625/Hello/` | ~5 MB | Stage 4 motion (speech + FLAME params) |
| `elite/checkpoints/2d_prior.pth` | ~415 MB | HuFix model checkpoint |

> `3d_prior.pth` (4.9 GB) will be **auto-downloaded** by setup_linux.sh via gdown — no need to upload it manually.
