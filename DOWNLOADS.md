# Required Manual Downloads

These files must be downloaded manually (license agreement required, or Google Drive blocks auto-download).
Download them **before running setup or any stage script**.

Run `bash setup_linux.sh` — it will check and tell you exactly what is missing.

---

## Setup (required before ANY stage)

**FLAME model** — requires free registration at https://flame.is.tue.mpg.de

1. Register at https://flame.is.tue.mpg.de
2. Go to **Downloads** tab
3. Download **FLAME 2023** (revised eye region, improved expressions — 103 MB) → unzip → rename inner `.pkl` to `flame2023.pkl`
4. Download **FLAME Vertex Masks** (1.1 MB) → unzip → rename inner `.pkl` to `FLAME_masks.pkl`
5. Place both files:

```
data/flame/flame2023.pkl
data/flame/FLAME_masks.pkl
```

`setup_linux.sh` will exit with a clear error if these are missing.

---

## Stage 3 — ELITE Avatar (3D Gaussian Splatting)

Source: https://drive.google.com/drive/folders/1GKVymlwRi9shK0G2Qi5JrOFfkIdyUaHM

| File | Destination | Size |
|------|-------------|------|
| `3d_prior.pth` | `elite/checkpoints/3d_prior.pth` | ~4.9 GB |
| `2d_prior.pth` | `elite/checkpoints/2d_prior.pth` | ~415 MB |

The stage script checks for these and exits with a clear error if missing.

---

## Stage 4 — CodeTalker (Audio-to-Motion)

Source: https://github.com/Doubiiu/CodeTalker

| File | Google Drive | Destination | Size |
|------|-------------|-------------|------|
| `vocaset_stage1.pth.tar` | https://drive.google.com/file/d/1RszIMsxcWX7WPlaODqJvax8M_dnCIzk5 | `stage4/CodeTalker/vocaset/vocaset_stage1.pth.tar` | ~50 MB |
| `vocaset_stage2.pth.tar` | https://drive.google.com/file/d/1phqJ_6AqTJmMdSq-__KY6eVwN4J9iCGP | `stage4/CodeTalker/vocaset/vocaset_stage2.pth.tar` | ~300 MB |

The stage script will attempt auto-download via gdown first.
If that fails, download manually and upload to the server via scp:

```bash
scp vocaset_stage1.pth.tar reblium@<server-ip>:~/Documents/avatar-train-project/stage4/CodeTalker/vocaset/
scp vocaset_stage2.pth.tar reblium@<server-ip>:~/Documents/avatar-train-project/stage4/CodeTalker/vocaset/
```

---

## Everything else is auto-downloaded

| Model | Stage | Downloaded by |
|-------|-------|---------------|
| XTTS v2 (voice cloning) | Stage 4 | `TTS` pip package (automatic) |
| Wav2Vec2 (facebook/wav2vec2-base-960h) | Stage 4 | HuggingFace Hub (automatic) |
| RobustVideoMatting | Stage 2 | Script (automatic) |
| FLAME model | Stage 2/3/4 | Included in vhap/ELITE (automatic) |
