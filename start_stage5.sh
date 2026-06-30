#!/usr/bin/env bash
# Stage 5: Render talking-head video using trained ELITE avatar
set -euo pipefail
trap 'echo ""; echo "ERROR: Stage 5 failed at line $LINENO. Command: $BASH_COMMAND"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"

# Detect ELITE python (miniforge3 → miniconda3 fallback)
if [ -f "/opt/miniforge3/envs/ELITE/bin/python" ]; then
    ELITE_PYTHON="/opt/miniforge3/envs/ELITE/bin/python"
elif [ -f "/venv/ELITE/bin/python" ]; then
    ELITE_PYTHON="/venv/ELITE/bin/python"
else
    ELITE_PYTHON="$HOME/miniconda3/envs/ELITE/bin/python"
fi

echo "================================================"
echo "  Stage 5: Render Avatar Video (ELITE)"
echo "================================================"
echo ""


# --- Inputs
DEFAULT_VIDEO=""
for f in "$SCRIPT_DIR/data/capture/IMG_9625.mov" "$SCRIPT_DIR/data/capture/IMG_9625.MOV"; do
    [ -f "$f" ] && DEFAULT_VIDEO="$f" && break
done
read -e -i "${DEFAULT_VIDEO:-$SCRIPT_DIR/data/capture/IMG_9625.mov}" -p "Input video path (driver): " VIDEO
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Video not found: $VIDEO"
    exit 1
fi

read -e -i "0" -p "GPU index: " GPU
export CUDA_VISIBLE_DEVICES=$GPU

ID="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
echo ""
echo "Person ID: $ID"
echo ""

# --- Motion source: Stage 4 output dir (or leave empty to use original video motion)
echo "Motion source dir (from Stage 4 output, or press Enter to use original video):"
read -e -i "" -p "  Stage 4 output dir (empty = original video): " STAGE4_OUTPUT_DIR

# --- Paths
PROCESSED_DIR="$ELITE_DIR/data/source/processed"
TRACKED_DIR="$ELITE_DIR/data/source/tracked"
PROCESSED_SUFFIX="_whiteBg_staticOffset_maskBelowLine"
PROCESSED_TARGET="$PROCESSED_DIR/${ID}${PROCESSED_SUFFIX}"
EXP_PATH="$ELITE_DIR/outputs/$ID"
ST2_CKPT="$EXP_PATH/st2/checkpoints/st2_final.pth"

# Motion name: use stage4 subdir name if provided, otherwise use video ID
if [ -n "$STAGE4_OUTPUT_DIR" ] && [ -d "$STAGE4_OUTPUT_DIR" ]; then
    MOTION_NAME="stage4_$(basename "$STAGE4_OUTPUT_DIR")"
    USE_STAGE4_MOTION=1
else
    MOTION_NAME="$ID"
    USE_STAGE4_MOTION=0
fi

DRIVE_DIR="$ELITE_DIR/data/drive/${MOTION_NAME}"
VIS_DIR="$EXP_PATH/vis_motion"
SAVE_PATH_RGB="$VIS_DIR/st2_rgb/${MOTION_NAME}"
SAVE_PATH_NRM="$VIS_DIR/st2_nrm/${MOTION_NAME}"
FINAL_RGB="$VIS_DIR/${ID}_rgb_${MOTION_NAME}.mp4"
FINAL_NRM="$VIS_DIR/${ID}_nrm_${MOTION_NAME}.mp4"
CAM_ID="our_cam"

export CONDA_ENV="ELITE"
export CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export ELITE_CFG="$ELITE_DIR/configs/3d_prior.yaml"
export HUFIX_CKPT="$ELITE_DIR/checkpoints/2d_prior.pth"
export SINGLEVIEW_PRC_ROOT="$PROCESSED_DIR"
export SINGLEVIEW_TRACKED_ROOT="$TRACKED_DIR"
export EXP_ROOT="$ELITE_DIR/outputs"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export PYTHONPATH="$ELITE_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export WANDB_MODE=disabled

# --- Check Stage 3 complete
if [ ! -f "$ST2_CKPT" ]; then
    echo "ERROR: Stage 3 not complete. Missing: $ST2_CKPT"
    echo "Run bash start_stage3.sh first."
    exit 1
fi

# ── Step 1/3 — Prepare drive data ─────────────────────────────────────────────
echo "Step 1/3 — Preparing motion drive data..."
echo ""

if [ -f "$DRIVE_DIR/transforms.json" ]; then
    echo "  [skip] Drive data already exists: $DRIVE_DIR"
elif [ "$USE_STAGE4_MOTION" = "1" ]; then
    # Stage 4 already produced transforms.json + flame_param/ — just symlink the drive dir
    echo "  Using Stage 4 motion output: $STAGE4_OUTPUT_DIR"
    mkdir -p "$DRIVE_DIR"
    cp "$STAGE4_OUTPUT_DIR/transforms.json" "$DRIVE_DIR/transforms.json"
    for subdir in images fg_masks flame_param; do
        src="$STAGE4_OUTPUT_DIR/$subdir"
        dst="$DRIVE_DIR/$subdir"
        if [ -d "$src" ] && [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
            echo "  Symlinked $subdir"
        fi
    done
    # Stage 4 transforms use placeholder image paths — symlink from processed for reference images
    for subdir in images fg_masks; do
        dst="$DRIVE_DIR/$subdir"
        if [ ! -e "$dst" ]; then
            ln -s "$PROCESSED_TARGET/$subdir" "$dst" && echo "  Symlinked $subdir from processed"
        fi
    done
    # Fix 6-digit → 5-digit filenames in drive images to match transforms.json paths
    # Also fix shape npz: ensure shape is (1, 300) not (1,) or (1,1,300)
    "$ELITE_PYTHON" - <<PYEOF_FIX
import os, re, numpy as np, glob

for subdir in ['images', 'fg_masks']:
    folder = os.path.join("$DRIVE_DIR", subdir)
    if not os.path.isdir(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        m = re.match(r'^(\d{6})(_\d{2}\.png)$', fname)
        if m:
            new = f"{int(m.group(1)):05d}{m.group(2)}"
            os.rename(os.path.join(folder, fname), os.path.join(folder, new))
    print(f"  [{subdir}] filenames normalized")

# Fix shape in flame_param npz files
flame_dir = os.path.join("$DRIVE_DIR", 'flame_param')
if os.path.isdir(flame_dir):
    npz_files = sorted(glob.glob(os.path.join(flame_dir, '*.npz')))
    if npz_files:
        sample = np.load(npz_files[0])
        shape_val = sample['shape']
        if shape_val.shape != (1, 300):
            # Get correct shape from tracked params
            tracked_base = "$SCRIPT_DIR/data/processed/IMG_9625_nerf"
            # Try to find shape from processed flame_param
            proc_npz = sorted(glob.glob(os.path.join(tracked_base, 'flame_param', '*.npz')))
            if proc_npz:
                correct_shape = np.load(proc_npz[0])['shape'].reshape(1, 300)
            else:
                correct_shape = shape_val.reshape(1, 300)
            for f in npz_files:
                d = dict(np.load(f))
                d['shape'] = correct_shape
                np.savez(f, **d)
            print(f"  [flame_param] shape fixed to (1,300) for {len(npz_files)} files")
        else:
            print(f"  [flame_param] shape already correct (1,300)")
PYEOF_FIX
    FRAME_COUNT=$(python3 -c "import json; d=json.load(open('$DRIVE_DIR/transforms.json')); print(len(d['frames']))")
    echo "  Drive data ready: $FRAME_COUNT frames"
else
    mkdir -p "$DRIVE_DIR"
    "$ELITE_PYTHON" - <<PYEOF
import os, json, re, numpy as np

processed = "$PROCESSED_TARGET"
drive_dir = "$DRIVE_DIR"

tf_path = os.path.join(processed, 'transforms.json')
with open(tf_path) as f:
    transforms_data = json.load(f)

frames = transforms_data['frames']
print(f"  transforms.json has {len(frames)} frames")

# Patch camera_id
for frame in frames:
    frame['camera_id'] = '$CAM_ID'

# Fix 5-digit → 6-digit filenames if needed
first_img = os.path.join(processed, frames[0]['file_path'])
if not os.path.isfile(first_img):
    def fix_path(p):
        return re.sub(r'(\d{5})(_\d{2}\.png)', lambda m: f"{int(m.group(1)):06d}{m.group(2)}", p)
    for frame in frames:
        frame['file_path']    = fix_path(frame['file_path'])
        frame['fg_mask_path'] = fix_path(frame['fg_mask_path'])
    print(f"  Fixed filenames: 5-digit → 6-digit")

drive_tf = dict(transforms_data)
drive_tf['frames'] = frames
with open(os.path.join(drive_dir, 'transforms.json'), 'w') as f:
    json.dump(drive_tf, f, indent=2)

for subdir in ['images', 'fg_masks', 'flame_param']:
    src = os.path.join(processed, subdir)
    dst = os.path.join(drive_dir, subdir)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"  Symlinked {subdir}")

print(f"  Drive data ready: {len(frames)} frames → {drive_dir}")
PYEOF
fi

# ── Step 2/3 — Render frames ───────────────────────────────────────────────────
echo ""
echo "Step 2/3 — Rendering avatar frames..."
echo ""

# Skip if frames exist OR driver video already created by ffmpeg
RENDER_COUNT=$(find "$SAVE_PATH_RGB" -name "*.png" 2>/dev/null | wc -l || echo "0")
DRIVER_VIDEO="$VIS_DIR/vid_drive/driver_${MOTION_NAME}.mp4"
if [ "$RENDER_COUNT" -gt 100 ] || [ -f "$DRIVER_VIDEO" ]; then
    echo "  [skip] Render already done ($RENDER_COUNT frames, driver: $DRIVER_VIDEO)"
else
    cd "$ELITE_DIR"
    mkdir -p "$SAVE_PATH_RGB" "$SAVE_PATH_NRM" "$VIS_DIR/vid_drive/${MOTION_NAME}"

    # Patch render.py: cam_ids and resolution
    "$ELITE_PYTHON" - <<PYEOF2
import re
path = "$ELITE_DIR/src/render.py"
with open(path) as f:
    code = f.read()
orig = code
code = re.sub(r"cam_ids=\[?['\"]?[^,\]]*['\"]?\]?,", "cam_ids=['$CAM_ID'],", code)
code = code.replace("res=(802, 550)", "res=(512, 512)")
code = code.replace("res=(802,550)", "res=(512,512)")
if code != orig:
    with open(path, 'w') as f:
        f.write(code)
    print("  Patched render.py: cam_ids=$CAM_ID, res=512x512")
else:
    print("  render.py already patched")
PYEOF2

    "$ELITE_PYTHON" src/render.py \
        --cfg_file "$EXP_PATH/st2/config.yaml" \
        --ckpt_file "$ST2_CKPT" \
        --person_id "$ID" \
        --motion_samples_dir "$DRIVE_DIR" \
        --stage '2' \
        --cam_path 'motion_id' \
        --save_fps 25 \
        --motion_id "$MOTION_NAME"
fi

# ── Step 3/3 — HuFix post-processing ──────────────────────────────────────────
echo ""
echo "Step 3/3 — Post-processing with HuFix enhancement..."
echo ""

if [ -f "$FINAL_RGB" ]; then
    echo "  [skip] Final video already exists: $FINAL_RGB"
else
    cd "$ELITE_DIR"

    REF_IMAGE=""
    for _f in "$PROCESSED_TARGET/images/"*.png; do
        [ -f "$_f" ] && REF_IMAGE="$_f" && break
    done
    if [ -z "$REF_IMAGE" ]; then
        echo "ERROR: Could not find reference image in $PROCESSED_TARGET/images"
        exit 1
    fi
    echo "  Reference image: $REF_IMAGE"

    # Try HuFix enhancement (requires ~9GB VRAM — skips gracefully if OOM)
    cp "$SCRIPT_DIR/hufix_chunked.py" "$ELITE_DIR/hufix_chunked.py"
    HUFIX_OK=0
    PYTHONPATH="$ELITE_DIR:${PYTHONPATH:-}" "$ELITE_PYTHON" "$ELITE_DIR/hufix_chunked.py" \
        --ref_image "$REF_IMAGE" \
        --input_rgb "$SAVE_PATH_RGB" \
        --input_nrm "$SAVE_PATH_NRM" \
        --save_dir_rgb "${SAVE_PATH_RGB}_difix" \
        --save_dir_nrm "${SAVE_PATH_NRM}_difix" \
        --save_fps 25 \
        --chunk_size 1 \
        --model_path "$HUFIX_CKPT" && HUFIX_OK=1 || echo "  [warn] HuFix OOM — using raw render output instead"

    # Find audio from Stage 4 output
    AUDIO_FILE=""
    if [ -n "$STAGE4_OUTPUT_DIR" ] && [ -f "$STAGE4_OUTPUT_DIR/speech.wav" ]; then
        AUDIO_FILE="$STAGE4_OUTPUT_DIR/speech.wav"
        echo "  Audio: $AUDIO_FILE"
    else
        echo "  [warn] No speech.wav found — output will have no audio"
    fi

    if [ "$HUFIX_OK" = "1" ]; then
        [ -f "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" "$FINAL_RGB"
        [ -f "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" "$FINAL_NRM"
    else
        # Fallback: package raw render frames directly as final output
        echo "  Packaging raw render as final output..."
        if [ -n "$AUDIO_FILE" ]; then
            ffmpeg -y -framerate 25 -i "$SAVE_PATH_RGB/%05d.png" -i "$AUDIO_FILE" \
                -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -shortest "$FINAL_RGB"
            ffmpeg -y -framerate 25 -i "$SAVE_PATH_NRM/%05d.png" -i "$AUDIO_FILE" \
                -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -shortest "$FINAL_NRM"
        else
            ffmpeg -y -framerate 25 -i "$SAVE_PATH_RGB/%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$FINAL_RGB"
            ffmpeg -y -framerate 25 -i "$SAVE_PATH_NRM/%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 "$FINAL_NRM"
        fi
    fi

    # Add audio to HuFix output if needed
    if [ "$HUFIX_OK" = "1" ] && [ -n "$AUDIO_FILE" ] && [ -f "$FINAL_RGB" ]; then
        TEMP_RGB="${FINAL_RGB%.mp4}_audio.mp4"
        ffmpeg -y -i "$FINAL_RGB" -i "$AUDIO_FILE" -c:v copy -c:a aac -shortest "$TEMP_RGB" \
            && mv "$TEMP_RGB" "$FINAL_RGB"
        if [ -f "$FINAL_NRM" ]; then
            TEMP_NRM="${FINAL_NRM%.mp4}_audio.mp4"
            ffmpeg -y -i "$FINAL_NRM" -i "$AUDIO_FILE" -c:v copy -c:a aac -shortest "$TEMP_NRM" \
                && mv "$TEMP_NRM" "$FINAL_NRM"
        fi
        echo "  [ok] Audio muxed into final video"
    fi

    [ -f "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_rgb_${MOTION_NAME}_difix.mp4" "$FINAL_RGB"
    [ -f "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" ] && mv "$VIS_DIR/final_nrm_${MOTION_NAME}_difix.mp4" "$FINAL_NRM"

    # Cleanup intermediate frames
    rm -rf "$SAVE_PATH_RGB" "$SAVE_PATH_NRM" "$VIS_DIR/vid_drive/${MOTION_NAME}"
fi

echo ""
echo "================================================"
echo "  Stage 5 complete!"
echo "================================================"
echo ""
echo "Output videos:"
echo "  RGB:    $FINAL_RGB"
echo "  Normal: $FINAL_NRM"
echo ""
echo "Next: bash start.sh  (choose next stage)"
