#!/usr/bin/env bash
# Avatar pipeline: Vast.ai setup
# Works on any CUDA driver version — uses conda cuda-toolkit 12.1 for compilation
# Same approach as the working Linux server setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELITE_DIR="$SCRIPT_DIR/elite"

echo "================================================"
echo "  Avatar Pipeline — Vast.ai Setup"
echo "================================================"
echo ""

# ── FLAME file check ──────────────────────────────────────────────────────────
for f in "$SCRIPT_DIR/data/flame/flame2023.pkl" "$SCRIPT_DIR/data/flame/FLAME_masks.pkl"; do
    [ -f "$f" ] || { echo "MISSING: $f — upload FLAME files first"; exit 1; }
done
echo "  [ok] FLAME files present"

# ── conda ─────────────────────────────────────────────────────────────────────
CONDA_DIR="/opt/miniforge3"
[ -f "$CONDA_DIR/bin/conda" ] || { echo "ERROR: conda not found at $CONDA_DIR"; exit 1; }
source "$CONDA_DIR/etc/profile.d/conda.sh"
CONDA="$CONDA_DIR/bin/conda"
echo "  [ok] conda at $CONDA_DIR"

# ── system packages ───────────────────────────────────────────────────────────
echo ""
echo "== [1/3] System packages =="
apt-get update -qq || true
for p in build-essential git ffmpeg ninja-build cmake gcc-11 g++-11; do
    dpkg -s "$p" &>/dev/null && echo "  [skip] $p" || apt-get install -y "$p"
done

# ── ELITE env ─────────────────────────────────────────────────────────────────
echo ""
echo "== [2/3] ELITE conda env =="

# Detect env path — Vast.ai uses /venv/<name>
if [ -d "/venv/ELITE" ]; then
    ELITE_ENV="/venv/ELITE"
else
    ELITE_ENV="$CONDA_DIR/envs/ELITE"
fi
ELITE_PIP="$ELITE_ENV/bin/pip"
ELITE_PYTHON="$ELITE_ENV/bin/python"

if "$ELITE_PYTHON" -c "import torch; import diff_surfel_rasterization; import vhap" &>/dev/null 2>&1; then
    echo "  [skip] ELITE env already complete"
else
    # Create env if missing
    [ -f "$ELITE_PYTHON" ] || "$CONDA" create --prefix "$ELITE_ENV" -y python=3.10

    # Install cuda-toolkit 12.1 via conda INTO the env — same as working server
    echo "  Installing cuda-toolkit 12.1 into ELITE env..."
    "$CONDA" install -y --prefix "$ELITE_ENV" -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake

    # Set CUDA_HOME to the env (has nvcc 12.1 now)
    export CUDA_HOME="$ELITE_ENV"
    export PATH="$ELITE_ENV/bin:$PATH"
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

    # Verify nvcc
    "$ELITE_ENV/bin/nvcc" --version

    # pip + base tools
    "$CONDA" install -y --prefix "$ELITE_ENV" pip
    "$ELITE_PIP" install --upgrade pip "setuptools==69.5.1" wheel hatchling editables

    # ELITE requirements (skip torch/chumpy/triton)
    grep -v "^chumpy\|^torch=\|^torchvision=\|^torchaudio=\|^triton=" \
        "$ELITE_DIR/requirements.txt" > /tmp/elite_req.txt
    "$ELITE_PIP" install -r /tmp/elite_req.txt

    "$ELITE_PIP" install --no-build-isolation git+https://github.com/mattloper/chumpy.git

    # PyTorch cu121 — matches cuda-toolkit 12.1 in the env
    echo "  Installing torch 2.1.0+cu121..."
    "$ELITE_PIP" install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121

    # pytorch3d pre-built wheel (py310 + cu121 + pyt210)
    "$ELITE_PIP" install \
        https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

    # diff-surfel-rasterization — compiled against cu121 in env
    echo "  Building diff_surfel_rasterization..."
    CUDA_HOME="$ELITE_ENV" CC=gcc-11 CXX=g++-11 \
        "$ELITE_PIP" install --no-build-isolation \
        git+https://github.com/hbb1/diff-surfel-rasterization.git

    # VHAP
    "$ELITE_PIP" install --no-build-isolation "$ELITE_DIR/vhap/" --no-deps

    # Fix paths.sh
    sed -i "s|anaconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true
    sed -i "s|miniconda3/etc/profile.d/conda.sh|miniforge3/etc/profile.d/conda.sh|g" \
        "$ELITE_DIR/configs/paths.sh" 2>/dev/null || true

    echo "  ELITE env ready."
fi

# ── TTS env ───────────────────────────────────────────────────────────────────
echo ""
echo "== [3/3] TTS conda env =="
if [ -d "/venv/tts" ]; then TTS_ENV="/venv/tts"; else TTS_ENV="$CONDA_DIR/envs/tts"; fi
TTS_PIP="$TTS_ENV/bin/pip"
TTS_PYTHON="$TTS_ENV/bin/python"

if "$TTS_PYTHON" -c "import TTS" &>/dev/null 2>&1; then
    echo "  [skip] tts env already complete"
else
    [ -f "$TTS_PYTHON" ] || "$CONDA" create --prefix "$TTS_ENV" -y python=3.10
    "$TTS_PIP" install --upgrade pip
    "$TTS_PIP" install transformers==4.39.3
    "$TTS_PIP" install torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121
    "$TTS_PIP" install TTS gdown
    echo "  tts env ready."
fi

# ── FLAME files — copy (not symlink) to elite/asset/flame ────────────────────
# Symlinks break when ELITE runs from inside elite/ dir (relative path resolution)
mkdir -p "$ELITE_DIR/asset/flame"
for f in flame2023.pkl FLAME_masks.pkl mediapipe_landmark_embedding.npz \
          head_template_mesh.obj landmark_embedding_with_eyes.npy; do
    src="$SCRIPT_DIR/data/flame/$f"
    dst="$ELITE_DIR/asset/flame/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        echo "  [ok] copied $f to elite/asset/flame/"
    fi
done

# ── vhap/model/flame.py — patch to absolute paths ────────────────────────────
# Installed package uses relative paths which break when run from elite/ dir
FLAME_PY="$ELITE_ENV/lib/python3.10/site-packages/vhap/model/flame.py"
if [ -f "$FLAME_PY" ]; then
    sed -i "s|FLAME_MODEL_PATH = \"asset/flame/flame2023.pkl\"|FLAME_MODEL_PATH = \"$ELITE_DIR/asset/flame/flame2023.pkl\"|" "$FLAME_PY"
    sed -i "s|FLAME_PARTS_PATH = \"asset/flame/FLAME_masks.pkl\"|FLAME_PARTS_PATH = \"$ELITE_DIR/asset/flame/FLAME_masks.pkl\"|" "$FLAME_PY"
    sed -i "s|FLAME_MESH_PATH = \"asset/flame/head_template_mesh.obj\"|FLAME_MESH_PATH = \"$ELITE_DIR/asset/flame/head_template_mesh.obj\"|" "$FLAME_PY"
    sed -i "s|FLAME_LMK_PATH = \"asset/flame/landmark_embedding_with_eyes.npy\"|FLAME_LMK_PATH = \"$ELITE_DIR/asset/flame/landmark_embedding_with_eyes.npy\"|" "$FLAME_PY"
    echo "  [ok] vhap/model/flame.py patched to absolute paths"
else
    echo "  WARNING: $FLAME_PY not found — patch manually after ELITE env is built"
fi

# ── elite/src patches — applied via sed on each new machine ──────────────────
# elite/src/ is now tracked directly in git — patches are already in place after git clone/pull

# test_dataloader.py line ~395: fp['shape'] → fp['shape'][0]
# Why: DataLoader batches shape(1,300) → (B,1,300) = 3D tensor → torch.cat dim mismatch
TEST_DL="$ELITE_DIR/src/dataloader/test_dataloader.py"
if [ -f "$TEST_DL" ]; then
    if grep -q "flame_shape = fp\['shape'\]$" "$TEST_DL"; then
        sed -i "s/        flame_shape = fp\['shape'\]$/        flame_shape = fp['shape'][0]/" "$TEST_DL"
        echo "  [ok] test_dataloader.py: flame_shape = fp['shape'][0]"
    else
        echo "  [skip] test_dataloader.py already patched"
    fi
fi

# test_dataloader.py + finetune_dataloader.py + personalize.py:
# sorted(os.listdir())[-1] picks non-date files (transforms*.json, images/) → filter to date folders only
for f in \
    "$ELITE_DIR/src/dataloader/test_dataloader.py" \
    "$ELITE_DIR/src/dataloader/finetune_dataloader.py" \
    "$ELITE_DIR/src/personalize.py"; do
    [ -f "$f" ] || continue
    if grep -q "sorted(os.listdir" "$f" && ! grep -q "re.match" "$f"; then
        # Add import re if missing
        grep -q "^import re" "$f" || sed -i '1s/^/import re\n/' "$f"
        # Patch sorted(os.listdir(x))[-1] → filter date-format folders
        sed -i "s/sorted(os\.listdir(\([^)]*\)))\[-1\]/sorted([x for x in os.listdir(\1) if re.match(r'^\\\\\d{4}-\\\\\d{2}-\\\\\d{2}', x)])[-1]/g" "$f"
        echo "  [ok] $(basename $f): date-folder filter applied"
    else
        echo "  [skip] $(basename $f): already patched or pattern not found"
    fi
done

# geom.py: valid_mask must be computed AFTER impainting, not before
# Bug: eye/nose UV holes (index_image==-1) get zero opacity → black artifacts
GEOM_PY="$ELITE_DIR/src/utils/geom.py"
if [ -f "$GEOM_PY" ]; then
    if grep -q "valid_mask = index_image\[..., 0\] != -1" "$GEOM_PY" && grep -q "# valid_mask = index_image" "$GEOM_PY"; then
        python3 - <<PYEOF_GEOM
import re
path = "$GEOM_PY"
with open(path) as f:
    code = f.read()
old = '''        # valid_mask = index_image[..., :1] != -1
        valid_mask = index_image[..., 0] != -1
        face_index, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        if impaint:
            if uv_size >= 1024:
                logger.info(
                    "impainting index image might take a while for sizes >= 1024"
                )

            index_image, bary_image = index_image_impaint(
                index_image, bary_image, impaint_threshold
            )
            # TODO: we can avoid doing this 2x
            face_index = index_image_impaint(
                face_index, distance_threshold=impaint_threshold
            )

        self.register_buffer("valid_mask", valid_mask.cpu())'''
new = '''        face_index, bary_image = make_uv_barys(
            self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv
        )
        if impaint:
            if uv_size >= 1024:
                logger.info(
                    "impainting index image might take a while for sizes >= 1024"
                )

            index_image, bary_image = index_image_impaint(
                index_image, bary_image, impaint_threshold
            )
            # TODO: we can avoid doing this 2x
            face_index = index_image_impaint(
                face_index, distance_threshold=impaint_threshold
            )

        # Compute valid_mask AFTER impainting so filled UV holes are included
        valid_mask = index_image[..., 0] != -1
        self.register_buffer("valid_mask", valid_mask.cpu())'''
if old in code:
    with open(path, 'w') as f:
        f.write(code.replace(old, new))
    print("  [ok] geom.py: valid_mask moved after impainting")
else:
    print("  [skip] geom.py: already patched or pattern not found")
PYEOF_GEOM
    else
        echo "  [skip] geom.py: already patched"
    fi
fi

# render.py: cam_ids and resolution
RENDER_PY="$ELITE_DIR/src/render.py"
if [ -f "$RENDER_PY" ]; then
    sed -i "s/cam_ids=\['222200037'\]/cam_ids=['our_cam']/" "$RENDER_PY"
    sed -i "s/res=(802, 550)/res=(512, 512)/" "$RENDER_PY"
    sed -i "s/res=(802,550)/res=(512,512)/" "$RENDER_PY"
    echo "  [ok] render.py: cam_ids=our_cam, res=512x512"
fi

# ── data symlinks ─────────────────────────────────────────────────────────────
PROCESSED_SRC="$SCRIPT_DIR/data/processed/IMG_9625_nerf"
mkdir -p "$ELITE_DIR/data/source/processed" "$ELITE_DIR/data/source/tracked"
if [ -d "$PROCESSED_SRC" ]; then
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/processed/IMG_9625_whiteBg_staticOffset_maskBelowLine"
    ln -sfn "$PROCESSED_SRC" \
        "$ELITE_DIR/data/source/tracked/IMG_9625_whiteBg_staticOffset"
    echo "  [ok] data symlinks created"
else
    echo "  WARNING: $PROCESSED_SRC not found — run Stage 2 first"
fi

echo ""
echo "================================================"
echo "  Setup complete! Run: bash start_stage3.sh or bash start_stage5.sh"
echo "================================================"
