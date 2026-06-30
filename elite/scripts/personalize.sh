#!/bin/bash
set -e

# =================================================================
# Usage: bash scripts/personalize.sh {id} {gpu}
# Configure all paths in configs/paths.sh before running.
# =================================================================

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 {id} [gpu_device]"
    exit 1
fi
id=$1
gpu=${2:-0}
exp_name="${id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_ROOT}/configs/paths.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# === Environment ===
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$gpu
export CUDA_HOME

source "${CONDA_INIT_SCRIPT}"
conda activate "${CONDA_ENV}"

# === Paths ===
EXP_DIR="${EXP_ROOT}/${exp_name}"

# === Logging ===
LOG_DIR="${LOG_ROOT}/personalize_${id}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="$LOG_DIR/${id}_${TIMESTAMP}.log"
start_time=$(date +%s)
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================="
echo "Running ELITE personalization"
echo "EXP_NAME=$exp_name | ID=$id | GPU=$gpu"
echo "EXP_DIR=$EXP_DIR"
echo "Log: $MASTER_LOG"
echo "============================================="

# === Resolve REF_IMAGE ===
REF_IMAGE="${SINGLEVIEW_PRC_ROOT}/${id}_whiteBg_staticOffset_maskBelowLine/images/00000_00.png"

# === Auto-detect resolution from source images ===
_SRC_IMG_DIR="${SINGLEVIEW_PRC_ROOT}/${id}_whiteBg_staticOffset_maskBelowLine/images"
if [ ! -d "$_SRC_IMG_DIR" ]; then
    echo "ERROR: source image directory not found: $_SRC_IMG_DIR"
    exit 1
fi
PERSONALIZE_RES=$(python -c "
from PIL import Image
import os, glob
imgs = sorted(glob.glob(os.path.join('$_SRC_IMG_DIR', '*.png')))
assert imgs, 'No PNG images found in $_SRC_IMG_DIR'
w, h = Image.open(imgs[0]).size
print('802x550' if h > 600 else '512x512')
")
echo "Auto-detected resolution: ${PERSONALIZE_RES} (from ${_SRC_IMG_DIR})"

# === Step 1: Personalize with real images ===
echo "[Step 1] personalize with real images - $(date)"
python src/personalize.py \
    --stage 1 \
    --exp_path "${EXP_DIR}" \
    --tgt_id "$id" \
    --prior_cfg "${ELITE_CFG}" \
    --prior_ckpt "${ELITE_CKPT}" \
    --res "${PERSONALIZE_RES}" \
    --log_wandb False \
    --eval_last_n_frames 64

# === Step 2: Generate random expression images ===
echo "[Step 2] save images using random expressions - $(date)"
python src/infer_randexpr.py \
    --num_frames 100 \
    --person_id "$id" \
    --res "${PERSONALIZE_RES}" \
    --cfg_file "${EXP_DIR}/st1/config.yaml" \
    --ckpt_file "${EXP_DIR}/st1/checkpoints/st1_final.pth"

# === Step 3: Run DIFIX to generate synthetic image dataset ===
echo "[Step 3] run difix to generate synthetic image datasets - $(date)"
python hufix/src/enhance_randexpr_dataset.py \
    --ref_image "${REF_IMAGE}" \
    --input_image "${EXP_DIR}/rand_expr_dataset" \
    --save_dir "${EXP_DIR}/rand_expr_dataset" \
    --model_path "${HUFIX_CKPT}"

# === Step 4: Jointly finetune with real + synthetic images ===
echo "[Step 4] jointly finetune with real/synthetic images - $(date)"
python src/personalize.py \
    --stage 2 \
    --exp_path "${EXP_DIR}" \
    --tgt_id "$id" \
    --prior_cfg "${ELITE_CFG}" \
    --prior_ckpt "${ELITE_CKPT}" \
    --res "${PERSONALIZE_RES}" \
    --eval_last_n_frames 64

echo "============================================="
echo "Finished personalization for ID=$id"
echo "============================================="
