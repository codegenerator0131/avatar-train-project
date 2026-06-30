#!/bin/bash
set -e

# =================================================================
# Usage: bash scripts/render_videos.sh {id} {motion_name} [gpu=0] [fps=25]
# {id}: person ID (used as both exp name and person ID)
# {motion_name}: directory name under data/drive/
# Configure all paths in configs/paths.sh before running.
# =================================================================

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 {id} {motion_name} [gpu_device=0] [fps=25]"
    exit 1
fi
id=$1
exp_name=$id
motion_name=$2
gpu=${3:-0}
fps=${4:-25}

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
LOG_DIR="${LOG_ROOT}/render_personalized_avatars_${id}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="$LOG_DIR/${id}_${TIMESTAMP}.log"
start_time=$(date +%s)
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "============================================="
echo "Running rendering pipeline"
echo "EXP_NAME=$exp_name | ID=$id | GPU=$gpu"
echo "Motion: $motion_name | FPS: $fps"
echo "EXP_DIR=$EXP_DIR"
echo "Log: $MASTER_LOG"
echo "Start: $(date)"
echo "============================================="

# === Resolve REF_IMAGE ===
REF_IMAGE="${SINGLEVIEW_PRC_ROOT}/${id}_whiteBg_staticOffset_maskBelowLine/images/00000_00.png"

# === Resolve motion source ===
MOTION_SAMPLES_DIR="${PROJECT_ROOT}/data/drive/${motion_name}"
if [ ! -d "$MOTION_SAMPLES_DIR" ]; then
    echo "ERROR: motion directory not found: $MOTION_SAMPLES_DIR"
    exit 1
fi

# === Render ===
echo "Rendering the personalized prior of $id - $(date)"
python src/render.py \
    --cfg_file "${EXP_DIR}/st2/config.yaml" \
    --ckpt_file "${EXP_DIR}/st2/checkpoints/st2_final.pth" \
    --person_id "$id" \
    --motion_samples_dir "${MOTION_SAMPLES_DIR}" \
    --stage '2' \
    --cam_path 'motion_id' \
    --save_fps "${fps}" \
    --motion_id "${motion_name}"

VIS_MOTION="${EXP_DIR}/vis_motion"

python hufix/src/post_process.py \
    --ref_image "${REF_IMAGE}" \
    --input_rgb "${VIS_MOTION}/st2_rgb/${motion_name}" \
    --input_nrm "${VIS_MOTION}/st2_nrm/${motion_name}" \
    --save_dir_rgb "${VIS_MOTION}/st2_rgb/${motion_name}_difix" \
    --save_dir_nrm "${VIS_MOTION}/st2_nrm/${motion_name}_difix" \
    --save_fps "${fps}" \
    --model_path "${HUFIX_CKPT}"

# === Rename final videos and clean up intermediate files ===
mv "${VIS_MOTION}/final_rgb_${motion_name}_difix.mp4" "${VIS_MOTION}/${id}_rgb_${motion_name}.mp4"
mv "${VIS_MOTION}/final_nrm_${motion_name}_difix.mp4" "${VIS_MOTION}/${id}_nrm_${motion_name}.mp4"
rm -rf "${VIS_MOTION}/st2_rgb"
rm -rf "${VIS_MOTION}/st2_nrm"
rm -rf "${VIS_MOTION}/vid_drive/${motion_name}"

echo "============================================="
echo "Finished rendering: ID=$id / Motion=${motion_name}"
echo "Output:"
echo "  ${VIS_MOTION}/${id}_rgb_${motion_name}.mp4"
echo "  ${VIS_MOTION}/${id}_nrm_${motion_name}.mp4"
echo "  ${VIS_MOTION}/vid_drive/driver_${motion_name}.mp4"
echo "End time: $(date)"
echo "============================================="
