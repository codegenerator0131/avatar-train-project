#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_ROOT}/configs/paths.sh"
cd "${PROJECT_ROOT}"

# =================================================================
# Configuration
# =================================================================
DATA_ROOT="data/source/input_videos"
TRACKED_OUTPUT_ROOT="data/source/tracked"
PROCESSED_OUTPUT_ROOT="data/source/processed"
LOG_DIR="logs"

# =================================================================
# Environment Setup
# =================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export CUDA_HOME
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

source "${CONDA_INIT_SCRIPT}"
conda activate "${CONDA_ENV}"

# =================================================================
# Argument Parsing
# =================================================================
ID=$1
if [ -z "$ID" ]; then
  echo "How to use: $0 <ID> [GPU_INDEX]"
  exit 1
fi
GPU_INDEX=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_INDEX
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Processing ID: $ID"

# =================================================================
# Log File Setup
# =================================================================
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/track_${ID}_$(date +'%Y%m%d-%H%M%S').log"
echo "Log file created at: $LOG_FILE"
echo "========== STARTING JOB FOR ID: $ID ==========" | tee -a "$LOG_FILE"

# =================================================================
# 1. VHAP Preprocessing
# =================================================================
echo "========== [1/3] VHAP PREPROCESSING ID:$ID ==========" | tee -a "$LOG_FILE"
INPUT_PATH="${DATA_ROOT}/${ID}.mp4"

if [ ! -f "$INPUT_PATH" ]; then
    echo "ERROR: Input file ($INPUT_PATH) not found." | tee -a "$LOG_FILE"
    exit 1
fi

python vhap/vhap/preprocess_video.py \
    --input "$INPUT_PATH" \
    --matting_method robust_video_matting
if [ $? -eq 0 ]; then
    echo "VHAP PREPROCESSING [1/3]: SUCCESS" >> "$LOG_FILE"
else
    echo "VHAP PREPROCESSING [1/3]: FAILED" >> "$LOG_FILE"
    echo "Preprocessing failed. Aborting." | tee -a "$LOG_FILE"
    exit 1
fi

# =================================================================
# 2. VHAP Tracking
# =================================================================
echo "========== [2/3] VHAP TRACKING ID:$ID ==========" | tee -a "$LOG_FILE"
python vhap/vhap/track.py \
  --data.root_folder "$DATA_ROOT" \
  --exp.output_folder "${TRACKED_OUTPUT_ROOT}/${ID}_whiteBg_staticOffset" \
  --data.sequence "${ID}"
if [ $? -eq 0 ]; then
    echo "VHAP TRACKING [2/3]: SUCCESS" >> "$LOG_FILE"
else
    echo "VHAP TRACKING [2/3]: FAILED" >> "$LOG_FILE"
    echo "Tracking failed. Aborting." | tee -a "$LOG_FILE"
    exit 1
fi

# =================================================================
# 3. VHAP Exporting
# =================================================================
echo "========== [3/3] VHAP EXPORTING OUTPUT ID:$ID ==========" | tee -a "$LOG_FILE"
python vhap/vhap/export_as_nerf_dataset.py \
  --src_folder "${TRACKED_OUTPUT_ROOT}/${ID}_whiteBg_staticOffset" \
  --tgt_folder "${PROCESSED_OUTPUT_ROOT}/${ID}_whiteBg_staticOffset_maskBelowLine"
if [ $? -eq 0 ]; then
    echo "VHAP EXPORTING [3/3]: SUCCESS" >> "$LOG_FILE"
else
    echo "VHAP EXPORTING [3/3]: FAILED" >> "$LOG_FILE"
    echo "Exporting failed. Aborting." | tee -a "$LOG_FILE"
    exit 1
fi

echo "========== ALL STEPS COMPLETED SUCCESSFULLY FOR ID: $ID ==========" | tee -a "$LOG_FILE"