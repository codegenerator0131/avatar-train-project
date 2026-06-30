#!/bin/bash
# =================================================================
# configs/paths.sh — User-configurable paths for all pipeline scripts
#
# Usage: source configs/paths.sh
# Edit the variables below to match your environment.
# =================================================================

# -----------------------------------------------------------------
# Environment
# -----------------------------------------------------------------
export CONDA_ENV="ELITE"                          # conda env name or full path to venv
export CONDA_INIT_SCRIPT="${HOME}/anaconda3/etc/profile.d/conda.sh"
export CUDA_HOME="/usr/local/cuda-11.8"

# -----------------------------------------------------------------
# Pretrained weights
# -----------------------------------------------------------------
# Download instructions: see README.md
export ELITE_CFG="configs/3d_prior.yaml"
export ELITE_CKPT="checkpoints/3d_prior.pth"
export HUFIX_CKPT="checkpoints/2d_prior.pth"

# -----------------------------------------------------------------
# Source video pipeline data roots
# (output of scripts/process_source_video.sh)
# -----------------------------------------------------------------
export SINGLEVIEW_PRC_ROOT="data/source/processed"
export SINGLEVIEW_TRACKED_ROOT="data/source/tracked"

# -----------------------------------------------------------------
# Experiment output
# -----------------------------------------------------------------
export EXP_ROOT="outputs"
export LOG_ROOT="logs"

