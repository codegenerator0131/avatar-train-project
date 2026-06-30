"""
Central path configuration for ELITE inference pipeline.

Environment variables override the defaults listed here.
Shell scripts export these via configs/paths.sh before invoking Python.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------
# Asset paths (bundled in the repo — always relative)
# -----------------------------------------------------------------
ASSET_ROOT = PROJECT_ROOT / "asset"
NS_PRESET_VIEWS_ROOT = ASSET_ROOT / "nersemble_preset_views"
NS_GEO_STATS_ROOT = ASSET_ROOT / "nersemble_geo_stats"
GEN_FLAME_PARAMS_PATH = ASSET_ROOT / "flame" / "gen_flame_params.npz"

# -----------------------------------------------------------------
# Data roots — source video pipeline
# (output of scripts/process_source_video.sh)
# Override with env vars or configs/paths.sh
# -----------------------------------------------------------------
SINGLEVIEW_PRC_ROOT = os.environ.get(
    "SINGLEVIEW_PRC_ROOT",
    str(PROJECT_ROOT / "data/source/processed"),
)
SINGLEVIEW_TRACKED_ROOT = os.environ.get(
    "SINGLEVIEW_TRACKED_ROOT",
    str(PROJECT_ROOT / "data/source/tracked"),
)

# -----------------------------------------------------------------
# Experiment / logging output
# -----------------------------------------------------------------
EXP_ROOT = os.environ.get("EXP_ROOT", str(PROJECT_ROOT / "experiments"))
LOG_PATH = os.environ.get("LOG_PATH", EXP_ROOT)
