"""
2D landmark detection using MediaPipe Face Mesh (478 points).

Provides a single function:
    detect_landmarks(image_rgb) -> np.ndarray (478, 2) in pixel coords
                                   or None if no face found

Also provides:
    detect_landmarks_batch(frame_paths) -> list of (478,2) or None per frame
"""
from __future__ import annotations

import urllib.request
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise ImportError("mediapipe missing: pip install mediapipe")

# MediaPipe Face Mesh gives 478 landmarks (468 face + 10 iris)
# We use only the 468 face landmarks (indices 0-467)
N_LANDMARKS = 468

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_CACHE = Path(tempfile.gettempdir()) / "face_landmarker.task"


def _get_landmarker() -> mp_vision.FaceLandmarker:
    if not _MODEL_CACHE.exists():
        print("Downloading MediaPipe Face Landmarker model (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_MODEL_CACHE)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


# Module-level singleton (lazy)
_landmarker: mp_vision.FaceLandmarker | None = None


def _get_singleton() -> mp_vision.FaceLandmarker:
    global _landmarker
    if _landmarker is None:
        _landmarker = _get_landmarker()
    return _landmarker


def detect_landmarks(image_rgb: np.ndarray,
                     image_wh: tuple[int, int] | None = None
                     ) -> np.ndarray | None:
    """
    Detect 468 2D face landmarks in pixel coordinates.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 RGB image
    image_wh  : (width, height) — inferred from image if None

    Returns
    -------
    np.ndarray (468, 2) in pixel (x, y) coords, or None if no face found
    """
    landmarker = _get_singleton()
    h, w = image_rgb.shape[:2]
    if image_wh is not None:
        w, h = image_wh

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        return None

    # Take the first (and only) face
    lm_list = result.face_landmarks[0][:N_LANDMARKS]
    pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)
    return pts  # (468, 2)


def detect_landmarks_from_path(frame_path: str | Path) -> np.ndarray | None:
    """Load a PNG frame and detect landmarks."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return detect_landmarks(rgb)


def detect_landmarks_batch(frame_paths: list[Path],
                            verbose: bool = True) -> list[np.ndarray | None]:
    """
    Detect landmarks on a list of frame paths.
    Returns a list of (468, 2) arrays or None per frame.
    """
    from tqdm import tqdm
    results = []
    it = tqdm(frame_paths, desc="detecting landmarks", unit="frame") if verbose else frame_paths
    for p in it:
        results.append(detect_landmarks_from_path(p))
    return results
