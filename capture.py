#!/usr/bin/env python3
"""
Stage 1: Capture preprocessing.

Turns a single locked-camera talking-head video into training-ready data:

  data/processed/<name>/
    frames/000000.png ...   square face crops, constant crop box (constant intrinsics)
    audio.wav               16 kHz mono (for the audio model later)
    audio_full.wav          original sample rate stereo/mono
    meta.json               fps, crop box, sizes, per-frame face boxes

Design choice: because the camera is LOCKED, we compute ONE fixed square crop
that contains the face in every frame (union of detections + margin) and apply
it identically to all frames. This keeps the camera intrinsics constant across
the sequence, which the tracking and splat-training stages assume.

Usage:
  python capture.py --video data/capture/take1.mov --name take1 \
      --out data/processed --size 1024 --fps 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    sys.exit("mediapipe missing: pip install mediapipe")


# ----------------------------------------------------------------------------- helpers
def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{res.stderr[-2000:]}")


def video_info(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    s = json.loads(res.stdout)["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": num / den,
        "duration": float(s.get("duration", 0) or 0),
    }


# ----------------------------------------------------------------------------- face box scan
def detect_union_face_box(video: Path, sample_stride: int = 5,
                          min_conf: float = 0.5) -> tuple[np.ndarray, list]:
    """Scan the video, detect the face every `sample_stride` frames, and return
    the union bounding box (x0, y0, x1, y1) in pixels plus per-sample boxes."""
    import urllib.request, tempfile, os

    # Download the MediaPipe face detector model if needed
    model_path = Path(tempfile.gettempdir()) / "blaze_face_short_range.tflite"
    if not model_path.exists():
        print("Downloading MediaPipe face detector model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            model_path)

    base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
    det_opts = mp_vision.FaceDetectorOptions(
        base_options=base_opts,
        min_detection_confidence=min_conf)
    detector = mp_vision.FaceDetector.create_from_options(det_opts)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    union = np.array([w, h, 0, 0], dtype=np.float64)  # x0,y0,x1,y1
    samples, idx, misses = [], 0, 0

    pbar = tqdm(desc="scanning face boxes", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.detections:
                d = max(result.detections,
                        key=lambda d: d.bounding_box.width)
                bb = d.bounding_box
                x0, y0 = float(bb.origin_x), float(bb.origin_y)
                x1, y1 = x0 + float(bb.width), y0 + float(bb.height)
                union[0] = min(union[0], x0)
                union[1] = min(union[1], y0)
                union[2] = max(union[2], x1)
                union[3] = max(union[3], y1)
                samples.append({"frame": idx, "box": [x0, y0, x1, y1]})
            else:
                misses += 1
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    detector.close()

    if not samples:
        raise RuntimeError("no face detected anywhere in the video")
    miss_rate = misses / max(1, len(samples) + misses)
    if miss_rate > 0.25:
        print(f"WARNING: face not detected in {miss_rate:.0%} of sampled frames. "
              f"Extreme poses (full profile, look-up) cause this and it is usually fine, "
              f"but check lighting if the rate is high during frontal talking.")
    return union, samples


def square_crop_from_union(union: np.ndarray, img_w: int, img_h: int,
                           margin: float = 0.45) -> tuple[int, int, int]:
    """Expand the union face box by `margin`, make it square, clamp to image.
    Returns (x, y, side). Margin must be generous: hair, chin during look-up,
    and ears during profile all need to stay inside the crop."""
    x0, y0, x1, y1 = union
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = min(side, img_w, img_h)
    x = int(round(np.clip(cx - side / 2, 0, img_w - side)))
    y = int(round(np.clip(cy - side / 2, 0, img_h - side)))
    return x, y, int(round(side))


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--name", required=True, help="dataset name, e.g. take1")
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--size", type=int, default=1024,
                    help="output crop resolution (square)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame extraction rate; match the source fps")
    ap.add_argument("--margin", type=float, default=0.45)
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not on PATH")

    info = video_info(args.video)
    print(f"input: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['duration']:.1f}s")
    if info["width"] < 1080 and info["height"] < 1080:
        print("WARNING: input below 1080p, expect soft splat detail")
    if abs(info["fps"] - args.fps) > 1:
        print(f"NOTE: extracting at {args.fps} fps while source is {info['fps']:.2f} fps")

    out_dir = args.out / args.name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1) one fixed square crop for the whole sequence
    union, samples = detect_union_face_box(args.video)
    x, y, side = square_crop_from_union(union, info["width"], info["height"], args.margin)
    print(f"fixed crop: x={x} y={y} side={side} -> resized to {args.size}px")

    # 2) extract cropped frames with ffmpeg (fast, color-correct, constant crop)
    print("extracting frames...")
    run(["ffmpeg", "-y", "-i", str(args.video),
         "-vf", f"fps={args.fps},crop={side}:{side}:{x}:{y},scale={args.size}:{args.size}:flags=lanczos",
         "-start_number", "0",
         str(frames_dir / "%06d.png")])
    n_frames = len(list(frames_dir.glob("*.png")))
    print(f"wrote {n_frames} frames")

    # 3) audio: full quality + 16 kHz mono for the audio model
    print("extracting audio...")
    run(["ffmpeg", "-y", "-i", str(args.video), "-vn",
         "-acodec", "pcm_s16le", str(out_dir / "audio_full.wav")])
    run(["ffmpeg", "-y", "-i", str(args.video), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(out_dir / "audio.wav")])

    # 4) metadata for the tracking stage
    meta = {
        "source_video": str(args.video),
        "source": info,
        "extract_fps": args.fps,
        "crop": {"x": x, "y": y, "side": side},
        "output_size": args.size,
        "n_frames": n_frames,
        "face_box_samples": samples,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done. dataset at: {out_dir}")
    print("sanity check: open a few frames, the whole head (hair to chin, both ears "
          "in profile frames) must be inside every crop. If clipped, raise --margin.")


if __name__ == "__main__":
    main()
