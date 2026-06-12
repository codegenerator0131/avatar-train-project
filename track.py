#!/usr/bin/env python3
"""
Stage 2: FLAME face tracking.

Fits the FLAME 2023 head model to every frame of the Stage 1 output,
producing per-frame parameters:
  - shape      (100,)  identity blendshape weights  [constant across frames]
  - expression (50,)   expression blendshape weights [per frame]
  - global_pose (3,)   head rotation (axis-angle)   [per frame]
  - jaw_pose   (3,)    jaw rotation (axis-angle)    [per frame]
  - camera_scale float  weak-perspective depth scale [per frame]
  - camera_t   (2,)    2D translation in pixels     [per frame]

Output: data/processed/<name>/tracking.json
        data/processed/<name>/tracking_smoothed.json  (Savitzky-Golay filtered)

Usage:
  python track.py --dataset data/processed/take1 \
                  --flame  data/flame/flame2023.pkl \
                  --device cuda

Requirements:
  pip install torch mediapipe opencv-python-headless tqdm scipy
  FLAME 2023 model from https://flame.is.tue.mpg.de  (free registration)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from flame import FLAME
from landmarks import detect_landmarks_batch, detect_landmarks_from_path


# ---------------------------------------------------------------------------
# Camera projection (weak perspective)
# ---------------------------------------------------------------------------

def project_weak_perspective(vertices_3d: torch.Tensor,
                              scale: torch.Tensor,
                              t2d: torch.Tensor,
                              image_size: int) -> torch.Tensor:
    """
    Weak-perspective projection.

    vertices_3d : (B, V, 3)
    scale       : (B,)   depth scale (unitless)
    t2d         : (B, 2) translation in normalised [-1,1] coords
    image_size  : int    square image side in pixels

    Returns
    -------
    (B, V, 2) projected pixel coordinates
    """
    # Project: take X, Y; scale; shift to pixel space
    xy = vertices_3d[:, :, :2]               # (B, V, 2)
    proj = scale[:, None, None] * xy         # (B, V, 2)
    proj = proj + t2d[:, None, :]            # (B, V, 2)  in [-1,1] approx
    # Map from [-1,1] to [0, image_size]
    proj = (proj + 1.0) * (image_size / 2.0)
    return proj                               # (B, V, 2) pixel coords


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def landmark_loss(proj_landmarks: torch.Tensor,
                  detected: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    L2 reprojection loss between projected and detected 2D landmarks.

    proj_landmarks : (B, L, 2)
    detected       : (B, L, 2)
    weights        : (L,) optional per-landmark weight
    """
    diff = proj_landmarks - detected           # (B, L, 2)
    dist = (diff ** 2).sum(dim=-1)             # (B, L)
    if weights is not None:
        dist = dist * weights.unsqueeze(0)
    return dist.mean()


def regularization_loss(params: torch.Tensor, weight: float) -> torch.Tensor:
    return weight * (params ** 2).mean()


# ---------------------------------------------------------------------------
# Per-frame fitter
# ---------------------------------------------------------------------------

class FrameFitter:
    """
    Fits FLAME to a single frame given detected 2D landmarks.
    Optimises: expression, global_pose, jaw_pose, camera_scale, camera_t.
    Shape is fixed (shared across all frames; passed in as a tensor).
    """

    def __init__(self, flame_model: FLAME, image_size: int, device: str):
        self.flame = flame_model
        self.image_size = image_size
        self.device = device

    def fit(self,
            detected_lm: np.ndarray,
            shape_params: torch.Tensor,
            init_expr: Optional[torch.Tensor] = None,
            init_global_pose: Optional[torch.Tensor] = None,
            init_jaw_pose: Optional[torch.Tensor] = None,
            init_scale: Optional[torch.Tensor] = None,
            init_t2d: Optional[torch.Tensor] = None,
            n_iters: int = 300,
            lr: float = 0.01,
            loss_weights: Optional[dict] = None,
            ) -> dict:
        """
        Fit FLAME to one frame.

        Parameters
        ----------
        detected_lm   : (L, 2)  detected landmark pixel coords
        shape_params  : (1, n_shape)  fixed identity params (on device)
        init_*        : warm-start tensors (optional, from previous frame)

        Returns
        -------
        dict with all fitted params as numpy arrays + loss info
        """
        if loss_weights is None:
            loss_weights = {
                "landmark":   1.0,
                "expression": 1e-3,
                "pose":       1e-4,
                "jaw":        1e-4,
                "camera":     1e-5,
            }

        n_expr = self.flame.n_expr
        dev = self.device

        # ---- Learnable parameters ----------------------------------------
        expr = nn.Parameter(
            init_expr.clone() if init_expr is not None
            else torch.zeros(1, n_expr, device=dev))
        global_pose = nn.Parameter(
            init_global_pose.clone() if init_global_pose is not None
            else torch.zeros(1, 3, device=dev))
        jaw_pose = nn.Parameter(
            init_jaw_pose.clone() if init_jaw_pose is not None
            else torch.zeros(1, 3, device=dev))
        scale = nn.Parameter(
            init_scale.clone() if init_scale is not None
            else torch.ones(1, device=dev) * 0.9)
        t2d = nn.Parameter(
            init_t2d.clone() if init_t2d is not None
            else torch.zeros(1, 2, device=dev))

        optimizer = torch.optim.Adam(
            [expr, global_pose, jaw_pose, scale, t2d],
            lr=lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.5)

        # Ground-truth landmarks on device
        gt_lm = torch.from_numpy(detected_lm).float().to(dev).unsqueeze(0)  # (1, L, 2)
        n_flame_lm = self.flame.landmark_indices.shape[0] if self.flame.landmark_indices is not None else 68
        n_gt = gt_lm.shape[1]
        # Use only min(n_flame_lm, n_gt) landmarks
        n_lm = min(n_flame_lm, n_gt)
        gt_lm = gt_lm[:, :n_lm, :]

        best_loss = float("inf")
        best_state = None

        for i in range(n_iters):
            optimizer.zero_grad()

            _, lm3d = self.flame(shape_params, expr, global_pose, jaw_pose)
            lm3d = lm3d[:, :n_lm, :]

            proj_lm = project_weak_perspective(lm3d, scale, t2d, self.image_size)

            loss = (loss_weights["landmark"]   * landmark_loss(proj_lm, gt_lm)
                  + loss_weights["expression"] * (expr ** 2).mean()
                  + loss_weights["pose"]       * (global_pose ** 2).mean()
                  + loss_weights["jaw"]        * (jaw_pose ** 2).mean()
                  + loss_weights["camera"]     * (t2d ** 2).mean())

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Clamp to physically plausible range
            with torch.no_grad():
                expr.clamp_(-3.0, 3.0)
                scale.clamp_(0.1, 5.0)
                jaw_pose[:, 0].clamp_(-0.5, 0.5)   # jaw opens / closes only

            lv = loss.item()
            if lv < best_loss:
                best_loss = lv
                best_state = {
                    "expr":        expr.detach().clone(),
                    "global_pose": global_pose.detach().clone(),
                    "jaw_pose":    jaw_pose.detach().clone(),
                    "scale":       scale.detach().clone(),
                    "t2d":         t2d.detach().clone(),
                }

        s = best_state
        # Reprojection error in pixels
        with torch.no_grad():
            _, lm3d = self.flame(shape_params, s["expr"], s["global_pose"], s["jaw_pose"])
            lm3d = lm3d[:, :n_lm, :]
            proj = project_weak_perspective(lm3d, s["scale"], s["t2d"], self.image_size)
            rms = ((proj - gt_lm) ** 2).sum(-1).sqrt().mean().item()

        return {
            "expression":    s["expr"].cpu().numpy()[0],
            "global_pose":   s["global_pose"].cpu().numpy()[0],
            "jaw_pose":      s["jaw_pose"].cpu().numpy()[0],
            "camera_scale":  s["scale"].cpu().numpy()[0],
            "camera_t":      s["t2d"].cpu().numpy()[0],
            "loss":          best_loss,
            "reprojection_error_px": rms,
        }


# ---------------------------------------------------------------------------
# Shape fitting (shared across all frames)
# ---------------------------------------------------------------------------

def fit_shape(flame_model: FLAME,
              all_landmarks: list[np.ndarray],
              image_size: int,
              device: str,
              n_iters: int = 200,
              lr: float = 0.005) -> torch.Tensor:
    """
    Fit a single shared identity (shape) across a subset of frames.
    Returns shape_params tensor (1, n_shape) on device.
    """
    n_shape = flame_model.n_shape
    n_expr  = flame_model.n_expr

    # Subsample frames for speed (use up to 30 evenly spaced)
    valid = [(i, lm) for i, lm in enumerate(all_landmarks) if lm is not None]
    if len(valid) > 30:
        step = len(valid) // 30
        valid = valid[::step][:30]

    print(f"  Fitting shared identity on {len(valid)} frames...")

    shape = nn.Parameter(torch.zeros(1, n_shape, device=device))
    optimizer = torch.optim.Adam([shape], lr=lr)

    for _ in tqdm(range(n_iters), desc="shape fitting", leave=False):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for _, lm_np in valid:
            gt = torch.from_numpy(lm_np).float().to(device).unsqueeze(0)
            n_flame_lm = flame_model.landmark_indices.shape[0] if flame_model.landmark_indices is not None else 68
            n_lm = min(n_flame_lm, gt.shape[1])
            gt = gt[:, :n_lm, :]

            expr = torch.zeros(1, n_expr, device=device)
            gp   = torch.zeros(1, 3, device=device)
            jp   = torch.zeros(1, 3, device=device)
            sc   = torch.ones(1, device=device) * 0.9
            t2   = torch.zeros(1, 2, device=device)

            _, lm3d = flame_model(shape, expr, gp, jp)
            lm3d = lm3d[:, :n_lm, :]
            proj = project_weak_perspective(lm3d, sc, t2, image_size)
            total_loss = total_loss + landmark_loss(proj, gt)

        total_loss = total_loss / len(valid) + 1e-4 * (shape ** 2).mean()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            shape.clamp_(-3.0, 3.0)

    return shape.detach()


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_tracking(results: list[dict], window: int = 11, polyorder: int = 3) -> list[dict]:
    """Apply Savitzky-Golay smoothing to per-frame parameter curves."""
    from scipy.signal import savgol_filter

    if len(results) < window:
        return results

    keys = ["expression", "global_pose", "jaw_pose", "camera_scale", "camera_t"]
    smoothed = [r.copy() for r in results]

    for key in keys:
        vals = np.array([r[key] for r in results])  # (T, ...) or (T,)
        scalar = vals.ndim == 1
        if scalar:
            vals = vals[:, None]
        for dim in range(vals.shape[1]):
            vals[:, dim] = savgol_filter(vals[:, dim], window_length=window, polyorder=polyorder)
        if scalar:
            vals = vals[:, 0]
        for i, r in enumerate(smoothed):
            r[key] = vals[i]

    return smoothed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Path to Stage 1 output dir (e.g. data/processed/take1)")
    ap.add_argument("--flame", required=True, type=Path,
                    help="Path to flame2023.pkl")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu (default: cuda)")
    ap.add_argument("--n-shape", type=int, default=100,
                    help="Number of shape PCA components (default: 100)")
    ap.add_argument("--n-expr", type=int, default=50,
                    help="Number of expression PCA components (default: 50)")
    ap.add_argument("--iters", type=int, default=300,
                    help="Optimizer iterations per frame (default: 300)")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="Adam learning rate (default: 0.01)")
    ap.add_argument("--smooth-window", type=int, default=11,
                    help="Savitzky-Golay smoothing window (odd number, default: 11)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Limit frames processed (0 = all, useful for testing)")
    args = ap.parse_args()

    # ---- Validate paths ---------------------------------------------------
    if not args.dataset.exists():
        sys.exit(f"Dataset not found: {args.dataset}")
    if not args.flame.exists():
        sys.exit(f"FLAME model not found: {args.flame}\n"
                 f"Download from https://flame.is.tue.mpg.de (free registration)")

    frames_dir = args.dataset / "frames"
    meta_path  = args.dataset / "meta.json"
    if not frames_dir.exists():
        sys.exit(f"frames/ directory not found in {args.dataset}")

    with open(meta_path) as f:
        meta = json.load(f)
    image_size = meta["output_size"]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # ---- Load FLAME -------------------------------------------------------
    print(f"Loading FLAME model from {args.flame}...")
    flame = FLAME(args.flame, n_shape=args.n_shape, n_expr=args.n_expr,
                  device=device).to(device)
    flame.eval()

    # ---- Collect frames ---------------------------------------------------
    frame_paths = sorted(frames_dir.glob("*.png"))
    if args.max_frames > 0:
        frame_paths = frame_paths[:args.max_frames]
    print(f"Found {len(frame_paths)} frames.")

    # ---- Detect 2D landmarks on all frames --------------------------------
    print("Step 1/3 — Detecting 2D landmarks...")
    all_landmarks = detect_landmarks_batch(frame_paths, verbose=True)
    n_detected = sum(1 for lm in all_landmarks if lm is not None)
    n_missing  = len(all_landmarks) - n_detected
    print(f"  Detected: {n_detected}/{len(frame_paths)} frames "
          f"({n_missing} frames with no face — will be skipped)")

    if n_detected == 0:
        sys.exit("No faces detected in any frame. Check lighting / video quality.")

    # ---- Fit shared identity (shape) --------------------------------------
    print("Step 2/3 — Fitting shared identity (shape params)...")
    shape_params = fit_shape(flame, all_landmarks, image_size, device,
                             n_iters=200, lr=0.005)
    print(f"  Shape norm: {shape_params.norm().item():.3f}")

    # ---- Fit per-frame expression + pose ----------------------------------
    print("Step 3/3 — Fitting per-frame expression + pose...")
    fitter = FrameFitter(flame, image_size, device)

    results = []
    prev = {}   # warm-start from previous frame

    for i, (frame_path, lm) in enumerate(
            tqdm(zip(frame_paths, all_landmarks), total=len(frame_paths),
                 desc="tracking frames", unit="frame")):

        if lm is None:
            # Copy previous frame's params if available
            if results:
                entry = results[-1].copy()
                entry["frame_id"] = i
                entry["skipped"]  = True
            else:
                entry = {"frame_id": i, "skipped": True,
                         "expression": np.zeros(args.n_expr).tolist(),
                         "global_pose": np.zeros(3).tolist(),
                         "jaw_pose": np.zeros(3).tolist(),
                         "camera_scale": 0.9,
                         "camera_t": [0.0, 0.0],
                         "loss": None, "reprojection_error_px": None}
            results.append(entry)
            continue

        fit = fitter.fit(
            detected_lm=lm,
            shape_params=shape_params,
            init_expr=prev.get("expr"),
            init_global_pose=prev.get("global_pose"),
            init_jaw_pose=prev.get("jaw_pose"),
            init_scale=prev.get("scale"),
            init_t2d=prev.get("t2d"),
            n_iters=args.iters,
            lr=args.lr,
        )

        # Store warm-start tensors for next frame
        prev = {
            "expr":        torch.from_numpy(fit["expression"]).float().to(device).unsqueeze(0),
            "global_pose": torch.from_numpy(fit["global_pose"]).float().to(device).unsqueeze(0),
            "jaw_pose":    torch.from_numpy(fit["jaw_pose"]).float().to(device).unsqueeze(0),
            "scale":       torch.tensor([fit["camera_scale"]], device=device),
            "t2d":         torch.from_numpy(fit["camera_t"]).float().to(device).unsqueeze(0),
        }

        entry = {
            "frame_id":            i,
            "skipped":             False,
            "shape":               shape_params.cpu().numpy()[0].tolist(),
            "expression":          fit["expression"].tolist(),
            "global_pose":         fit["global_pose"].tolist(),
            "jaw_pose":            fit["jaw_pose"].tolist(),
            "camera_scale":        float(fit["camera_scale"]),
            "camera_t":            fit["camera_t"].tolist(),
            "loss":                fit["loss"],
            "reprojection_error_px": fit["reprojection_error_px"],
            "landmarks_2d":        lm.tolist(),
        }
        results.append(entry)

    # ---- Save raw tracking -----------------------------------------------
    out_raw = args.dataset / "tracking.json"
    with open(out_raw, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw tracking: {out_raw}")

    # ---- Temporal smoothing ----------------------------------------------
    valid_results = [r for r in results if not r.get("skipped")]
    if len(valid_results) > args.smooth_window:
        print(f"Applying Savitzky-Golay smoothing (window={args.smooth_window})...")

        # Build arrays for smoothing (only valid frames)
        def _to_arr(key):
            return np.array([r[key] for r in valid_results])

        from scipy.signal import savgol_filter
        w = args.smooth_window

        for key in ["expression", "global_pose", "jaw_pose", "camera_t"]:
            arr = _to_arr(key)          # (T, D) or (T,)
            scalar = arr.ndim == 1
            if scalar:
                arr = arr[:, None]
            for d in range(arr.shape[1]):
                arr[:, d] = savgol_filter(arr[:, d], window_length=w, polyorder=3)
            if scalar:
                arr = arr[:, 0]
            for j, r in enumerate(valid_results):
                r[key + "_smoothed"] = arr[j].tolist()

        # Propagate smoothed values to skipped frames (copy nearest valid)
        valid_ids = {r["frame_id"]: r for r in valid_results}
        valid_id_list = sorted(valid_ids.keys())
        for r in results:
            if r.get("skipped"):
                # find nearest valid frame
                nearest = min(valid_id_list, key=lambda x: abs(x - r["frame_id"]))
                src = valid_ids[nearest]
                for key in ["expression", "global_pose", "jaw_pose", "camera_t"]:
                    sk = key + "_smoothed"
                    if sk in src:
                        r[sk] = src[sk]

    out_smooth = args.dataset / "tracking_smoothed.json"
    with open(out_smooth, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved smoothed tracking: {out_smooth}")

    # ---- Summary ---------------------------------------------------------
    errors = [r["reprojection_error_px"] for r in results
              if r.get("reprojection_error_px") is not None]
    if errors:
        print(f"\nReprojection error — mean: {np.mean(errors):.2f}px  "
              f"max: {np.max(errors):.2f}px  "
              f"median: {np.median(errors):.2f}px")
        if np.median(errors) > 10:
            print("WARNING: median error > 10px. Consider:")
            print("  - Increasing --iters (e.g. 500)")
            print("  - Checking FLAME model loaded correctly (verify_env.py)")
            print("  - Checking lighting / video quality")

    print("\nStage 2 complete. Output:")
    print(f"  {out_raw}")
    print(f"  {out_smooth}")
    print("Pass tracking_smoothed.json to Stage 3 (splat training).")


if __name__ == "__main__":
    main()
