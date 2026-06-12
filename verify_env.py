#!/usr/bin/env python3
"""Verify the training machine is ready. Run inside the venv: python verify_env.py"""
import shutil
import subprocess
import sys

OK, BAD = "[PASS]", "[FAIL]"
failures = 0


def check(label: str, fn):
    global failures
    try:
        detail = fn()
        print(f"{OK} {label}: {detail}")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"{BAD} {label}: {e}")


def torch_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (driver or wheel mismatch)")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"torch {torch.__version__}, cuda {torch.version.cuda}, {name}, {vram_gb:.1f} GB VRAM"


def cuda_compiler():
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is None: nvcc not found, install cuda-toolkit-12-4")
    return CUDA_HOME


def gpu_tensor_op():
    import torch
    a = torch.randn(2048, 2048, device="cuda")
    b = (a @ a).sum().item()
    return f"matmul on GPU ok (checksum {b:.2e})"


def ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not on PATH")
    out = subprocess.run([path, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    return out


def opencv():
    import cv2
    return f"opencv {cv2.__version__}"


def mediapipe():
    import mediapipe as mp  # noqa: F401
    return f"mediapipe {mp.__version__}"


check("PyTorch + GPU", torch_cuda)
check("CUDA compiler (nvcc)", cuda_compiler)
check("GPU compute test", gpu_tensor_op)
check("ffmpeg", ffmpeg)
check("OpenCV", opencv)
check("MediaPipe", mediapipe)

print()
if failures:
    print(f"{failures} check(s) failed. Fix before proceeding.")
    sys.exit(1)
print("All checks passed. Machine is ready for the capture stage.")
