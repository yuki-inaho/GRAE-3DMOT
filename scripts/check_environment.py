"""Smoke-check the uv-managed GRAE-3DMOT environment.

Verifies that the core dependencies import, that PyTorch sees a CUDA 12.8 GPU,
and that the self-contained ops (BEV IoU + SimpleTrack NMS) run.  Does NOT touch
nuScenes data or start training.

Usage:
    uv run python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORE_MODULES = [
    "numpy",
    "scipy",
    "shapely",
    "pyquaternion",
    "torch",
    "torchvision",
    "pandas",
    "matplotlib",
    "tensorboard",
    "lap",
    "fvcore",
]


def _check_imports() -> list[str]:
    failures = []
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    for name in CORE_MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"[OK]   {name}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"[FAIL] {name}: {exc.__class__.__name__}: {exc}")
            failures.append(name)
    return failures


def _check_cuda() -> list[str]:
    import torch
    import torch.version  # noqa: F401  (ensure torch.version submodule is resolved)

    failures = []
    print(f"\nPyTorch CUDA build : {torch.version.cuda}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.version.cuda is None or not torch.version.cuda.startswith("12.8"):
        print("[WARN] expected a CUDA 12.8 build (required for Blackwell / sm_120 GPUs)")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"CUDA device        : {torch.cuda.get_device_name(idx)}")
        print(f"Compute capability : {torch.cuda.get_device_capability(idx)}")
        try:
            x = torch.randn(512, 512, device="cuda")
            float((x @ x.t()).sum())
            print("[OK]   GPU matmul")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"[FAIL] GPU matmul: {exc}")
            failures.append("gpu-matmul")
    else:
        print("[WARN] no CUDA device visible; training/eval will fall back to CPU")
    return failures


def _check_self_contained_ops() -> list[str]:
    import torch

    failures = []
    print("\nSelf-contained ops:")
    try:
        from ops.iou3d_nms_cuda import boxes_iou_bev, boxes_iou_bev_cpu

        a = torch.tensor([[0, 0, 0, 2, 2, 1, 0.0]])
        b = torch.tensor([[1, 0, 0, 2, 2, 1, 0.0]])
        ans = torch.zeros((1, 1))
        boxes_iou_bev_cpu(a, b, ans)
        iou = float(boxes_iou_bev(a, b)[0, 0])
        print(f"[OK]   iou3d BEV IoU (cpu_ref={float(ans[0, 0]):.4f}, torch={iou:.4f})")
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"[FAIL] iou3d: {exc.__class__.__name__}: {exc}")
        failures.append("iou3d")

    try:
        from models.structures.boxes import BBox
        from ops.simpletrack_nms import nms

        d0 = BBox(x=0, y=0, z=0, h=1.5, w=2, l=4, o=0.0)
        d0.s = 0.9
        d1 = BBox(x=0.1, y=0, z=0, h=1.5, w=2, l=4, o=0.0)
        d1.s = 0.5
        keep, _ = nms([d0, d1], [0, 0], threshold_low=0.1)
        print(f"[OK]   SimpleTrack NMS (2 overlapping -> kept {keep})")
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"[FAIL] simpletrack_nms: {exc.__class__.__name__}: {exc}")
        failures.append("simpletrack_nms")
    return failures


def main() -> int:
    failures = _check_imports() + _check_cuda() + _check_self_contained_ops()
    print()
    if failures:
        print("Environment check FAILED for: " + ", ".join(failures))
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
