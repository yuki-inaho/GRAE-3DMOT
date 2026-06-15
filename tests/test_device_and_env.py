"""Tests for device resolution, dataset constants, and the CUDA 12.8 build."""

import pytest
import torch

from utils.data_util import NuScenesClasses
from utils.device import resolve_device

CUDA = torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# resolve_device                                                              #
# --------------------------------------------------------------------------- #
def test_resolve_cpu():
    dev = resolve_device("cpu", cuda_available=False)
    assert dev.type == "cpu"


def test_resolve_bare_cuda_maps_to_index_0():
    dev = resolve_device("cuda", cuda_available=True, set_device=False)
    assert dev.type == "cuda" and (dev.index or 0) == 0


def test_resolve_explicit_overrides_config():
    dev = resolve_device("cpu", config_device="cuda:0", cuda_available=True, set_device=False)
    assert dev.type == "cpu"


def test_resolve_falls_back_to_config():
    dev = resolve_device(None, config_device="cuda:1", cuda_available=True, set_device=False)
    assert dev.type == "cuda" and dev.index == 1


def test_resolve_cuda_unavailable_raises():
    with pytest.raises(RuntimeError):
        resolve_device("cuda:0", cuda_available=False)


# --------------------------------------------------------------------------- #
# Dataset constants                                                           #
# --------------------------------------------------------------------------- #
def test_nuscenes_classes_map():
    assert len(NuScenesClasses) == 7
    assert NuScenesClasses["car"] == 0
    assert NuScenesClasses["truck"] == 6
    assert sorted(NuScenesClasses.values()) == list(range(7))


# --------------------------------------------------------------------------- #
# CUDA 12.8 build                                                             #
# --------------------------------------------------------------------------- #
def test_torch_built_with_cuda_12_8():
    assert torch.version.cuda is not None, "PyTorch must be a CUDA build"
    assert torch.version.cuda.startswith("12.8"), (
        f"expected a CUDA 12.8 PyTorch build (Blackwell/sm_120 support), got {torch.version.cuda}"
    )


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_gpu_is_usable():
    assert torch.cuda.device_count() >= 1
    x = torch.randn(256, 256, device="cuda")
    y = (x @ x.t()).sum()
    assert torch.isfinite(y).item()
    # Blackwell reports compute capability (12, 0); we only require Volta+ here.
    major, _ = torch.cuda.get_device_capability(0)
    assert major >= 7
