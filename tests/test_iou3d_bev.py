"""Tests for the self-contained BEV IoU op (ops/iou3d_nms_cuda.py).

Covers the exact shapely CPU reference (``boxes_iou_bev_cpu``), the vectorised
PyTorch implementation (``boxes_iou_bev``) on CPU, and the GPU path under
CUDA 12.8.
"""

import numpy as np
import pytest
import torch

from ops.iou3d_nms_cuda import boxes_iou_bev, boxes_iou_bev_cpu

CUDA = torch.cuda.is_available()


def _box(x, y, dx, dy, heading, z=0.0, dz=1.0):
    return [x, y, z, dx, dy, dz, heading]


def _random_boxes(n, seed):
    rng = np.random.default_rng(seed)
    boxes = np.zeros((n, 7), dtype=np.float64)
    boxes[:, 0] = rng.uniform(-5, 5, n)  # x
    boxes[:, 1] = rng.uniform(-5, 5, n)  # y
    boxes[:, 2] = rng.uniform(-1, 1, n)  # z
    boxes[:, 3] = rng.uniform(1, 4, n)  # dx
    boxes[:, 4] = rng.uniform(1, 4, n)  # dy
    boxes[:, 5] = rng.uniform(1, 2, n)  # dz
    boxes[:, 6] = rng.uniform(-np.pi, np.pi, n)  # heading
    return boxes


def _shapely_ref(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    out = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float64)
    boxes_iou_bev_cpu(a, b, out)
    return out.numpy()


# --------------------------------------------------------------------------- #
# Exact CPU reference                                                         #
# --------------------------------------------------------------------------- #
def test_identical_box_iou_is_one():
    a = torch.tensor([_box(0, 0, 2, 2, 0)])
    out = torch.zeros((1, 1))
    boxes_iou_bev_cpu(a, a, out)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_disjoint_box_iou_is_zero():
    a = torch.tensor([_box(0, 0, 2, 2, 0)])
    b = torch.tensor([_box(100, 100, 2, 2, 0)])
    out = torch.zeros((1, 1))
    boxes_iou_bev_cpu(a, b, out)
    assert out[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_axis_aligned_known_overlap():
    # A: x in [-1,1], B: x in [0,2]; both y in [-1,1] -> inter 2, union 6 -> 1/3
    a = torch.tensor([_box(0, 0, 2, 2, 0)])
    b = torch.tensor([_box(1, 0, 2, 2, 0)])
    out = torch.zeros((1, 1))
    boxes_iou_bev_cpu(a, b, out)
    assert out[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_bev_iou_cpu_functional_return():
    # ans_iou is optional: calling without it returns the [N, M] matrix.
    a = torch.tensor([_box(0, 0, 2, 2, 0)])
    b = torch.tensor([_box(1, 0, 2, 2, 0)])
    iou = boxes_iou_bev_cpu(a, b)
    assert tuple(iou.shape) == (1, 1)
    assert iou[0, 0].item() == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_bev_iou_centerpoint_axis_convention():
    # Locks the CenterPoint box convention: col[3]=dx is the local-x extent and
    # col[4]=dy the local-y extent (NOT length-along-heading). With dx=2, dy=4:
    #   * shifting +2 along x  -> the dx=2 footprints just touch -> IoU 0
    #   * shifting +2 along y  -> the dy=4 footprints overlap     -> IoU 1/3
    a = torch.tensor([_box(0, 0, 2, 4, 0)])
    shift_x = torch.tensor([_box(2, 0, 2, 4, 0)])
    shift_y = torch.tensor([_box(0, 2, 2, 4, 0)])
    assert boxes_iou_bev_cpu(a, shift_x)[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert boxes_iou_bev_cpu(a, shift_y)[0, 0].item() == pytest.approx(1.0 / 3.0, abs=1e-6)
    # the vectorised torch path must agree
    assert boxes_iou_bev(a, shift_x)[0, 0].item() == pytest.approx(0.0, abs=2e-3)
    assert boxes_iou_bev(a, shift_y)[0, 0].item() == pytest.approx(1.0 / 3.0, abs=2e-3)


def test_cpu_reference_matrix_shape():
    a = torch.tensor(_random_boxes(5, 1), dtype=torch.float32)
    b = torch.tensor(_random_boxes(3, 2), dtype=torch.float32)
    out = torch.zeros((5, 3))
    boxes_iou_bev_cpu(a, b, out)
    assert out.shape == (5, 3)
    assert (out >= 0).all() and (out <= 1).all()


# --------------------------------------------------------------------------- #
# Vectorised torch implementation                                             #
# --------------------------------------------------------------------------- #
def test_torch_matches_shapely_axis_aligned():
    a = torch.tensor([_box(0, 0, 2, 2, 0)], dtype=torch.float32)
    b = torch.tensor([_box(1, 0, 2, 2, 0)], dtype=torch.float32)
    iou = boxes_iou_bev(a, b)
    assert iou[0, 0].item() == pytest.approx(1.0 / 3.0, abs=2e-3)


def test_torch_matches_shapely_random():
    a = _random_boxes(11, 10)
    b = _random_boxes(7, 20)
    ref = _shapely_ref(a, b)
    got = boxes_iou_bev(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)).numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=3e-3)


def test_torch_rotated_box_matches_shapely():
    # axis aligned 2x2 vs a 2x2 rotated 45 deg about the same centre
    a = np.array([_box(0, 0, 2, 2, 0.0)])
    b = np.array([_box(0, 0, 2, 2, np.pi / 4)])
    ref = _shapely_ref(a, b)
    got = boxes_iou_bev(torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(got, ref, atol=3e-3)
    assert 0.5 < got[0, 0] < 1.0  # octagon overlap, sanity bound


def test_torch_in_place_ans_iou():
    a = torch.tensor(_random_boxes(4, 3), dtype=torch.float32)
    b = torch.tensor(_random_boxes(6, 4), dtype=torch.float32)
    ans = torch.zeros((4, 6))
    ret = boxes_iou_bev(a, b, ans)
    np.testing.assert_allclose(ans.numpy(), ret.numpy(), atol=1e-6)


def test_torch_empty_inputs():
    a = torch.zeros((0, 7))
    b = torch.tensor(_random_boxes(3, 5), dtype=torch.float32)
    assert boxes_iou_bev(a, b).shape == (0, 3)
    assert boxes_iou_bev(b, a).shape == (3, 0)


# --------------------------------------------------------------------------- #
# GPU path (CUDA 12.8 / Blackwell)                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_torch_gpu_matches_cpu():
    a = torch.tensor(_random_boxes(13, 30), dtype=torch.float32)
    b = torch.tensor(_random_boxes(9, 40), dtype=torch.float32)
    cpu = boxes_iou_bev(a, b)
    gpu = boxes_iou_bev(a.cuda(), b.cuda())
    assert gpu.is_cuda
    np.testing.assert_allclose(gpu.cpu().numpy(), cpu.numpy(), atol=1e-4)


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_torch_gpu_matches_shapely():
    a = _random_boxes(8, 50)
    b = _random_boxes(8, 60)
    ref = _shapely_ref(a, b)
    gpu = boxes_iou_bev(torch.tensor(a, dtype=torch.float32).cuda(), torch.tensor(b, dtype=torch.float32).cuda())
    np.testing.assert_allclose(gpu.cpu().numpy(), ref, atol=3e-3)
