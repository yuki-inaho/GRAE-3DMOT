"""Tests that the jaxtyping + beartype runtime contracts are actually enforced.

These guard against silently passing wrong-shaped / wrong-typed arguments to the
self-contained ops (a real source of hard-to-debug bugs in geometry code).
"""

import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintViolation

from ops.iou3d_nms_cuda import boxes_iou_bev, boxes_iou_bev_cpu
from ops.simpletrack_nms import nu_array2mot_bbox
from utils.device import resolve_device

# jaxtyping.TypeCheckError subclasses TypeError; beartype param violations are
# BeartypeCallHintViolation. Accept either.
TYPE_ERRORS = (TypeError, BeartypeCallHintViolation)


def test_boxes_iou_bev_accepts_valid_shapes():
    out = boxes_iou_bev(torch.zeros(3, 7), torch.zeros(5, 7))
    assert out.shape == (3, 5)


def test_boxes_iou_bev_rejects_wrong_last_dim():
    with pytest.raises(TYPE_ERRORS):
        boxes_iou_bev(torch.zeros(3, 6), torch.zeros(5, 7))


def test_boxes_iou_bev_rejects_1d():
    with pytest.raises(TYPE_ERRORS):
        boxes_iou_bev(torch.zeros(7), torch.zeros(5, 7))


def test_boxes_iou_bev_cpu_rejects_wrong_ans_shape():
    a, b = torch.zeros(3, 7), torch.zeros(5, 7)
    with pytest.raises(TYPE_ERRORS):
        boxes_iou_bev_cpu(a, b, torch.zeros(3, 4))  # ans must be [3, 5]


def test_boxes_iou_bev_cpu_accepts_numpy_and_tensor():
    # the public CPU function accepts both numpy arrays and tensors
    a_np = np.zeros((2, 7), dtype=np.float64)
    b_t = torch.zeros(4, 7)
    assert tuple(boxes_iou_bev_cpu(a_np, b_t).shape) == (2, 4)


def test_nu_array2mot_bbox_rejects_non_array():
    with pytest.raises(TYPE_ERRORS):
        nu_array2mot_bbox([0, 0, 0, 2, 2, 2, 1, 0, 0, 0])  # python list, not ndarray


def test_resolve_device_rejects_wrong_type():
    with pytest.raises(TYPE_ERRORS):
        resolve_device(requested=123)  # not a str/None
