"""Tests for the self-contained SimpleTrack NMS port (ops/simpletrack_nms.py)."""

import numpy as np
import pytest
from pyquaternion import Quaternion

from models.structures.boxes import BBox
from ops.simpletrack_nms import nms, nu_array2mot_bbox, weird_bbox


def _mot_box(x, y, l, w, h=1.5, yaw=0.0, z=0.0, score=0.9):
    b = BBox(x=x, y=y, z=z, h=h, w=w, l=l, o=yaw)
    b.s = score
    return b


# --------------------------------------------------------------------------- #
# nu_array2mot_bbox                                                           #
# --------------------------------------------------------------------------- #
def test_nu_array2mot_bbox_identity_quat():
    arr = np.array([1.0, 2.0, 3.0, 1.8, 4.5, 1.6, 1.0, 0.0, 0.0, 0.0, 0.7])
    box = nu_array2mot_bbox(arr)
    assert box.x == pytest.approx(1.0)
    assert box.y == pytest.approx(2.0)
    assert box.z == pytest.approx(3.0)
    assert box.w == pytest.approx(1.8)
    assert box.l == pytest.approx(4.5)
    assert box.h == pytest.approx(1.6)
    assert box.o == pytest.approx(0.0, abs=1e-9)
    assert box.s == pytest.approx(0.7)


def test_nu_array2mot_bbox_yaw_from_quaternion():
    yaw = 0.9
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    arr = np.array([0, 0, 0, 2, 2, 2, q.w, q.x, q.y, q.z])  # no score -> 10 elems
    box = nu_array2mot_bbox(arr)
    assert box.o == pytest.approx(yaw, abs=1e-6)
    assert box.s is None


# --------------------------------------------------------------------------- #
# nms                                                                         #
# --------------------------------------------------------------------------- #
def test_nms_suppresses_overlapping_same_class():
    dets = [
        _mot_box(0.0, 0.0, 4.0, 2.0, score=0.9),
        _mot_box(0.1, 0.0, 4.0, 2.0, score=0.5),  # heavy overlap, lower score
    ]
    keep, types = nms(dets, [0, 0], threshold_low=0.1)
    assert keep == [0]
    assert types == [0]


def test_nms_keeps_different_classes():
    dets = [
        _mot_box(0.0, 0.0, 4.0, 2.0, score=0.9),
        _mot_box(0.1, 0.0, 4.0, 2.0, score=0.5),
    ]
    keep, _ = nms(dets, [0, 1], threshold_low=0.1)
    assert sorted(keep) == [0, 1]


def test_nms_keeps_distant_same_class():
    dets = [
        _mot_box(0.0, 0.0, 4.0, 2.0, score=0.9),
        _mot_box(50.0, 50.0, 4.0, 2.0, score=0.8),
    ]
    keep, _ = nms(dets, [0, 0], threshold_low=0.1)
    assert sorted(keep) == [0, 1]


def test_nms_keeps_highest_score_first():
    dets = [
        _mot_box(0.0, 0.0, 4.0, 2.0, score=0.3),
        _mot_box(0.05, 0.0, 4.0, 2.0, score=0.95),  # higher score, overlaps
    ]
    keep, _ = nms(dets, [0, 0], threshold_low=0.1)
    assert keep == [1]  # the higher-scoring detection survives


def test_nms_skips_weird_bbox():
    good = _mot_box(0.0, 0.0, 4.0, 2.0, score=0.9)
    bad = BBox(x=10.0, y=10.0, z=0.0, h=1.0, w=-2.0, l=4.0, o=0.0)  # negative w
    bad.s = 0.99  # highest score but degenerate
    assert weird_bbox(bad)
    keep, _ = nms([good, bad], [0, 0], threshold_low=0.1)
    assert keep == [0]  # degenerate box dropped, valid one kept
