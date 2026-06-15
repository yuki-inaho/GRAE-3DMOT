"""Regression tests: pin the exact behaviour of the deterministic ops.

Unlike the cross-checks in ``test_iou3d_bev.py`` (which compare two
implementations), these lock hard-coded golden numbers verified independently
(analytically and by Monte-Carlo) so that *any* future change to the geometry,
the NMS algorithm, or the box-axis convention is caught.
"""

import numpy as np
import pytest
import torch

from models.main import GRAE
from models.structures.boxes import BBox
from ops.iou3d_nms_cuda import boxes_iou_bev, boxes_iou_bev_cpu
from ops.simpletrack_nms import nms

CUDA = torch.cuda.is_available()


def _box(x, y, dx, dy, heading):
    return [x, y, 0.0, dx, dy, 1.0, heading]


# (box_a, box_b, golden IoU, tolerance) — golden values verified by analysis / Monte-Carlo
GOLDEN_BEV_IOU = [
    (_box(0, 0, 2, 2, 0.0), _box(0, 0, 2, 2, 0.0), 1.0, 1e-9),            # identical
    (_box(0, 0, 2, 2, 0.0), _box(1, 0, 2, 2, 0.0), 1.0 / 3.0, 1e-6),     # half-overlap
    (_box(0, 0, 2, 2, 0.0), _box(100, 0, 2, 2, 0.0), 0.0, 1e-9),         # disjoint
    (_box(0, 0, 2, 2, 0.0), _box(0, 0, 2, 2, np.pi / 4), 0.7071, 2e-3),  # 45-deg octagon
    (_box(0, 0, 6, 6, 0.2), _box(0, 0, 2, 2, 0.9), 1.0 / 9.0, 1e-6),     # containment 4/36
]


@pytest.mark.parametrize("a,b,golden,tol", GOLDEN_BEV_IOU)
def test_bev_iou_golden_cpu(a, b, golden, tol):
    iou = boxes_iou_bev_cpu(torch.tensor([a]), torch.tensor([b]))
    assert iou[0, 0].item() == pytest.approx(golden, abs=tol)


@pytest.mark.parametrize("a,b,golden,tol", GOLDEN_BEV_IOU)
def test_bev_iou_golden_torch(a, b, golden, tol):
    iou = boxes_iou_bev(torch.tensor([a], dtype=torch.float32), torch.tensor([b], dtype=torch.float32))
    assert iou[0, 0].item() == pytest.approx(golden, abs=max(tol, 2e-3))


def test_nms_golden_keep_indices():
    # Fixed detection set with a known outcome:
    #  - idx0 (c0, s.9) keeps; idx1 (c0, s.5) overlaps idx0 -> suppressed
    #  - idx2 (c0, s.7) far away -> kept; idx3 (c1, s.8) same spot but other class -> kept
    dets = [
        BBox(x=0.0, y=0.0, z=0.0, h=1.5, w=2.0, l=4.0, o=0.0),
        BBox(x=0.2, y=0.0, z=0.0, h=1.5, w=2.0, l=4.0, o=0.0),
        BBox(x=20.0, y=20.0, z=0.0, h=1.5, w=2.0, l=4.0, o=0.0),
        BBox(x=0.0, y=0.0, z=0.0, h=1.5, w=2.0, l=4.0, o=0.0),
    ]
    for d, s in zip(dets, [0.9, 0.5, 0.7, 0.8]):
        d.s = s
    classes = [0, 0, 0, 1]
    keep, types = nms(dets, classes, threshold_low=0.1)
    assert [int(i) for i in keep] == [0, 3, 2]   # score-descending processing order
    assert [int(t) for t in types] == [0, 1, 0]


def test_bev_iou_axis_convention_golden():
    # CenterPoint convention regression: col[3]=dx is the local-x extent.
    a = torch.tensor([_box(0, 0, 2, 4, 0.0)])
    assert boxes_iou_bev_cpu(a, torch.tensor([_box(2, 0, 2, 4, 0.0)]))[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert boxes_iou_bev_cpu(a, torch.tensor([_box(0, 2, 2, 4, 0.0)]))[0, 0].item() == pytest.approx(1 / 3, abs=1e-6)


def _model_inputs(n, t, device, d=128):
    g = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)
    return (mk(n, 18), mk(n, n, 19), mk(n, n).abs().to(device),
            mk(n, t, 12), mk(t, n).abs().to(device), mk(t, d))


def test_model_output_shapes_and_determinism():
    torch.manual_seed(0)
    n, t, d = 7, 5, 128
    model = GRAE(in_channels=d, layers=3, device="cpu").to("cpu").eval()
    ci, si, sd, ti, td, tf = _model_inputs(n, t, "cpu", d)

    out1 = model(ci, si, sd, ti, td, tf, first_frame=False)
    out2 = model(ci, si, sd, ti, td, tf, first_frame=False)
    _, _, _, motion_features, affinity_scores, attention_scores = out1

    # locked output contract
    assert motion_features.shape == (3, n, d)
    assert affinity_scores.shape == (3, t, n, 1)
    assert attention_scores.shape == (3, t, 8, n, 1)
    assert torch.isfinite(affinity_scores).all()
    # deterministic in eval mode
    assert torch.allclose(out1[4], out2[4], atol=1e-6)
