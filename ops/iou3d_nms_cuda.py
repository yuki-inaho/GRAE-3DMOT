"""Self-contained BEV IoU, a drop-in replacement for CenterPoint's ``iou3d_nms_cuda``.

The original GRAE-3DMOT code only used a single symbol from CenterPoint's
hand-built CUDA extension::

    import iou3d_nms_cuda
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a, boxes_b, ans_iou)

That extension had to be compiled against a specific CUDA toolkit, which made
the project hard to build and impossible to run on newer GPUs (e.g. Blackwell /
sm_120, which requires CUDA 12.8+).  This module re-implements the same
behaviour without any compilation step:

* :func:`boxes_iou_bev_cpu` -- an exact, shapely-based reference used by the
  data-preprocessing / target-assignment path (CPU only, matches the original).
* :func:`boxes_iou_bev` -- a vectorised pure-PyTorch implementation that runs on
  both CPU and CUDA tensors (so the op works on GPU under CUDA 12.8).

Box convention matches CenterPoint exactly: each box is
``[x, y, z, dx, dy, dz, heading]`` and the BEV footprint is the rectangle
centred at ``(x, y)`` with axis-aligned half-extents ``(dx/2, dy/2)`` rotated by
``heading`` (counter-clockwise) about the centre.  Only columns ``x, y, dx, dy``
and ``heading`` participate in the BEV IoU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from shapely.geometry import Polygon
from torch import Tensor

__all__ = ["boxes_iou_bev_cpu", "boxes_iou_bev", "boxes_bev_iou"]

_EPS = 1e-9

# Runtime shape/dtype checking: jaxtyping binds the named dims (n, m, p, ...)
# per call and beartype enforces them, so a wrong-shaped tensor raises instead
# of silently producing garbage.
typecheck = jaxtyped(typechecker=beartype)

# Boxes are [x, y, z, dx, dy, dz, heading]; only x, y, dx, dy, heading are used.
Boxes = Float[Tensor, "n 7"]
BoxesA = Float[Tensor, "n 7"] | Float[np.ndarray, "n 7"]
BoxesB = Float[Tensor, "m 7"] | Float[np.ndarray, "m 7"]
IoUMatrix = Float[Tensor, "n m"]


# --------------------------------------------------------------------------- #
# Exact CPU reference (shapely)                                               #
# --------------------------------------------------------------------------- #
def _bev_corners_np(box) -> np.ndarray:
    """Return the 4 BEV corners (CCW) of a single ``[x, y, z, dx, dy, dz, heading]`` box."""
    x, y = float(box[0]), float(box[1])
    dx, dy = float(box[3]), float(box[4])
    angle = float(box[6])
    cos, sin = math.cos(angle), math.sin(angle)
    half = ((-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2))
    return np.array(
        [(x + lx * cos - ly * sin, y + lx * sin + ly * cos) for lx, ly in half],
        dtype=np.float64,
    )


@typecheck
def boxes_iou_bev_cpu(boxes_a: BoxesA, boxes_b: BoxesB, ans_iou: IoUMatrix | None = None) -> IoUMatrix:
    """BEV IoU between ``boxes_a[i]`` and ``boxes_b[j]`` (exact shapely reference).

    Supports both calling conventions:

    * in-place, like CenterPoint's CUDA op -- ``boxes_iou_bev_cpu(a, b, ans_iou)``
      writes into the pre-allocated ``[N, M]`` ``ans_iou`` tensor; and
    * functional -- ``iou = boxes_iou_bev_cpu(a, b)`` returns a new ``[N, M]`` tensor.

    Either way the ``[N, M]`` IoU tensor is returned.

    Args:
        boxes_a: ``[N, 7]`` tensor / array, ``[x, y, z, dx, dy, dz, heading]``.
        boxes_b: ``[M, 7]`` tensor / array.
        ans_iou: optional ``[N, M]`` float tensor written in place.
    """
    a = np.asarray(boxes_a.detach().cpu() if torch.is_tensor(boxes_a) else boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b.detach().cpu() if torch.is_tensor(boxes_b) else boxes_b, dtype=np.float64)
    a = a.reshape(-1, a.shape[-1]) if a.size else a.reshape(0, 7)
    b = b.reshape(-1, b.shape[-1]) if b.size else b.reshape(0, 7)

    polys_a = [Polygon(_bev_corners_np(box)) for box in a]
    polys_b = [Polygon(_bev_corners_np(box)) for box in b]
    areas_a = [p.area for p in polys_a]
    areas_b = [p.area for p in polys_b]

    result = np.zeros((len(polys_a), len(polys_b)), dtype=np.float32)
    for i, pa in enumerate(polys_a):
        for j, pb in enumerate(polys_b):
            inter = pa.intersection(pb).area
            union = areas_a[i] + areas_b[j] - inter
            result[i, j] = inter / union if union > _EPS else 0.0

    if ans_iou is not None:
        ans_iou.copy_(torch.as_tensor(result, dtype=ans_iou.dtype, device=ans_iou.device))
        return ans_iou
    if torch.is_tensor(boxes_a):
        dtype = boxes_a.dtype if boxes_a.is_floating_point() else torch.float32
        return torch.as_tensor(result, dtype=dtype, device=boxes_a.device)
    return torch.from_numpy(result)


# --------------------------------------------------------------------------- #
# Vectorised PyTorch implementation (CPU + CUDA)                              #
# --------------------------------------------------------------------------- #
@typecheck
def _corners_torch(boxes: Float[Tensor, "n 7"]) -> Float[Tensor, "n 4 2"]:
    """``[n, 7]`` boxes -> ``[n, 4, 2]`` CCW BEV corners."""
    x, y = boxes[..., 0], boxes[..., 1]
    dx, dy, ang = boxes[..., 3], boxes[..., 4], boxes[..., 6]
    cos, sin = torch.cos(ang), torch.sin(ang)
    hx, hy = dx / 2, dy / 2
    lx = torch.stack([-hx, hx, hx, -hx], dim=-1)  # [..., 4]
    ly = torch.stack([-hy, -hy, hy, hy], dim=-1)
    cos, sin = cos.unsqueeze(-1), sin.unsqueeze(-1)
    px = x.unsqueeze(-1) + lx * cos - ly * sin
    py = y.unsqueeze(-1) + lx * sin + ly * cos
    return torch.stack([px, py], dim=-1)  # [..., 4, 2]


@typecheck
def _point_in_poly(pts: Float[Tensor, "p k 2"], poly: Float[Tensor, "p 4 2"]) -> Bool[Tensor, "p k"]:
    """Whether ``pts`` lie inside the CCW convex ``poly``; returns ``[p, k]`` bool."""
    v0 = poly.unsqueeze(1)                       # [P, 1, 4, 2]
    v1 = torch.roll(poly, shifts=-1, dims=1).unsqueeze(1)  # [P, 1, 4, 2]
    edge = v1 - v0                               # [P, 1, 4, 2]
    rel = pts.unsqueeze(2) - v0                  # [P, K, 4, 2]
    cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]  # [P, K, 4]
    return (cross >= -1e-6).all(dim=-1)          # CCW: inside => all left turns


@typecheck
def _seg_intersections(
    poly_a: Float[Tensor, "p 4 2"], poly_b: Float[Tensor, "p 4 2"]
) -> tuple[Float[Tensor, "p 16 2"], Bool[Tensor, "p 16"]]:
    """All pairwise edge intersections between two ``[p, 4, 2]`` polygons.

    Returns ``(points [p, 16, 2], valid [p, 16])``.
    """
    a0 = poly_a                                  # [P, 4, 2]
    a1 = torch.roll(poly_a, shifts=-1, dims=1)
    b0 = poly_b
    b1 = torch.roll(poly_b, shifts=-1, dims=1)

    a0 = a0.unsqueeze(2)                          # [P, 4, 1, 2]
    a1 = a1.unsqueeze(2)
    b0 = b0.unsqueeze(1)                          # [P, 1, 4, 2]
    b1 = b1.unsqueeze(1)

    r = a1 - a0                                   # [P, 4, 1, 2]
    s = b1 - b0                                   # [P, 1, 4, 2]
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]  # [P, 4, 4]
    diff = b0 - a0                                # [P, 4, 4, 2]
    t = (diff[..., 0] * s[..., 1] - diff[..., 1] * s[..., 0]) / (denom + (denom == 0) * _EPS)
    u = (diff[..., 0] * r[..., 1] - diff[..., 1] * r[..., 0]) / (denom + (denom == 0) * _EPS)

    valid = (denom.abs() > _EPS) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts = a0 + t.unsqueeze(-1) * r                # [P, 4, 4, 2]
    P = poly_a.shape[0]
    return pts.reshape(P, 16, 2), valid.reshape(P, 16)


@typecheck
def _pairwise_intersection_area(poly_a: Float[Tensor, "p 4 2"], poly_b: Float[Tensor, "p 4 2"]) -> Float[Tensor, "p"]:
    """Convex intersection area for paired polygons ``[p, 4, 2]`` -> ``[p]``."""
    P = poly_a.shape[0]
    in_a = _point_in_poly(poly_a, poly_b)         # corners of A inside B  [P, 4]
    in_b = _point_in_poly(poly_b, poly_a)         # corners of B inside A  [P, 4]
    inter_pts, inter_valid = _seg_intersections(poly_a, poly_b)  # [P, 16, 2], [P, 16]

    points = torch.cat([poly_a, poly_b, inter_pts], dim=1)        # [P, 24, 2]
    valid = torch.cat([in_a, in_b, inter_valid], dim=1)           # [P, 24]

    n_valid = valid.sum(dim=1)                                    # [P]
    # Centroid of valid points (avoid div-by-zero; masked pairs are discarded later).
    cnt = n_valid.clamp(min=1).unsqueeze(-1)
    centroid = (points * valid.unsqueeze(-1)).sum(dim=1) / cnt    # [P, 2]

    ang = torch.atan2(points[..., 1] - centroid[:, 1:2], points[..., 0] - centroid[:, 0:1])  # [P, 24]
    ang = torch.where(valid, ang, torch.full_like(ang, 1e9))      # invalid -> sort to the end
    order = torch.argsort(ang, dim=1)                             # [P, 24]

    sorted_pts = torch.gather(points, 1, order.unsqueeze(-1).expand(-1, -1, 2))
    sorted_valid = torch.gather(valid, 1, order)

    # Collapse every invalid vertex onto the first (smallest-angle, always valid
    # when n_valid>=3) vertex so the shoelace closing edge is correct and all
    # spurious edges contribute zero area.
    first = sorted_pts[:, 0:1, :]
    sorted_pts = torch.where(sorted_valid.unsqueeze(-1), sorted_pts, first)

    nxt = torch.roll(sorted_pts, shifts=-1, dims=1)
    cross = sorted_pts[..., 0] * nxt[..., 1] - nxt[..., 0] * sorted_pts[..., 1]  # [P, 24]
    area = 0.5 * cross.sum(dim=1).abs()
    return torch.where(n_valid >= 3, area, torch.zeros_like(area))


@typecheck
def boxes_iou_bev(boxes_a: Boxes, boxes_b: Float[Tensor, "m 7"], ans_iou: IoUMatrix | None = None) -> IoUMatrix:
    """Vectorised BEV IoU that runs on CPU and CUDA tensors.

    Args:
        boxes_a: ``[N, 7]`` tensor.
        boxes_b: ``[M, 7]`` tensor (same device / dtype as ``boxes_a``).
        ans_iou: optional ``[N, M]`` tensor written in place (CenterPoint-style).

    Returns:
        The ``[N, M]`` IoU tensor (also written into ``ans_iou`` when provided).
    """
    boxes_a = boxes_a.reshape(-1, boxes_a.shape[-1])
    boxes_b = boxes_b.reshape(-1, boxes_b.shape[-1])
    N, M = boxes_a.shape[0], boxes_b.shape[0]
    out = boxes_a.new_zeros((N, M))
    if N == 0 or M == 0:
        if ans_iou is not None:
            ans_iou[...] = out
        return out

    corners_a = _corners_torch(boxes_a.float())   # [N, 4, 2]
    corners_b = _corners_torch(boxes_b.float())   # [M, 4, 2]

    pa = corners_a.unsqueeze(1).expand(N, M, 4, 2).reshape(N * M, 4, 2)
    pb = corners_b.unsqueeze(0).expand(N, M, 4, 2).reshape(N * M, 4, 2)

    inter = _pairwise_intersection_area(pa, pb).reshape(N, M)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4]).abs().reshape(N, 1)
    area_b = (boxes_b[:, 3] * boxes_b[:, 4]).abs().reshape(1, M)
    union = area_a + area_b - inter
    iou = torch.where(union > _EPS, inter / union, torch.zeros_like(inter)).clamp_(0.0, 1.0)
    iou = iou.to(out.dtype)

    if ans_iou is not None:
        ans_iou[...] = iou
    return iou


# Backwards-compatible alias (some forks call it ``boxes_bev_iou``).
boxes_bev_iou = boxes_iou_bev
