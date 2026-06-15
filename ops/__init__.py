"""Self-contained replacements for the external CUDA / research dependencies.

The upstream GRAE-3DMOT README asked users to clone and build two external
projects:

* CenterPoint's ``iou3d_nms`` CUDA op (for BEV IoU based target assignment), and
* SimpleTrack (for the detection NMS used during data pre-processing).

To keep this repository self-contained (no git submodules, no external
``pip install -e`` steps, no hand-built CUDA extensions tied to a specific CUDA
toolkit), those pieces are re-implemented here in pure Python / PyTorch:

* :mod:`ops.iou3d_nms_cuda` -- a drop-in for ``import iou3d_nms_cuda`` exposing
  ``boxes_iou_bev_cpu`` and a CPU/GPU capable ``boxes_iou_bev``.
* :mod:`ops.simpletrack_nms` -- ``nms`` and ``nu_array2mot_bbox`` ported from
  SimpleTrack's ``mot_3d.preprocessing``.
"""

from . import iou3d_nms_cuda  # noqa: F401
from . import simpletrack_nms  # noqa: F401
