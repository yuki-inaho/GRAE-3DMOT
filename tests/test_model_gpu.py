"""Forward/backward smoke tests for the GRAE model on CPU and GPU.

The synthetic tensor shapes mirror exactly what ``trainer/single_trainer.py``
builds from a frame of detections (see the cat() calls there):

* ``coordinate_info`` : ``[N, 18]``  (center 3 + rotation 4 + size 3 + score 1 + onehot-class 7)
* ``spatial_info``     : ``[N, N, 19]`` (the spatial relation graph + distance)
* ``spatial_dist``     : ``[N, N]``
* ``temporal_info``    : ``[N, T, 12]`` (det_info - track_info)
* ``temporal_dist``    : ``[T, N]``
* ``tracked_feature``  : ``[T, d_model]``
"""

import pytest
import torch

from models.main import GRAE

CUDA = torch.cuda.is_available()
D_MODEL = 128


def _make_inputs(n, t, device, d=D_MODEL):
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)
    return {
        "coordinate_info": mk(n, 18),
        "spatial_info": mk(n, n, 19),
        "spatial_dist": mk(n, n).abs().to(device),
        "temporal_info": mk(n, t, 12),
        "temporal_dist": mk(t, n).abs().to(device),
        "tracked_feature": mk(t, d),
    }


def _run_first_frame(model, x):
    return model(x["coordinate_info"], x["spatial_info"], x["spatial_dist"], x["temporal_info"], first_frame=True)


def _run_temporal(model, x):
    return model(
        x["coordinate_info"],
        x["spatial_info"],
        x["spatial_dist"],
        x["temporal_info"],
        x["temporal_dist"],
        x["tracked_feature"],
        first_frame=False,
    )


def test_model_first_frame_forward_cpu():
    torch.manual_seed(0)
    model = GRAE(in_channels=D_MODEL, layers=3, device="cpu").to("cpu").eval()
    x = _make_inputs(6, 5, "cpu")
    coord, inst, motion = _run_first_frame(model, x)
    assert coord.shape == (6, D_MODEL)
    assert inst.shape == (6, D_MODEL)
    assert motion.shape == (6, D_MODEL)
    assert torch.isfinite(coord).all()


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_model_forward_backward_gpu():
    torch.manual_seed(0)
    n, t = 12, 9
    model = GRAE(in_channels=D_MODEL, layers=3, device="cuda:0").to("cuda:0")
    x = _make_inputs(n, t, "cuda:0")

    # first-frame path (runs under no_grad inside the model)
    coord, inst, motion = _run_first_frame(model, x)
    assert coord.is_cuda and coord.shape == (n, D_MODEL)

    # association path
    out = _run_temporal(model, x)
    coordinate_feature, instance_feature, motion_feature, motion_features, affinity_scores, attention_scores = out
    assert affinity_scores.shape == (3, t, n, 1)
    assert attention_scores.shape == (3, t, 8, n, 1)
    assert motion_features.shape == (3, n, D_MODEL)
    assert affinity_scores.is_cuda
    assert torch.isfinite(affinity_scores).all()

    # backward proves autograd works end-to-end on the GPU
    loss = affinity_scores.sum() + motion_feature.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
