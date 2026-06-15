"""Device resolution helpers for single-GPU / CPU runs.

Kept dependency-light (only ``torch``) so it can be imported and unit-tested
without pulling in the data-loading stack.
"""

from __future__ import annotations

import torch
from beartype import beartype

__all__ = ["resolve_device"]


@beartype
def resolve_device(
    requested: str | None = None,
    config_device: str = "cuda:0",
    cuda_available: bool | None = None,
    set_device: bool = True,
) -> torch.device:
    """Resolve a torch device, honouring an explicit request over the config default.

    Args:
        requested: explicit device string (e.g. from ``--device``), or ``None``.
        config_device: fallback device string from the config (``arch.args.device``).
        cuda_available: override for ``torch.cuda.is_available()`` (used in tests).
        set_device: when True and the device is CUDA, call ``torch.cuda.set_device``.

    Returns:
        ``torch.device``.

    Raises:
        RuntimeError: if a CUDA device is requested but CUDA is unavailable.
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    req = requested or config_device or "cuda:0"
    if req == "cuda":
        req = "cuda:0"

    device = torch.device(req)
    if device.type == "cuda":
        if not cuda_available:
            raise RuntimeError(
                f"CUDA device {req!r} was requested, but torch.cuda.is_available() is False. "
                f"Install the CUDA 12.8 build of PyTorch (see pyproject.toml) or pass --device cpu."
            )
        if set_device:
            torch.cuda.set_device(device.index or 0)
    return device
