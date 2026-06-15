"""Dataset-level constants shared by the legacy loaders.

``base/base_dataset.py`` imports :data:`NuScenesClasses` from ``utils.data_util``,
but the file was missing from the beta archive.  The class map is centralised
here, using the exact numeric labels already used by ``tools/convert_dataset.py``.
"""

from __future__ import annotations

# Tracking class -> contiguous label id (must match tools/convert_dataset.py).
NuScenesClasses = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "bus": 3,
    "motorcycle": 4,
    "trailer": 5,
    "truck": 6,
}
