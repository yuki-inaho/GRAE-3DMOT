"""Pytest bootstrap: make the repository root importable.

A ``conftest.py`` at the repo root makes pytest add the root to ``sys.path`` so
``import ops`` / ``import models`` / ``import utils`` resolve regardless of the
working directory pytest is launched from.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
