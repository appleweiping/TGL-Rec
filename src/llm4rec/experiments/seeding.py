"""Global seeding for deterministic smoke experiments."""

from __future__ import annotations

import os
import random

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised by bare Python smoke commands.
    np = None


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and subprocess hash seeding."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
