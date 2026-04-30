"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy, and set hash seeding for subprocesses.

    PYTHONHASHSEED only affects new Python processes after interpreter startup,
    but writing it here makes spawned commands deterministic by default.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
