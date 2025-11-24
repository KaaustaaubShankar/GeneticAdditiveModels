"""Reproducibility helpers: set seeds for common RNGs.

Call `set_global_seed(seed)` early in your program to make runs more
deterministic across Python, NumPy and (if available) PyTorch.
"""
import os
import random
import numpy as _np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python and NumPy RNGs and advise deterministic flags.

    This function attempts to set commonly-used RNG seeds. It also sets
    `PYTHONHASHSEED` which can help deterministic hashing order. If PyTorch
    is available it will also set PyTorch seeds and deterministic flags.

    Note: absolute reproducibility across platforms and library versions
    is not guaranteed, but this reduces run-to-run variance for most uses.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    _np.random.seed(seed)

    # Optional: set PyTorch seeds/flags if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # make cuDNN deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed or failed to configure; ignore silently
        pass
