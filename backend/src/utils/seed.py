from __future__ import annotations

import hashlib
import random

import numpy as np


def stable_hash_int(text: str) -> int:
    """Return a reproducible integer hash independent of Python hash randomization."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
