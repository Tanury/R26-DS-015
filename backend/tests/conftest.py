from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch

    torch.set_num_threads(1)
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
