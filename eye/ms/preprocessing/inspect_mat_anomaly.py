"""
inspect_mat_anomaly.py
=======================
Run against one of the 3 subjects whose .mat file failed to parse
(ms14, ms16, ms17) to see what it actually contains, since it's
missing the 'control_pts' key that every other subject has.

Usage:
    python3 inspect_mat_anomaly.py "path/to/ms14_spectralis_macula_v1_s1_R.mat"
"""
import sys

import scipy.io as sio

path = sys.argv[1]
data = sio.loadmat(path)

print(f"=== {path} ===")
print("All top-level keys:")
for k, v in data.items():
    if k.startswith("__"):
        continue
    shape = getattr(v, "shape", None)
    dtype = getattr(v, "dtype", None)
    print(f"  {k}: shape={shape}, dtype={dtype}")
    if hasattr(v, "dtype") and v.dtype.names:
        print(f"    struct fields: {v.dtype.names}")