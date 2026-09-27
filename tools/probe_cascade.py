"""Probe the adaptive-size cascade + cross-row structural penalty on a synthetic case."""
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from lfrd import inpaint

img = np.zeros((64, 64, 3), np.uint8)
img[:] = 120
img[20:44, 20:44] = 200
img[10:20, :] = np.linspace(60, 180, 64)[None, :, None].astype(np.uint8)
hole = np.zeros((64, 64), bool)
hole[26:38, 26:38] = True
depth = np.full((64, 64), 80, np.uint8)
depth[20:44, 20:44] = 200
kw = dict(patch_size=9, search_w=41, search_h=41, return_meta=True)

runs = [
    ("single 9", dict()),
    ("cascade", dict(sizes=(9, 7, 5, 3), beta=35.0)),
    ("cascade+pen8", dict(sizes=(9, 7, 5, 3), beta=35.0, struct_pen=8.0)),
    ("cascade+pen30", dict(sizes=(9, 7, 5, 3), beta=35.0, struct_pen=30.0)),
]
res = {}
for name, extra in runs:
    r = inpaint.inpaint(img, hole, depth, **dict(kw, **extra))
    res[name] = r
    print("%-15s iters %4d  cost %7.1f  size_hist %-28s unfilled %d"
          % (name, r["n_iters"], r["match_cost_mean"], str(r["size_hist"]), r["n_unfilled"]))
print()
print("single == cascade output      :", bool((res["single 9"]["filled"]
                                               == res["cascade"]["filled"]).all()))
print("cascade == cascade+pen8       :", bool((res["cascade"]["filled"]
                                               == res["cascade+pen8"]["filled"]).all()))
