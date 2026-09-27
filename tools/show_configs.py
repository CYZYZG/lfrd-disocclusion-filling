"""Show the code defaults next to the config of each existing run."""
import json
import os
import sys

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

from lfrd.config import RunConfig

c = RunConfig()
print("当前代码默认值:")
for k in ("sizes", "struct_pen", "beta", "size_rule", "temporal_frames", "temporal_q",
          "temporal_tol", "fill_oofa", "patch_size"):
    print("  %-16s %s" % (k, getattr(c, k)))

print()
print("%-16s %-14s %-10s %-10s %-10s" % ("run", "temporal", "fill_oofa", "sizes",
                                         "struct_pen"))
for r in ("ba54_seq", "ba54_temporal", "ba54_oofa", "ba54_best"):
    p = rf"D:\项目\空洞填补2\output\{r}\config.json"
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        print("%-16s %-14s %-10s %-10s %-10s" % (
            r, j.get("temporal_frames"), j.get("fill_oofa"), j.get("sizes"),
            j.get("struct_pen")))
    else:
        print("%-16s (no config.json)" % r)
