"""Compare the three shipped configurations over the 10-frame BA54 sequence."""
import json
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

CFG = [("paper-literal", "base10"),
       ("+ temporal", "ba54_temporal"),
       ("+ temporal + photo-correct", "best_final"),
       ("+ temporal + OOFA (no photo)", "ba54_best"),
       ("FINAL: temporal+OOFA+photo", "best_final")]
print("%-26s %9s %9s %9s %9s %9s" % ("config", "whole PSNR", "whole SSIM",
                                     "filled PSNR", "filled SSIM", "valid PSNR"))
for name, run in CFG:
    with open(rf"D:\项目\空洞填补2\output\{run}\eval\metrics.json", encoding="utf-8") as fh:
        d = json.load(fh)
    g = [r for r in d if "ours_psnr" in r]
    m = lambda k: float(np.mean([r[k] for r in g]))
    print("%-26s %9.3f %9.4f %9.3f %9.4f %9.3f (%d frames)"
          % (name, m("ours_psnr"), m("ours_ssim"), m("ours_psnr_filled"),
             m("ours_ssim_filled"), m("ours_psnr_valid"), len(g)))
    print("%-26s %9s %9s %9s %9s %9s"
          % ("  (plain warp)", "16.759", "0.7416", "7.357", "0.0201", "31.882"))
