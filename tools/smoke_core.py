"""Smoke test for the shared core (calib / warp / io / viz)."""
import sys

import numpy as np

sys.path.insert(0, r"D:\项目\空洞填补2")
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, warp, viz, RunConfig, parse_frames

root = io_utils.DATASET_ROOT_DEFAULT
cams = calib.load_calib(root)
print("cams:", sorted(cams))
print("content shift cam5->cam4:", np.round(calib.content_shift(cams, 5, 4), 2),
      "disocc side:", calib.disocclusion_opens_on(cams, 5, 4))
print("content shift cam5->cam6:", np.round(calib.content_shift(cams, 5, 6), 2),
      "disocc side:", calib.disocclusion_opens_on(cams, 5, 6))

v = io_utils.load_view(root, 5, "f000")
res = calib.describe(cams, 5, 4, v["depth"])
for l in res["lines"]:
    print("  ", l)

w = warp.warp_view(cams, 5, 4, v["color"], v["depth"])
print("warped:", w["warped_color"].shape, w["warped_color"].dtype,
      "hole %%: %.2f" % (w["hole"].mean() * 100))
print("backward idx >=0:", int((w["backward"] >= 0).sum()), "== valid:",
      int((~w["hole"]).sum()))

crack = warp.crack_mask(w["hole"], 2)
filled = warp.fill_cracks(w["warped_color"], crack)
print("crack px:", int(crack.sum()))

# boundary-derived maps
enclosed, oofa = warp.interior_gap(w["hole"], dilate=1)
print("enclosed holes: %d px, oofa: %d px" % (enclosed.sum(), oofa.sum()))

out = io_utils.run_dir("smoke", "warp")
viz.panel([(v["color"], "ref cam5"),
           (w["warped_color"], "warped cam4 (+holes)"),
           (viz.mask_overlay(w["warped_color"], crack, (0, 255, 255)), "cracks"),
           (viz.mask_overlay(w["warped_color"], enclosed, (255, 0, 0)),
            "enclosed holes"),
           (viz.mask_overlay(w["warped_color"], oofa, (0, 0, 255)), "OOFA")],
          path=f"{out}/panel_smoke.png", title="core smoke test")
print("wrote", f"{out}/panel_smoke.png")
