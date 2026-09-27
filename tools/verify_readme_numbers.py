"""Verify the numbers quoted in README §7.5 directly from the artefacts."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from lfrd import io_utils, metrics

for run, src, dst in (("q_54", 5, 4), ("q_45", 4, 5), ("q_56", 5, 6),
                      ("q_67", 6, 7), ("q_30", 3, 0)):
    a = {k: [] for k in ("hole", "disocc", "fw", "fo", "ww", "wo", "dw")}
    for fd in sorted(glob.glob(os.path.join(io_utils.run_dir(run, create=False),
                                            f"cam{src}-cam{dst}-f*"))):
        frame = fd.split("-")[-1]
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
        fin = os.path.join(fd, "60_final", "final.png")
        w = os.path.join(fd, "20_warp")
        if not os.path.isfile(fin) or not os.path.isfile(os.path.join(w, "hole_all.png")):
            continue
        ours = io_utils.imread(fin)
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        fil = hole & ~oofa
        a["hole"].append(100 * hole.mean())
        a["disocc"].append(dis.sum())
        a["fw"].append(metrics.psnr(warp, gt, mask=fil))
        a["fo"].append(metrics.psnr(ours, gt, mask=fil))
        a["ww"].append(metrics.psnr(warp, gt))
        a["wo"].append(metrics.psnr(ours, gt))
        dp = os.path.join(io_utils.run_dir(run, create=False), "ablation",
                          os.path.basename(fd), "direct_inpaint.png")
        if os.path.isfile(dp):
            di = io_utils.imread(dp)
            dm = ((di.astype(float) - gt.astype(float)) ** 2).mean(2)
            a["dw"].append(10 * np.log10(255.0 ** 2 / dm[fil].mean()))
    m = lambda k: float(np.mean(a[k])) if a[k] else float("nan")
    print("cam%d->cam%d: n=%d  hole %.1f%%  disocc %.0f px  whole %.2f->%.2f  "
          "filled %.2f->%.2f  direct %.2f  advantage %+.2f"
          % (src, dst, len(a["hole"]), m("hole"), m("disocc"), m("ww"), m("wo"),
             m("fw"), m("fo"), m("dw"), m("fo") - m("dw")))
