"""How much of the hole does a SECOND reference view resolve?

`tools/oracle_ceiling.py` shows that even a perfect occlusion layer only reaches ~43% of the
disocclusion pixels when a single reference view is used: the rest of the hole is content the
reference camera never captured (outside its frustum, or occluded there as well).

Those pixels are NOT missing from the scene, though -- they are exactly the content the
virtual camera sees and a NEIGHBOURING camera also sees.  This script measures the ceiling
when the unfilled pixels (our OOFA + everything the single-view step could not reach) are
taken from a second reference camera:

    PSNR(ours)                       current
    PSNR(ours + OOFA filled from 2nd view)      + one extra reference view, real content
    PSNR(ours + ALL holes from 2nd view)        information ceiling with 2 views

The last one is an oracle (it uses the second view even for the disocclusions our first view
could resolve), the middle one is a legitimate, implementable improvement.

    python tools/second_view.py --run q_67 --src_cam 6 --dst_cam 7 --second 5 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, warp as warp_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--second", type=int, required=True, help="second reference camera")
    ap.add_argument("--frames", default="0,1,2")
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"run={a.run}  first ref cam{a.src_cam} -> cam{a.dst_cam}, second ref cam{a.second}")
    print(f"{'frame':6s} {'hole px':>9s} {'ours':>7s} {'+OOFA 2nd':>10s} {'+all 2nd':>9s} "
          f"{'luma whole ours':>16s} {'+OOFA':>8s} {'+all':>8s}")
    agg = {k: [] for k in ("ours", "oofa", "all", "w0", "w1", "w2", "oofa_px")}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        root = io_utils.run_dir(a.run, create=False)
        fd = os.path.join(root, f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(fd, "20_warp")
        if not os.path.isdir(w):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        crack = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        region = (dis | crack) & ~oofa
        oofa_only = oofa & ~(dis | crack)

        # warp the SECOND reference view into the virtual camera
        v2 = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.second, frame)
        w2 = warp_mod.warp_view(cams, a.second, a.dst_cam, v2["color"], v2["depth"])
        got = ~w2["hole"]

        with_oofa = np.array(ours, copy=True)
        m1 = oofa_only & got
        with_oofa[m1] = w2["warped_color"][m1]

        with_all = np.array(ours, copy=True)
        m2 = hole & got
        with_all[m2] = w2["warped_color"][m2]

        p = lambda img, mk: metrics.psnr(img, gt, mask=mk)
        # luma-ish whole frame (RGB PSNR on the whole frame keeps it comparable to §7)
        agg["ours"].append(p(ours, region))
        agg["oofa"].append(p(with_oofa, region))
        agg["all"].append(p(with_all, region))
        agg["w0"].append(metrics.psnr(ours, gt))
        agg["w1"].append(metrics.psnr(with_oofa, gt))
        agg["w2"].append(metrics.psnr(with_all, gt))
        agg["oofa_px"].append(int(oofa_only.sum()))
        print(f"{frame:6s} {int(hole.sum()):9d} {agg['ours'][-1]:7.2f} "
              f"{agg['oofa'][-1]:10.2f} {agg['all'][-1]:9.2f} "
              f"{agg['w0'][-1]:16.2f} {agg['w1'][-1]:8.2f} {agg['w2'][-1]:8.2f}")
    if agg["ours"]:
        m = lambda k: float(np.mean(agg[k]))
        print(f"\nMEAN  ours {m('ours'):.2f} | +OOFA-from-2nd-view {m('oofa'):.2f} "
              f"({m('oofa') - m('ours'):+.2f} dB on the fill region)")
        print(f"      whole-frame RGB PSNR: ours {m('w0'):.2f} -> +OOFA {m('w1'):.2f} "
              f"({m('w1') - m('w0'):+.2f}) -> +all holes {m('w2'):.2f} "
              f"({m('w2') - m('w0'):+.2f})")
        print(f"      (OOFA gap filled from the 2nd view: {m('oofa_px'):.0f} px/frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
