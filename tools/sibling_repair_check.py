"""Is the sibling's crack / under-covered repair good, or does it just rewrite correct pixels?

The sibling fills ~9.5k "crack" px and ~47k "under-covered" px, the latter being pixels the warp
DID produce.  tools/head_to_head.py runs it on the same warp, so this measures per pixel crowd:

  * the pixels the sibling changed but the warp already had (its extra repairs)
  * split by whether the sibling's value is better or worse than the warp
  * and the same for our own crack set, so the two detection criteria can be compared

A repair mechanism worth porting must make those pixels better on average; the earlier coarse
check (blocks near the hole) said -0.03 dB, which is why this looks at every changed pixel.

    python tools/sibling_repair_check.py --run ba54_seq --src_cam 5 --dst_cam 4 `
        --frames 0,1,2 --tag ba54
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def luma(img):
    a = np.asarray(img, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def mae(gt, img, mask):
    if not np.asarray(mask, bool).any():
        return float("nan")
    return float(np.abs(luma(gt) - luma(img))[mask].mean())


def lpsnr(gt, img, mask):
    if not np.asarray(mask, bool).any():
        return float("nan")
    d = ((luma(gt) - luma(img)) ** 2)[mask]
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    h2h = os.path.join(io_utils.run_dir("_h2h", create=False), a.tag)
    if not os.path.isdir(h2h):
        h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                           f"{a.tag}_cam{a.src_cam}{a.dst_cam}")
    print(f"{'frame':6s} {'extra chg':>9s} {'better':>7s} {'worse':>7s} {'mae warp':>9s} "
          f"{'mae sib':>8s} {'delta':>7s} | {'our crack':>9s} {'warp mae':>9s} "
          f"{'sib mae':>8s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isfile(fp):
            print(f"[skip] {frame}")
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        gl = luma(gt)
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        sib = io_utils.imread(fp)
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        valid = ~hole
        chg = (sib.astype(np.int16) != warp.astype(np.int16)).any(2)
        extra = chg & valid                       # pixels the warp already produced
        e_warp = np.abs(gl - luma(warp))
        e_sib = np.abs(gl - luma(sib))
        better = extra & (e_sib < e_warp - 1e-9)
        worse = extra & (e_sib > e_warp + 1e-9)
        m0, m1 = mae(gt, warp, extra), mae(gt, sib, extra)
        # our own crack set: what the sibling does to it
        oc0, oc1 = mae(gt, warp, cr), mae(gt, sib, cr)
        print(f"{frame:6s} {int(extra.sum()):9d} {int(better.sum()):7d} {int(worse.sum()):7d} "
              f"{m0:9.3f} {m1:8.3f} {m1 - m0:+7.3f} | {int(cr.sum()):9d} {oc0:9.3f} "
              f"{oc1:8.3f}")
        rows.append((int(extra.sum()), int(better.sum()), int(worse.sum()), m0, m1, oc0, oc1))
    if rows:
        m = lambda i: float(np.nanmean([r[i] for r in rows]))
        print(f"\nMEAN  extras {m(0):.0f} px: better {m(1):.0f}, worse {m(2):.0f} "
              f"({100 * m(1) / max(m(0), 1):.0f}% better)")
        print(f"      MAE on those px: warp {m(3):.3f} -> sibling {m(4):.3f} "
              f"({m(4) - m(3):+.3f}; negative = sibling better)")
        print(f"      我们的裂纹像素 MAE: warp {m(5):.3f} -> sibling {m(6):.3f} "
              f"({m(6) - m(5):+.3f})")
        print(f"\n  判断：{'值得移植（这些像素确实变好了）' if m(4) < m(3) - 0.05 else '不值得移植（改写没有变好或更差）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
