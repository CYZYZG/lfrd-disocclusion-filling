"""Sanity check on the head-to-head: does the sibling also rewrite OOFA / warp-valid pixels?

tools/head_to_head.py passes the sibling a hole mask that EXCLUDES OOFA, but if the sibling's
own pipeline still fills them on its way, the +1.9 dB comparison would be partly an artefact.
This checks where the sibling's output differs from the input warp.

    python tools/h2h_scope_check.py --run ba54_seq --src_cam 5 --dst_cam 4 --frames 0,1,2 `
        --tag ba54
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
    print(f"{'frame':6s} {'disocc+crack':>12s} {'changed there':>13s} | "
          f"{'OOFA px':>8s} {'changed in OOFA':>15s} {'OOFA PSNR sib':>13s} | "
          f"{'valid chg':>9s}")
    rows = []
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = os.path.join(io_utils.run_dir(a.run, create=False),
                         f"cam{a.src_cam}-cam{a.dst_cam}-{frame}")
        w = os.path.join(d, "20_warp")
        fp = os.path.join(h2h, f"{frame}_B_noOOFA", "03_filled.png")
        if not os.path.isfile(fp):
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        warp = io_utils.imread(os.path.join(w, "warped_color.png"))
        sib = io_utils.imread(fp)
        hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
        dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
        region = dis | cr
        chg = (sib.astype(np.int16) != warp.astype(np.int16)).any(2)
        chg_r = chg & region
        chg_o = chg & oofa
        chg_v = chg & ~hole
        print(f"{frame:6s} {int(region.sum()):12d} {int(chg_r.sum()):13d} | "
              f"{int(oofa.sum()):8d} {int(chg_o.sum()):15d} "
              f"{lpsnr(gt, sib, oofa):13.2f} | {int(chg_v.sum()):9d}")
        rows.append((int(chg_r.sum()), int(region.sum()), int(chg_o.sum()),
                     int(oofa.sum()), int(chg_v.sum())))
    if rows:
        m = lambda i: float(np.mean([r[i] for r in rows]))
        print(f"\nMEAN  在我们要评的区域内改写了 {m(0):.0f}/{m(1):.0f} px "
              f"({100 * m(0) / m(1):.1f}%)")
        print(f"      OOFA 内改写 {m(2):.0f}/{m(3):.0f} px "
              f"({100 * m(2) / max(m(3), 1):.1f}%)   <- 应为 0，否则比较不公平")
        print(f"      warp 有效像素内改写 {m(4):.0f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
