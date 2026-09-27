"""Region-by-region breakdown: where exactly do the two projects differ?

For one camera pair/frame this prints, in the SIBLING project's metric formulation
(gray/luma PSNR, holes counted as black), the PSNR of every disjoint region:
    valid      -- pixels the warp produced
    crack      -- 1-2 px DIBR cracks
    disocc     -- disocclusions (what the method is supposed to fill)
    oofa       -- out of field area (unreachable by design, black for both sides)
    filled     -- disocclusion + crack (the region that gets synthesised)
plus a few synthetic reconstructions that answer specific questions:
    ours                = our final view
    warp                = our plain warp (holes black)
    ours[oofa=GT]       = our final view with the OOFA taken from the ground truth
    ours[holes=GT]      = our final view with every hole taken from the ground truth
    warp[holes=GT]      = the plain warp with every hole taken from the ground truth
The last three show how much of the deficit is the unfilled OOFA and how much is the
quality of the synthesised content itself.

    python tools/region_breakdown.py --run q_67 --src_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(img):
    a = np.asarray(img, np.float64)
    if a.ndim == 3:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def psnr(gt, img, mask=None):
    d = (gray(gt) - gray(img)) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        d = d[m]
        if d.size == 0:
            return float("nan")
        return float(10 * np.log10(255.0 ** 2 / d.mean()))
    return float(10 * np.log10(255.0 ** 2 / d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frame", default="f000")
    a = ap.parse_args()
    fd = os.path.join(io_utils.run_dir(a.run, create=False),
                      f"cam{a.src_cam}-cam{a.dst_cam}-{a.frame}")
    w = os.path.join(fd, "20_warp")
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, a.frame)["color"]
    ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
    warp = io_utils.imread(os.path.join(w, "warped_color.png"))
    hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
    crack = io_utils.imread(os.path.join(w, "hole_crack.png"), gray=True) > 0
    dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
    valid = ~hole
    filled = hole & ~oofa
    N = float(hole.size)

    def subst(img, mask):
        o = np.array(img)
        o[mask] = gt[mask]
        return o

    print(f"run={a.run} cam{a.src_cam}->cam{a.dst_cam} {a.frame}   ({int(N)} px total)")
    print(f"  valid {100 * valid.mean():6.2f}%   crack {100 * crack.mean():5.2f}%   "
          f"disocc {100 * dis.mean():5.2f}%   oofa {100 * oofa.mean():5.2f}%   "
          f"hole {100 * hole.sum() / N:5.2f}%")
    print()
    rows = [
        ("ALL (holes black)      ", ours),
        ("ALL, holes <- GT       ", subst(ours, hole)),
        ("ALL, oofa only <- GT   ", subst(ours, oofa)),
        ("warp (holes black)     ", warp),
        ("warp, holes <- GT      ", subst(warp, hole)),
    ]
    print(f"{'view':26s} {'whole':>8s} {'valid':>8s} {'filled':>8s} {'disocc':>8s} "
          f"{'crack':>8s} {'oofa':>8s}")
    for name, img in rows:
        print(f"{name:26s} {psnr(gt, img):8.2f} {psnr(gt, img, valid):8.2f} "
              f"{psnr(gt, img, filled):8.2f} {psnr(gt, img, dis):8.2f} "
              f"{psnr(gt, img, crack):8.2f} {psnr(gt, img, oofa):8.2f}")
    print()
    # how much of the whole-frame gap is OOFA?
    w_black = psnr(gt, ours)
    w_oofa = psnr(gt, subst(ours, oofa))
    w_all = psnr(gt, subst(ours, hole))
    print(f"whole-frame PSNR: holes black {w_black:.2f}  |  OOFA from GT {w_oofa:.2f} "
          f"(+{w_oofa - w_black:.2f})  |  all holes from GT {w_all:.2f} "
          f"(+{w_all - w_black:.2f})")
    print(f"=> the OOFA alone accounts for {w_oofa - w_black:.2f} dB of the whole-frame "
          f"deficit; the synthesised content accounts for the rest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
