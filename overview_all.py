"""One big figure: the same frame index across several camera pairs, GT / warp / proposed.

    python overview_all.py --runs q_54:5:4 q_45:4:5 q_56:5:6 q_67:6:7 q_30:3:0 --frame f000

Rows: one camera pair each.  Columns: ground truth, region the proposed method filled
(green), plain warp, proposed result, |proposed-GT| amplified.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:src:dst")
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--out", default="_compare/overview.png")
    a = ap.parse_args()
    rows = []
    for spec in a.runs:
        run, s, d = spec.split(":")
        src, dst = int(s), int(d)
        fd = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{a.frame}")
        fin = os.path.join(fd, "60_final", "final.png")
        wrp = os.path.join(fd, "20_warp")
        if not (os.path.isfile(fin) and os.path.isfile(os.path.join(wrp, "hole_all.png"))):
            print(f"[skip] {spec} {a.frame}: incomplete")
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, a.frame)["color"]
        ours = io_utils.imread(fin)
        warp = io_utils.imread(os.path.join(wrp, "warped_color.png"))
        hole = io_utils.imread(os.path.join(wrp, "hole_all.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(wrp, "hole_oofa.png"), gray=True) > 0
        filled = hole & ~oofa
        pf = metrics.psnr(ours, gt, mask=filled)
        pw = metrics.psnr(warp, gt, mask=filled)
        tiles = [
            (gt, f"cam{src}->cam{dst}  GT"),
            (viz.mask_overlay(ours, filled, (0, 255, 0)), f"filled region {100 * hole.mean():.1f}% holes"),
            (warp, f"plain warp  {pw:.2f} dB (filled)"),
            (ours, f"proposed  {pf:.2f} dB  ({pf - pw:+.2f})"),
            (viz.diff_map(ours, gt), "|proposed-GT| x3"),
        ]
        rows.append(tiles)
        print(f"cam{src}->cam{dst}: hole {100 * hole.mean():.2f}%  filled PSNR "
              f"{pw:.2f} -> {pf:.2f} dB ({pf - pw:+.2f})")
    if not rows:
        print("nothing to draw")
        return 1
    # build a grid: rows = pairs, cols = tiles
    merged = []
    ncol = max(len(r) for r in rows)
    for tiles in rows:
        merged.extend(tiles)
    out = os.path.join(io_utils.run_dir(a.out.split("/")[0], create=True),
                       *a.out.split("/")[1:]) if "/" in a.out else a.out
    viz.grid(merged, cols=ncol, height=230, path=out,
             title=f"hole filling across camera pairs — {a.frame} "
                   f"(green = region the method had to synthesise)")
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
