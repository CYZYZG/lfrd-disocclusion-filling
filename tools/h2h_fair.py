"""Fair-mask head-to-head: score only the pixels that are ACTUALLY EMPTY in the warp.

The sibling project fills more than the empty pixels: it detects ~9.5k px of "cracks /
under-covered" pixels (a lam=5 depth-edge criterion) and re-inpaints them too, whereas this
paper's step 6 asserts the final view is pixel-identical to the warp wherever the warp was
valid.  Scoring on "disocclusion + cracks" therefore penalises our method for not touching
pixels it deliberately preserves.

This script re-scores both results on
    EMPTY  = pixels where no source sample landed at all (our hole_all mask)
only, and also reports the sibling's own hole accounting so the two conventions are visible.

    python tools/h2h_fair.py --run q_67 --src_cam 6 --dst_cam 7 --frames 0,1,2,3,4
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils


def gray(a):
    a = np.asarray(a, np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def psnr(gt, img, mask):
    d = (gray(gt) - gray(img)) ** 2
    m = np.asarray(mask, bool)
    if m.ndim == 3:
        m = m[..., 0]
    d = d[m]
    return float(10 * np.log10(255.0 ** 2 / d.mean())) if d.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:src:dst")
    ap.add_argument("--frames", default="0,1,2,3,4")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    rows = []
    print(f"{'pair':9s} {'frame':6s} {'empty px':>9s} {'warp':>7s} {'ours':>7s} "
          f"{'sibling':>8s} {'diff':>7s} | {'sib unsup px':>12s} {'sib holes_after':>15s}")
    for spec in a.runs:
        run, s, d = spec.split(":")
        src, dst = int(s), int(d)
        for fi in frames:
            frame = io_utils.frame_name(fi)
            fd = os.path.join(io_utils.run_dir(run, create=False),
                              f"cam{src}-cam{dst}-{frame}")
            w = os.path.join(fd, "20_warp")
            if not os.path.isdir(w):
                continue
            h2h = os.path.join(io_utils.run_dir("_h2h", create=False),
                               f"{run}_cam{src}{dst}", f"{frame}_B_noOOFA")
            fp = os.path.join(h2h, "03_filled.png")
            if not os.path.isfile(fp):
                continue
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
            warp = io_utils.imread(os.path.join(w, "warped_color.png"))
            ours = io_utils.imread(os.path.join(fd, "60_final", "final.png"))
            sib = io_utils.imread(fp)
            empty = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
            pw, po, psh = (psnr(gt, warp, empty), psnr(gt, ours, empty),
                           psnr(gt, sib, empty))
            st = {}
            sp = os.path.join(h2h, "stats.json")
            if os.path.isfile(sp):
                with open(sp, encoding="utf-8") as fh:
                    st = json.load(fh)
            rows.append(dict(pair=f"{src}->{dst}", frame=frame,
                             empty_px=int(empty.sum()), warp=pw, ours=po, sibling=psh,
                             sib_filled_px=st.get("filled_px_actual"),
                             sib_holes_before=st.get("holes_before"),
                             sib_holes_after=st.get("holes_after"),
                             sib_crack_px=st.get("crack_px")))
            print(f"{src:>3d}->{dst:<3d} {frame:6s} {int(empty.sum()):9d} {pw:7.2f} "
                  f"{po:7.2f} {psh:8.2f} {po - psh:+7.2f} | "
                  f"{str(st.get('filled_px_actual', '?')):>12s} "
                  f"{str(st.get('holes_after', '?')):>15s}")
    if rows:
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"\nMEAN on the truly-empty pixels:  warp {m('warp'):.2f}  ours {m('ours'):.2f}  "
              f"sibling {m('sibling'):.2f}   ->  ours-sibling {m('ours') - m('sibling'):+.2f} dB")
        lines = ["# Fair-mask head-to-head", "",
                 "Scored only on pixels that are genuinely empty in the warp "
                 "(`hole_all`), so neither method is rewarded or punished for touching "
                 "warp-valid pixels.", "",
                 "| pair | frame | empty px | warp | ours | sibling | ours-sibling | "
                 "sibling filled px | sibling holes after |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            lines.append(f"| {r['pair']} | {r['frame']} | {r['empty_px']} | "
                         f"{r['warp']:.2f} | **{r['ours']:.2f}** | {r['sibling']:.2f} | "
                         f"{r['ours'] - r['sibling']:+.2f} | {r['sib_filled_px']} | "
                         f"{r['sib_holes_after']} |")
        lines += ["", f"**Mean: warp {m('warp'):.2f} dB, ours {m('ours'):.2f} dB, "
                      f"sibling {m('sibling'):.2f} dB "
                      f"(ours - sibling = {m('ours') - m('sibling'):+.2f} dB).**", ""]
        out = os.path.join(io_utils.run_dir("_h2h", create=True), "fair_summary.md")
        io_utils.write_text(out, "\n".join(lines) + "\n")
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
