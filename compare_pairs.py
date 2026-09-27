"""Cross-pair comparison: one figure per camera pair with GT / warp / baseline / ours.

    python compare_pairs.py --runs ba54_seq:5:4 t_cam67:6:7 t_cam30:3:0 --frames 0

Writes output/_compare/<run>_cam<src>-cam<dst>_f###.png and a summary table plus
output/_compare/summary.md.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics, viz


def frame_dirs(run, src, dst):
    root = io_utils.run_dir(run, create=False)
    if not os.path.isdir(root):
        return []
    pre = f"cam{src}-cam{dst}-f"
    return sorted(d for d in os.listdir(root)
                  if d.startswith(pre) and os.path.isdir(os.path.join(root, d)))


def load_arm(run, fd, name):
    p = os.path.join(io_utils.run_dir(run, create=False), fd, "60_final", "final.png")
    if name == "ours":
        return io_utils.imread(p) if os.path.isfile(p) else None
    if name == "warp":
        q = os.path.join(io_utils.run_dir(run, create=False), fd, "20_warp",
                         "warped_color.png")
        return io_utils.imread(q) if os.path.isfile(q) else None
    q = os.path.join(io_utils.run_dir(run, create=False), "ablation", fd, f"{name}.png")
    return io_utils.imread(q) if os.path.isfile(q) else None


def masked_psnr(img, gt, mask):
    a = np.array(img, np.float64)
    b = np.array(gt, np.float64)
    d = ((a - b) ** 2).mean(axis=2)
    if not np.asarray(mask, bool).any():
        return float("nan")
    return 10.0 * np.log10(255.0 ** 2 / float(d[mask].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run:src:dst  e.g. ba54_seq:5:4")
    ap.add_argument("--frames", default="0",
                    help="frame indices to include in the figure (comma list / 'all')")
    ap.add_argument("--out", default="_compare")
    a = ap.parse_args()
    outdir = io_utils.run_dir(a.out, create=True)
    rows = []
    for spec in a.runs:
        run, s, d = spec.split(":")
        src, dst = int(s), int(d)
        fds = frame_dirs(run, src, dst)
        for fd in fds:
            fi = int(fd.split("-f")[1])
            frame = fd.split("-")[-1]
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
            basep = os.path.join(io_utils.run_dir(run, create=False), fd, "20_warp")
            need = (os.path.join(basep, "hole_all.png"),
                    os.path.join(basep, "hole_oofa.png"),
                    os.path.join(basep, "hole_disocc.png"))
            if not all(os.path.isfile(q) for q in need):
                continue          # frame dir left incomplete by a failed stage
            hole = io_utils.imread(need[0], gray=True) > 0
            oofa = io_utils.imread(need[1], gray=True) > 0
            disocc = io_utils.imread(need[2], gray=True) > 0
            filled = hole & ~oofa
            arms = {}
            for name in ("warp", "direct_inpaint", "ours"):
                img = load_arm(run, fd, name)
                if img is not None:
                    arms[name] = img
            if "ours" not in arms:
                continue
            rec = {"pair": f"{src}->{dst}", "run": run, "frame": frame,
                   "hole_pct": 100.0 * float(hole.mean()),
                   "n_disocc": int(disocc.sum()), "n_oofa": int(oofa.sum()),
                   "ours_whole": metrics.psnr(arms["ours"], gt),
                   "ours_filled": masked_psnr(arms["ours"], gt, filled),
                   "ours_disocc": masked_psnr(arms["ours"], gt, disocc)}
            for name in ("warp", "direct_inpaint"):
                if name in arms:
                    rec[f"{name}_whole"] = metrics.psnr(arms[name], gt)
                    rec[f"{name}_filled"] = masked_psnr(arms[name], gt, filled)
            rows.append(rec)
            # per-frame figure for the frames the user asked for
            want = (a.frames == "all" or fi in
                    [int(x) for x in a.frames.split(",") if x.strip()])
            if want:
                tiles = [(gt, f"GT cam{dst}"),
                         (arms["warp"], f"plain warp ({100 * hole.mean():.1f}% holes)")]
                if "direct_inpaint" in arms:
                    tiles.append((arms["direct_inpaint"],
                                  "direct inpaint [17] %.2f dB" % rec["direct_inpaint_filled"]))
                tiles.append((arms["ours"], "proposed %.2f dB" % rec["ours_filled"]))
                tiles += [(viz.diff_map(arms["ours"], gt), "|proposed-GT| x3")]
                if "direct_inpaint" in arms:
                    tiles.insert(4, (viz.diff_map(arms["direct_inpaint"], gt),
                                     "|baseline-GT| x3"))
                tiles.append((viz.mask_overlay(arms["ours"], filled, (0, 255, 0)),
                              "filled regions"))
                viz.grid(tiles, cols=4, height=300,
                         path=os.path.join(outdir, f"{run}_{fd}.png"),
                         title=f"{run} cam{src}->cam{dst} {frame}  hole "
                               f"{100 * hole.mean():.2f}%  filled-region PSNR: "
                               f"warp {rec.get('warp_filled', float('nan')):.2f} -> "
                               f"proposed {rec['ours_filled']:.2f} dB")
    # summary
    lines = ["# Cross-pair comparison", "",
             "`filled` = disocclusion + cracks (the region the method owns); PSNR averages "
             "the MSE over the region's own pixels.", "",
             "| pair | run | frame | hole % | disocc px | whole warp | whole ours | "
             "filled warp | filled direct | **filled ours** | ours-direct |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda x: (x["pair"], x["frame"])):
        lines.append(
            f"| {r['pair']} | {r['run']} | {r['frame']} | {r['hole_pct']:.2f} | "
            f"{r['n_disocc']} | {r.get('warp_whole', float('nan')):.2f} | "
            f"{r['ours_whole']:.2f} | {r.get('warp_filled', float('nan')):.2f} | "
            f"{r.get('direct_inpaint_filled', float('nan')):.2f} | "
            f"**{r['ours_filled']:.2f}** | "
            f"{r['ours_filled'] - r.get('direct_inpaint_filled', float('nan')):+.2f} |")
    io_utils.write_text(os.path.join(outdir, "summary.md"), "\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwritten:", os.path.join(outdir, "summary.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
