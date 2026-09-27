"""Aggregate per-frame ablation results into one comparison table.

    python summarize.py --run ba54_seq --pair 5:4

Reads output/<run>/ablation/cam<src>-cam<dst>-f###/metrics.json and
output/<run>/eval/metrics.json and writes output/<run>/eval/summary.md (+ console).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ba54_seq")
    ap.add_argument("--pair", default="5:4")
    a = ap.parse_args()
    src, dst = [int(x) for x in a.pair.split(":")]
    ab = os.path.join(io_utils.run_dir(a.run), "ablation")
    frames = sorted(glob.glob(os.path.join(ab, f"cam{src}-cam{dst}-f*")))

    arms = ["warp_raw", "warp_ghost", "direct_inpaint", "ours"]
    acc = {k: {"psnr_whole": [], "psnr_valid": [], "psnr_filled": [], "psnr_disocc": [],
               "ssim_filled": []} for k in arms}
    ghost = {"raw": [], "ghost": []}
    for fd in frames:
        p = os.path.join(fd, "metrics.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        for row in d["rows"]:
            k = row["arm"]
            if k in acc:
                for m in acc[k]:
                    v = row.get(m)
                    if v is not None and np.isfinite(v):
                        acc[k][m].append(v)
            elif k == "ghost_removal":
                if np.isfinite(row.get("psnr_common_raw", np.nan)):
                    ghost["raw"].append(row["psnr_common_raw"])
                if np.isfinite(row.get("psnr_common_ghost", np.nan)):
                    ghost["ghost"].append(row["psnr_common_ghost"])

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    lines = [f"# Summary — run `{a.run}`, pair {src}:{dst}, {len(frames)} frames", "",
             "Metrics against the real capture of cam%d. `filled` = disocclusion + crack "
             "pixels (the region this method must synthesise); `valid` = pixels the plain "
             "warp already had (regression check); `disocc` = disocclusions only." % dst, "",
             "## Ablation arms (mean over frames)", "",
             "| arm | whole PSNR | valid PSNR | filled PSNR | filled SSIM | disocc PSNR |",
             "| --- | --- | --- | --- | --- | --- |"]
    label = {"warp_raw": "plain warp (raw depth)",
             "warp_ghost": "plain warp (+ghost removal)",
             "direct_inpaint": "direct virtual-view inpainting [17]",
             "ours": "**proposed (full pipeline)**"}
    for k in arms:
        c = acc[k]
        if not c["psnr_filled"]:
            continue
        lines.append(f"| {label[k]} | {mean(c['psnr_whole']):.3f} | "
                     f"{mean(c['psnr_valid']):.3f} | **{mean(c['psnr_filled']):.3f}** | "
                     f"{mean(c['ssim_filled']):.4f} | {mean(c['psnr_disocc']):.3f} |")
    lines += ["", "## Component gains", ""]
    if acc["warp_raw"]["psnr_filled"] and acc["warp_ghost"]["psnr_filled"]:
        lines.append(f"- ghost removal (raw -> preprocessed depth), on the filled region: "
                     f"{mean(acc['warp_raw']['psnr_filled']):.3f} -> "
                     f"{mean(acc['warp_ghost']['psnr_filled']):.3f} dB "
                     f"({mean(acc['warp_ghost']['psnr_filled']) - mean(acc['warp_raw']['psnr_filled']):+.3f})")
    if ghost["raw"] and ghost["ghost"]:
        lines.append(f"- ghost removal measured on the pixels BOTH arms warp "
                     f"(common valid set, continuous, where a ghost is visible): "
                     f"{mean(ghost['raw']):.3f} -> {mean(ghost['ghost']):.3f} dB "
                     f"({mean(ghost['ghost']) - mean(ghost['raw']):+.3f})")
    if acc["ours"]["psnr_filled"] and acc["direct_inpaint"]["psnr_filled"]:
        lines += ["",
                  "## Headline: reference-image prediction vs direct inpainting", "",
                  f"- direct virtual-view inpainting (Criminisi-style baseline [17]): "
                  f"**{mean(acc['direct_inpaint']['psnr_filled']):.3f} dB** / SSIM "
                  f"{mean(acc['direct_inpaint']['ssim_filled']):.4f}",
                  f"- proposed (local foreground removal + reference-image prediction): "
                  f"**{mean(acc['ours']['psnr_filled']):.3f} dB** / SSIM "
                  f"{mean(acc['ours']['ssim_filled']):.4f}",
                  f"- **advantage: "
                  f"{mean(acc['ours']['psnr_filled']) - mean(acc['direct_inpaint']['psnr_filled']):+.3f} dB** "
                  f"/ {mean(acc['ours']['ssim_filled']) - mean(acc['direct_inpaint']['ssim_filled']):+.4f} SSIM",
                  ""]
    # per-frame table
    rows = []
    for fd in frames:
        p = os.path.join(fd, "metrics.json")
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        get = {r["arm"]: r for r in d["rows"]}
        if "ours" not in get or "warp_ghost" not in get:
            continue
        rows.append((os.path.basename(fd), get))
    if rows:
        lines += ["## Per frame", "",
                  "| frame | filled PSNR warp | filled PSNR direct | filled PSNR ours | "
                  "ours - direct |", "| --- | --- | --- | --- | --- |"]
        for name, g in rows:
            o = g["ours"]["psnr_filled"]
            w = g["warp_ghost"]["psnr_filled"]
            b = g["direct_inpaint"]["psnr_filled"] if "direct_inpaint" in g else float("nan")
            lines.append(f"| {name} | {w:.3f} | {b:.3f} | **{o:.3f}** | {o - b:+.3f} |")
        lines.append("")
    out = os.path.join(io_utils.run_dir(a.run, "eval"), "summary.md")
    io_utils.write_text(out, "\n".join(lines) + "\n")
    for l in lines:
        print(l)
    print("\nwritten:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
