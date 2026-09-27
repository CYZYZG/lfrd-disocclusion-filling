"""Clean A/B: paper-literal vs adaptive cascade, measured directly from the artefacts.

    python tools/ab_cascade.py --runs q_67:6:7 q_54:5:4 --frames 0,1,2

For each frame the two configurations are run back to back (step5 + step6) and the final
view is scored with lfrd.metrics on the disocclusion mask.  No log parsing.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics
from lfrd.config import RunConfig

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "_ab")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--fill_oofa", action="store_true", default=False)
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    os.makedirs(OUT, exist_ok=True)
    configs = {
        "paper-literal": dict(sizes=None, beta=150.0, struct_pen=0.0),
        "cascade": dict(sizes=(9, 7, 5, 3), beta=150.0, struct_pen=0.0),
    }
    paths = {}
    for name, ov in configs.items():
        c = RunConfig()
        for k, v in ov.items():
            setattr(c, k, v)
        c.fill_oofa = bool(a.fill_oofa)
        p = os.path.join(OUT, f"{name}.json")
        c.save(p)
        paths[name] = p
    table = {}
    for name, cfgp in paths.items():
        for spec in a.runs:
            base_run, s, d = spec.split(":")
            src, dst = int(s), int(d)
            for fi in frames:
                frame = io_utils.frame_name(fi)
                sub = run([PY, os.path.join(ROOT, "step5_inpaint.py"), "--run", base_run,
                           "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame,
                           "--config", cfgp, "--no_ablate", "--no_panel"])
                if sub.returncode != 0:
                    print(f"[warn] step5 {name} {frame}: "
                          f"{(sub.stderr or sub.stdout).strip().splitlines()[-1][:140]}")
                sub = run([PY, os.path.join(ROOT, "step6_render.py"), "--run", base_run,
                           "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame])
                src_run = os.path.join(io_utils.run_dir(base_run, create=False),
                                       f"cam{src}-cam{dst}-{frame}")
                wd = os.path.join(src_run, "20_warp")
                fp = os.path.join(io_utils.run_dir(base_run, "final", create=False),
                                  "final.png")
                if not os.path.isfile(fp):
                    continue
                gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
                fin = io_utils.imread(fp)
                warp = io_utils.imread(os.path.join(wd, "warped_color.png"))
                dis = io_utils.imread(os.path.join(wd, "hole_disocc.png"), gray=True) > 0
                crack = io_utils.imread(os.path.join(wd, "hole_crack.png"), gray=True) > 0
                oofa = io_utils.imread(os.path.join(wd, "hole_oofa.png"), gray=True) > 0
                reg = dis | crack
                # luma too, so the numbers are comparable with the sibling project's metric
                g = lambda im: (0.299 * np.asarray(im, np.float64)[..., 0]
                                + 0.587 * np.asarray(im, np.float64)[..., 1]
                                + 0.114 * np.asarray(im, np.float64)[..., 2])
                d2 = (g(fin) - g(gt)) ** 2
                luma_dis = 10 * np.log10(255.0 ** 2 / d2[dis].mean())
                table[(name, f"{src}->{dst}", frame)] = dict(
                    dis=metrics.psnr(fin, gt, mask=dis),
                    filled=metrics.psnr(fin, gt, mask=reg),
                    whole=metrics.psnr(fin, gt),
                    luma_dis=float(luma_dis),
                    warp_dis=metrics.psnr(warp, gt, mask=dis))
                keep = os.path.join(OUT, name, f"{src}-{dst}-{frame}")
                os.makedirs(keep, exist_ok=True)
                io_utils.imwrite(os.path.join(keep, "final.png"), fin)
                print(f"{name:14s} {src}->{dst} {frame}: disocc {table[(name, f'{src}->{dst}', frame)]['dis']:6.2f}  "
                      f"filled {table[(name, f'{src}->{dst}', frame)]['filled']:6.2f}  "
                      f"luma_disocc {luma_dis:6.2f}")
    # summary
    keys = sorted(table)
    names = list(configs)
    means = {n: float(np.mean([table[k]["dis"] for k in keys if k[0] == n]))
             for n in names if any(k[0] == n for k in keys)}
    lmeans = {n: float(np.mean([table[k]["luma_dis"] for k in keys if k[0] == n]))
              for n in names if any(k[0] == n for k in keys)}
    lines = ["# A/B: paper-literal vs adaptive patch-size cascade", "",
             "Both rows: same warp, same step4, only the step-5 patch-size policy differs.",
             "", "| config | disocclusion PSNR (RGB) | disocclusion PSNR (luma) | vs literal |",
             "| --- | --- | --- | --- |"]
    base = means.get("paper-literal", float("nan"))
    for n in names:
        if n not in means:
            continue
        lines.append(f"| {n} | {means[n]:.3f} | {lmeans[n]:.3f} | {means[n] - base:+.3f} |")
    # per pair
    pairs = sorted({(k[1], k[2]) for k in keys})
    lines += ["", "## Per frame", "", "| pair | frame | literal | cascade | delta |",
              "| --- | --- | --- | --- | --- |"]
    for p, f in pairs:
        if ("paper-literal", p, f) in table and ("cascade", p, f) in table:
            x = table[("paper-literal", p, f)]["dis"]
            y = table[("cascade", p, f)]["dis"]
            lines.append(f"| {p} | {f} | {x:.2f} | {y:.2f} | {y - x:+.2f} |")
    txt = "\n".join(lines) + "\n"
    io_utils.write_text(os.path.join(OUT, "report.md"), txt)
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
