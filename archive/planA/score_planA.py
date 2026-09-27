"""Grid over (adaptive cascade, beta, structural penalty) scored on the final view.

    python tools/score_planA.py --runs q_67:6:7 q_54:5:4 --frames 0,1
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
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEARCH_ROOT = os.path.join(ROOT, "output", "_planA")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


def candidate_grid():
    cs = [("paper-literal", dict(adaptive_size=False, struct_pen=0.0))]
    for beta in (75.0, 150.0, 300.0):
        for pen in (0.0, 2.0, 8.0):
            cs.append((f"casc_b{int(beta)}_p{int(pen)}",
                       dict(adaptive_size=True, sizes=(9, 7, 5, 3), beta=beta,
                            struct_pen=pen)))
    for sizes in ((9, 5, 3), (11, 9, 7, 5, 3)):
        cs.append((f"casc_s{'_'.join(str(s) for s in sizes)}_p0",
                   dict(adaptive_size=True, sizes=sizes, beta=150.0, struct_pen=0.0)))
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--frames", default="0")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--oos", action="store_true", help="score occluded-region PSNR too")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    cands = candidate_grid()
    if a.candidates:
        want = set(a.candidates.split(","))
        cands = [c for c in cands if c[0] in want]
    os.makedirs(SEARCH_ROOT, exist_ok=True)
    table = {}
    for name, ov in cands:
        cfg_path = os.path.join(SEARCH_ROOT, f"{name}.json")
        c = RunConfig()
        for k, v in ov.items():
            setattr(c, k, v)
        c.fill_oofa = False
        c.save(cfg_path)
        for spec in a.runs:
            base_run, s, d = spec.split(":")
            src, dst = int(s), int(d)
            for fi in frames:
                frame = io_utils.frame_name(fi)
                args5 = [PY, os.path.join(ROOT, "step5_inpaint.py"), "--run", base_run,
                         "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame,
                         "--config", cfg_path, "--no_ablate", "--no_panel"]
                rc, out, err = run(args5)
                sizes_line = next((ln.strip() for ln in out.splitlines()
                                   if "adaptive patch size" in ln), "")
                args6 = [PY, os.path.join(ROOT, "step6_render.py"), "--run", base_run,
                         "--src_cam", str(src), "--dst_cam", str(dst), "--frame", frame]
                rc6, out6, err6 = run(args6)
                src_run = os.path.join(io_utils.run_dir(base_run, create=False),
                                       f"cam{src}-cam{dst}-{frame}")
                warp_dir = os.path.join(src_run, "20_warp")
                fin_p = os.path.join(io_utils.run_dir(base_run, "final", create=False),
                                     "final.png")
                if not os.path.isfile(fin_p):
                    continue
                gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
                fin = io_utils.imread(fin_p)
                warp = io_utils.imread(os.path.join(warp_dir, "warped_color.png"))
                dis = io_utils.imread(os.path.join(warp_dir, "hole_disocc.png"),
                                      gray=True) > 0
                crack = io_utils.imread(os.path.join(warp_dir, "hole_crack.png"),
                                        gray=True) > 0
                oofa = io_utils.imread(os.path.join(warp_dir, "hole_oofa.png"),
                                       gray=True) > 0
                rec = dict(dis=metrics.psnr(fin, gt, mask=dis),
                           filled=metrics.psnr(fin, gt, mask=dis | crack),
                           whole=metrics.psnr(fin, gt),
                           nonoofa=metrics.psnr(fin, gt, mask=~oofa),
                           sizes=sizes_line.split("used")[-1].strip(" :") if sizes_line else "")
                table[(name, f"{src}->{dst}", frame)] = rec
                keep = os.path.join(SEARCH_ROOT, name, f"{src}-{dst}-{frame}")
                os.makedirs(keep, exist_ok=True)
                io_utils.imwrite(os.path.join(keep, "final.png"), fin)
                print(f"{name:22s} {src}->{dst} {frame}: disocc {rec['dis']:6.2f}  "
                      f"filled {rec['filled']:6.2f}  whole {rec['whole']:6.2f}  "
                      f"size_hist {rec['sizes']}")
    names = [n for n, _ in cands]
    keys = sorted(table)
    means = {n: float(np.mean([table[k]["dis"] for k in keys if k[0] == n]))
             for n in names if any(k[0] == n for k in keys)}
    base = means.get("paper-literal", float("nan"))
    best = max(means.values()) if means else float("nan")
    lines = ["# plan A sweep — disocclusion PSNR on the FINAL view", "",
             "Every row is a full step5+step6 run on the same warp.", "",
             "| config | disocclusion PSNR | vs paper-literal | filled region | whole frame |",
             "| --- | --- | --- | --- | --- |"]
    for n in names:
        if n not in means:
            continue
        fv = float(np.mean([table[k]["filled"] for k in keys if k[0] == n]))
        wv = float(np.mean([table[k]["whole"] for k in keys if k[0] == n]))
        star = " **<-- best**" if means[n] == best else ""
        lines.append(f"| {n} | {means[n]:.3f} | {means[n] - base:+.3f} | {fv:.3f} | "
                     f"{wv:.3f} |{star}")
    txt = "\n".join(lines) + "\n"
    io_utils.write_text(os.path.join(SEARCH_ROOT, "sweep_report.md"), txt)
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
