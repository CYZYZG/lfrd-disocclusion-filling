"""A/B the query-template foreground filter (plan item 3).

    python tools/ab_template_fg.py --src_cam 5 --dst_cam 4 --frames 0,1,2

Runs step5+step6 for both settings on the same warp and scores the disocclusion region.
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
OUT = os.path.join(ROOT, "output", "_tpl")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--base_run", default=None,
                    help="run whose per-frame warp is reused (default: derived)")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    os.makedirs(OUT, exist_ok=True)
    variants = {"off": dict(template_exclude_fg=False),
                "on": dict(template_exclude_fg=True)}
    # the reference warp for this pair must already exist in a run dir
    base = a.base_run or f"ba54_seq__cam{a.src_cam}-{a.dst_cam}-f000"
    results = {}
    for name, ov in variants.items():
        cfgp = os.path.join(OUT, f"{name}.json")
        c = RunConfig()
        for k, v in ov.items():
            setattr(c, k, v)
        c.save(cfgp)
        for fi in frames:
            frame = io_utils.frame_name(fi)
            src_run = a.base_run or f"ba54_seq__cam{a.src_cam}-{a.dst_cam}-{frame}"
            wd = os.path.join(io_utils.run_dir(src_run, create=False), "20_warp")
            rd = os.path.join(io_utils.run_dir(src_run, create=False), "40_removal")
            if not (os.path.isdir(wd) and os.path.isdir(rd)):
                print(f"[skip] {frame}: {src_run} lacks warp/removal")
                continue
            work = os.path.join(OUT, f"run_{name}_{frame}")
            os.makedirs(work, exist_ok=True)
            # point the stage scripts at this frame's artefacts by copying them into a run dir
            import shutil
            for stage, key in (("20_warp", "warp"), ("40_removal", "removal"),
                               ("50_fill", "fill")):
                s = os.path.join(io_utils.run_dir(src_run, create=False), stage)
                if os.path.isdir(s):
                    d = os.path.join(io_utils.run_dir(work, create=False), stage)
                    os.makedirs(d, exist_ok=True)
                    for n in os.listdir(s):
                        p = os.path.join(s, n)
                        if os.path.isfile(p) and os.path.getsize(p) < 24 * 1024 * 1024:
                            shutil.copy2(p, os.path.join(d, n))
            for script, extra in (("step5_inpaint.py", ["--no_ablate", "--no_panel"]),
                                  ("step6_render.py", [])):
                r = run([PY, os.path.join(ROOT, script), "--run", work,
                         "--src_cam", str(a.src_cam), "--dst_cam", str(a.dst_cam),
                         "--frame", frame, "--config", cfgp] + extra)
                if r.returncode != 0:
                    print(f"  [warn] {name} {script} {frame}: "
                          f"{(r.stderr or r.stdout).strip().splitlines()[-1][:140]}")
            gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam,
                                    frame)["color"]
            fp = os.path.join(io_utils.run_dir(work, "final", create=False), "final.png")
            if not os.path.isfile(fp):
                continue
            fin = io_utils.imread(fp)
            dis = io_utils.imread(os.path.join(wd, "hole_disocc.png"), gray=True) > 0
            cr = io_utils.imread(os.path.join(wd, "hole_crack.png"), gray=True) > 0
            results[(name, frame)] = (metrics.psnr(fin, gt, mask=dis),
                                      metrics.psnr(fin, gt, mask=dis | cr),
                                      metrics.psnr(fin, gt))
            print(f"{name:4s} {frame}: disocc {results[(name, frame)][0]:6.2f}  "
                  f"filled {results[(name, frame)][1]:6.2f}  whole "
                  f"{results[(name, frame)][2]:6.2f}")
    fs = sorted({f for _, f in results})
    print()
    for name in variants:
        v = [results[(name, f)][0] for f in fs if (name, f) in results]
        if v:
            print(f"{name:4s} mean disocc PSNR {np.mean(v):.3f}   (n={len(v)})")
    if "off" in variants and "on" in variants:
        vo = [results[("off", f)][0] for f in fs if ("off", f) in results]
        vn = [results[("on", f)][0] for f in fs if ("on", f) in results]
        if vo and vn:
            print(f"delta = {np.mean(vn) - np.mean(vo):+.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
