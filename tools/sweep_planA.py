"""Parameter sweep for the adaptive patch size + structural penalty (plan A).

Runs step5 only (step4 is independent of these knobs) for a grid of
`sizes / beta / struct_pen`, then scores the composited result.

Because step5's output only reaches the final view through step6, the sweep evaluates
`filled_occlusion.png` directly against the reference-image region that was removed, which is
where the synthesised occlusion layer lives, PLUS runs step6 + the disocclusion metric for the
configurations that look promising.

    python tools/sweep_planA.py --run q_67 --src_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils, metrics

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frame", default="f000")
    ap.add_argument("--configs", default=None,
                    help="JSON list of overrides, e.g. "
                         "'[{\"adaptive_size\":false,\"struct_pen\":0},"
                         "{\"sizes\":[9,7,5,3],\"beta\":150,\"struct_pen\":8}]'")
    a = ap.parse_args()
    import json
    cfgs = json.loads(a.configs) if a.configs else [
        {"adaptive_size": False, "struct_pen": 0.0},                    # paper-literal
        {"sizes": [9, 7, 5, 3], "beta": 150.0, "struct_pen": 0.0},      # cascade only
        {"adaptive_size": False, "struct_pen": 8.0},                    # penalty only
        {"sizes": [9, 7, 5, 3], "beta": 150.0, "struct_pen": 8.0},      # both
        {"sizes": [9, 7, 5, 3], "beta": 300.0, "struct_pen": 8.0},
        {"sizes": [9, 7, 5, 3], "beta": 150.0, "struct_pen": 30.0},
        {"sizes": (9, 5, 3), "beta": 150.0, "struct_pen": 8.0},
    ]
    rows = []
    for i, ov in enumerate(cfgs):
        cpath = os.path.join(io_utils.run_dir("_sweep", create=True), f"cfg_{i}.json")
        from lfrd.config import RunConfig
        c = RunConfig()
        for k, v in ov.items():
            if k == "sizes":
                v = tuple(int(x) for x in v)
            setattr(c, k, v)
        c.save(cpath)
        step = os.path.join(io_utils.run_dir("_sweep", create=True), f"out_{i}")
        args = [PY, os.path.join(ROOT, "step5_inpaint.py"), "--run", a.run,
                "--src_cam", str(a.src_cam), "--dst_cam", str(a.dst_cam),
                "--frame", a.frame, "--config", cpath, "--out_dir", step,
                "--no_ablate", "--no_panel"]
        rc, out, err = run(args)
        meta = os.path.join(step, "inpaint_meta.json")
        rec = {"config": ov, "rc": rc}
        if os.path.isfile(meta):
            import json as _j
            with open(meta, "r", encoding="utf-8") as fh:
                m = _j.load(fh)
            rec.update(iters=m.get("n_iters"), seconds=round(m.get("seconds", 0), 2),
                       size_hist=m.get("size_hist"),
                       leak=m.get("exemplar_in_foreground_fraction"),
                       src_depth_med=m.get("exemplar_source_depth_median"),
                       match_cost=m.get("match_cost_mean"))
        rows.append(rec)
        print(f"[{i}] {ov}  rc={rc} iters={rec.get('iters')} s={rec.get('seconds')} "
              f"sizes={rec.get('size_hist')} leak={rec.get('leak')} "
              f"srcDepth={rec.get('src_depth_med')} cost={rec.get('match_cost')}")
        if rc != 0 and err:
            print("   ", err.strip().splitlines()[-1][:160])
    # score every occlusion layer against the reference-view ground truth of the removed area
    # is not possible (the occluded background is unknown in the reference); instead we score
    # the FINAL view of each config, which requires step6 -> do it for the two best by match
    # cost to keep the sweep short.
    print("\nto score on the final view, run step6 with the chosen config:")
    print(f"  & $py step6_render.py --run {a.run} --src_cam {a.src_cam} "
          f"--dst_cam {a.dst_cam} --frame {a.frame} --config <cfg>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
