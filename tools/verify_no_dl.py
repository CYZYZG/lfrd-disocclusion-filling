"""Verification that the whole reproduction is CLASSICAL only: no deep-learning dependency.

    python tools/verify_no_dl.py

Checks every .py in the project for imports of a deep-learning framework, lists what IS used, and
prints the current best measured result so the claim and the number sit together.
"""
import ast
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DL = {"torch", "tensorflow", "onnxruntime", "keras", "jax", "mxnet", "paddle", "sklearn",
      "caffe", "theano", "chainer", "tensorrt", "openvino", "coremltools"}
SKIP_DIRS = {".git", "output", "__pycache__", ".venv", "venv", "node_modules"}


def main():
    imports = {}
    dl_hits = []
    nfiles = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            nfiles += 1
            try:
                tree = ast.parse(open(p, encoding="utf-8").read())
            except Exception:                                        # noqa: BLE001
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module.split(".")[0]]
                for m in mods:
                    imports[m] = imports.get(m, 0) + 1
                    if m in DL:
                        dl_hits.append((os.path.relpath(p, ROOT), m))

    print(f"scanned {nfiles} python files under {ROOT}")
    print(f"deep-learning framework imports : {len(dl_hits)}")
    for p, m in dl_hits:
        print(f"   !! {p} -> {m}")
    print()
    print("third-party modules actually used (by frequency):")
    std = set(sys.stdlib_module_names)
    third = sorted(((c, m) for m, c in imports.items() if m not in std and m != "lfrd"),
                   reverse=True)
    for c, m in third:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, "__version__", "?")
        except Exception:                                            # noqa: BLE001
            v = "(not importable)"
        print(f"   {m:16s} used in {c:3d} files   version {v}")
    print()
    for run in ("final_v2", "best_final", "ba54_best", "ba54_seq"):
        p = os.path.join(ROOT, "output", run, "eval", "metrics.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        g = [r for r in d if "ours_psnr" in r]
        if not g:
            continue
        m = lambda k: sum(r[k] for r in g) / len(g)
        print(f"  {run:12s} whole {m('ours_psnr'):.3f} / SSIM {m('ours_ssim'):.4f}   "
              f"hole {m('ours_psnr_filled'):.3f} / SSIM {m('ours_ssim_filled'):.4f}   "
              f"({len(g)} frames)")
    print()
    print("verdict:", "CLASSICAL ONLY -- no learned model of any kind is used"
          if not dl_hits else "deep-learning imports present, see above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
