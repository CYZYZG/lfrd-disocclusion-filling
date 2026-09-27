"""Look harder for anything that could serve as a learned prior, and check package access."""
import glob
import importlib.util
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
PY = sys.executable

print("=== 1. pip access (offline?) ===")
for cmd in ([PY, "-m", "pip", "--version"], [PY, "-m", "pip", "download", "--no-deps",
                                             "-d", os.path.join(os.environ.get("TEMP", "."),
                                                                "pipchk"), "opencv-contrib-python"]):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=90)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        print(f"  {' '.join(cmd[2:]):50s} rc={r.returncode}  {out[-1][:90] if out else ''}")
    except Exception as e:                                          # noqa: BLE001
        print(f"  {' '.join(cmd[2:]):50s} FAILED {type(e).__name__}")

print()
print("=== 2. cv2 / PIL bundled data ===")
import cv2
import numpy as np
print("  cv2 data dir :", getattr(cv2, "data", None) and getattr(cv2.data, "haarcascades", "?"))
print("  cv2 samples  :", getattr(cv2, "samples", "?"))
import PIL
print("  PIL path     :", os.path.dirname(PIL.__file__))
extra = os.path.join(os.path.dirname(PIL.__file__), "..", "cv2", "data")
print("  cv2 data exists:", os.path.isdir(extra))
if os.path.isdir(extra):
    for dp, dn, fn in os.walk(extra):
        for f in fn:
            print("   ", os.path.join(dp, f))

print()
print("=== 3. any model-ish file on C: and D: drives (deeper search) ===")
pats = ["*.onnx", "*.caffemodel", "*.prototxt", "*.tflite", "*.pb", "*.weights", "*.h5"]
hits = []
for root in ("C:\\Users\\ZHJ", "D:\\项目", "C:\\Program Files", "C:\\ProgramData"):
    if not os.path.isdir(root):
        continue
    for p in pats:
        for f in glob.glob(os.path.join(root, "**", p), recursive=True):
            try:
                sz = os.path.getsize(f)
            except OSError:
                continue
            if sz > 300_000:
                hits.append((sz, f))
    if len(hits) > 60:
        break
hits.sort(reverse=True)
print(f"  found {len(hits)}")
for sz, f in hits[:25]:
    print(f"    {sz / 1e6:8.1f} MB  {f}")

print()
print("=== 4. python wheels / caches with DL content ===")
for d in (os.path.join(os.environ.get("LOCALAPPDATA", ""), "pip", "Cache"),
          os.path.join(os.path.expanduser("~"), ".cache", "pip"),
          os.path.join(os.environ.get("TEMP", ""), "pipchk")):
    ok = os.path.isdir(d)
    n = len(glob.glob(os.path.join(d, "**", "*.whl"), recursive=True)) if ok else 0
    print(f"  {d}: exists={ok} wheels={n}")
