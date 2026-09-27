"""Probe what is available for a learned inpainting approach."""
import importlib
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

print("python:", sys.version.split()[0])
print()
for m in ("torch", "tensorflow", "onnxruntime", "cv2", "numpy", "scipy", "sklearn", "PIL"):
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {m:14s} OK   {v}")
    except Exception as e:                                          # noqa: BLE001
        print(f"  {m:14s} --   {type(e).__name__}")

cv2 = importlib.import_module("cv2")
print()
print("cv2.dnn          :", hasattr(cv2, "dnn"))
print("cv2.inpaint      :", hasattr(cv2, "inpaint"))
print("cv2.xphoto       :", hasattr(cv2, "xphoto"))
print()

# look for any model weights already on disk
ROOTS = [r"C:\Users\ZHJ\.dsh", r"D:\项目", os.path.dirname(os.__file__)]
EXT = (".pth", ".pt", ".onnx", ".caffemodel", ".pb", ".tflite", ".h5", ".weights")
found = []
for r in ROOTS:
    if not os.path.isdir(r):
        continue
    for dirpath, dirnames, filenames in os.walk(r):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
        for f in filenames:
            if f.lower().endswith(EXT):
                p = os.path.join(dirpath, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > 200_000:
                    found.append((sz, p))
    if len(found) > 40:
        break
print(f"model-like files >200 KB under the searched roots: {len(found)}")
for sz, p in sorted(found, reverse=True)[:15]:
    print(f"  {sz / 1e6:8.1f} MB  {p}")
