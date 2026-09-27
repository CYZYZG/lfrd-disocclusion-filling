"""Minimal, runnable examples for the fill_holes interface.

    python examples_fill_holes.py            # runs all the examples that work here
    python examples_fill_holes.py --case 1   # just one

See 接口使用说明.md for the full documentation.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import io_utils          # noqa: E402
from lfrd.api import fill_holes    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_examples")


def case1_paper():
    """The paper's own setting: one reference view, single frame, method region only."""
    res = fill_holes("ex1_paper", src_cam=5, dst_cam=4, frame="f000")
    print("  ", res.summary())
    print("   saved:", res.save(os.path.join(OUT, "ex1_paper.png")))


def case2_best():
    """Best measured: temporal background + OOFA filling + photometric match (default)."""
    res = fill_holes("ex2_best", src_cam=5, dst_cam=4, frame="f000",
                     temporal_frames=100, fill_oofa=True)
    print("  ", res.summary())
    print("   saved:", res.save(os.path.join(OUT, "ex2_best.png")))


def case3_single_view_plus_refguide():
    """No sequence available: reference-guided patch search is the better single-view fill."""
    res = fill_holes("ex3_refguide", src_cam=5, dst_cam=4, frame="f000", refguide=True)
    print("  ", res.summary())
    print("   saved:", res.save(os.path.join(OUT, "ex3_refguide.png")))


def case4_own_images():
    """Your own pictures: array in, PNG out.  Needs a calibration file (MSR format)."""
    root = io_utils.DATASET_ROOT_DEFAULT
    ref = io_utils.load_view(root, 0, "f000")            # stand in for "your left view"
    gt = io_utils.load_view(root, 1, "f000")["color"]    # stand in for "your right view"
    res = fill_holes("ex4_own", src_cam=0, dst_cam=1, frame="f000",
                     ref_image=ref["color"], ref_depth=ref["depth"],
                     calib_file=os.path.join(root, "calibParams-ballet.txt"),
                     gt_image=gt, strict=False)
    print("  ", res.summary())
    print("   saved:", res.save(os.path.join(OUT, "ex4_own.png")))


def case5_masks():
    """Work with the masks and depth the result carries, not just the picture."""
    import numpy as np
    res = fill_holes("ex5_masks", src_cam=5, dst_cam=4, frame="f000", quiet=True)
    files = res.save_all(os.path.join(OUT, "ex5_bundle"))
    print(f"   hole {int(res.hole.sum())} px = disocclusion {int(res.disocclusion.sum())} "
          f"+ crack {int(res.crack.sum())} + OOFA {int(res.oofa.sum())}")
    print(f"   filled from the occlusion layer: {int(res.filled_from_layer.sum())} px")
    print(f"   untouched warp-valid pixels   : {int(res.valid_untouched.sum())} px "
          f"(PSNR {res.metrics.get('psnr_valid', float('nan')):.3f} dB, "
          f"identical to the plain warp)")
    print(f"   bundle: {len(files)} files in {os.path.join(OUT, 'ex5_bundle')}")


def case6_loop():
    """Batch: several frames, one run name, per-frame artefacts kept apart."""
    import numpy as np
    vals = []
    for i in range(3):
        frame = io_utils.frame_name(i)
        res = fill_holes("ex6_loop", src_cam=5, dst_cam=4, frame=frame,
                         temporal_frames=100, fill_oofa=True, quiet=True)
        vals.append(res.metrics.get("psnr_filled", float("nan")))
        print(f"   {frame}: hole PSNR {vals[-1]:.2f} dB, whole "
              f"{res.metrics.get('psnr_whole', float('nan')):.2f} dB")
    print(f"   mean over {len(vals)} frames: {np.mean(vals):.2f} dB")


CASES = {1: case1_paper, 2: case2_best, 3: case3_single_view_plus_refguide,
         4: case4_own_images, 5: case5_masks, 6: case6_loop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = [a.case] if a.case else sorted(CASES)
    for c in todo:
        fn = CASES.get(c)
        if fn is None:
            print(f"unknown case {c}")
            continue
        print(f"\n=== case {c}: {fn.__doc__.splitlines()[0]} ===")
        t = time.time()
        try:
            fn()
        except Exception as exc:                                     # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {exc}")
        print(f"   ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
