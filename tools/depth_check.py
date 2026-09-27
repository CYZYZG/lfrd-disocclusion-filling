"""How good is the predicted occlusion-layer DEPTH (paper III-D) ?

The paper's step 5.1 predicts the depth of the region it removed; that predicted depth is
what carries the synthesised texture to the right place in the virtual view.  A wrong depth
immediately shows up as a misplaced patch in the hole.  This script measures it directly:

    error = | predicted reference depth -> warped into the virtual view
              - the real capture's depth of the virtual camera |

evaluated on the disocclusion pixels (where the prediction is actually used).

    python tools/depth_check.py --run q_67 --src_cam 6 --dst_cam 7 --frame f000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, warp as warp_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frame", default="f000")
    a = ap.parse_args()
    fd = os.path.join(io_utils.run_dir(a.run, create=False),
                      f"cam{a.src_cam}-cam{a.dst_cam}-{a.frame}")
    w = os.path.join(fd, "20_warp")
    fdir = os.path.join(fd, "60_final")

    cams = calib.load_calib()
    ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.src_cam, a.frame)
    gt_view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, a.frame)
    gt_color = gt_view["color"]
    gt_depth = gt_view["depth"]            # ground-truth inverse depth of the VIRTUAL camera

    ours = io_utils.imread(os.path.join(fdir, "final.png"))
    hole = io_utils.imread(os.path.join(w, "hole_all.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(w, "hole_oofa.png"), gray=True) > 0
    dis = io_utils.imread(os.path.join(w, "hole_disocc.png"), gray=True) > 0
    filled = hole & ~oofa

    # the final depth map our pipeline produced for the virtual view (if saved)
    dp = os.path.join(fdir, "final_depth.npy")
    has_depth = os.path.isfile(dp)
    # what our final depth is on the disocclusion: prefer the warped prediction layer
    slot = os.path.join(io_utils.run_dir(a.run, create=False), "50_fill",
                        "filled_occlusion_depth.png")
    pred_depth = None
    if os.path.isfile(slot):
        pred_depth = io_utils.imread(slot, gray=True)

    print(f"run={a.run} cam{a.src_cam}->cam{a.dst_cam} {a.frame}")
    print(f"  disocclusion {int(dis.sum())} px, ground-truth depth available "
          f"({int((gt_depth > 0).sum())} nz px)")
    if not has_depth:
        print("  [warn] 60_final/final_depth.npy missing -> cannot locate our predicted "
              "depth in the virtual view")
    # ---- baseline sanity: how well does the GROUND TRUTH reference depth reproduce the
    #      ground-truth virtual depth when warped? that is the ceiling of this measure.
    P_ref = ref["depth"]
    z_ref = calib.depth_from_P(P_ref)
    dx, dy, _, _, oob = calib.displacement_field(cams, a.src_cam, a.dst_cam, P_ref)
    wz, _, hole_w, _ = warp_mod.forward_warp(z_ref.astype(np.float32), dx.astype(np.float32),
                                             dy.astype(np.float32), z=z_ref.astype(np.float32),
                                             rule="zbuf", splat="sub", hole_depth=-1.0)
    z_gt = calib.depth_from_P(gt_depth)
    m = dis & (wz > 0) & (gt_depth > 0)
    if m.any():
        e = np.abs(np.log(wz[m]) - np.log(z_gt[m]))
        print(f"  CEILING (ground-truth reference depth, warped) on disocclusion "
              f"({int(m.sum())} px): median log-error {np.median(e):.4f}, "
              f"p90 {np.percentile(e, 90):.4f}")
    # ---- our predicted depth: the occlusion layer warped with the PREDICTED depth
    if pred_depth is not None and os.path.isfile(dp):
        # rebuild the warped prediction from the occlusion layer + predicted depth
        rem = io_utils.imread(os.path.join(io_utils.run_dir(a.run, create=False),
                                           "40_removal", "removed_depth.png"), gray=True)
        occ_depth = io_utils.imread(os.path.join(io_utils.run_dir(a.run, create=False),
                                                 "50_fill",
                                                 "filled_occlusion_depth.png"), gray=True)
        pred = io_utils.imread(os.path.join(io_utils.run_dir(a.run, create=False),
                                            "40_removal", "depth_pred.png"), gray=True)
        filled_ref = io_utils.imread(os.path.join(io_utils.run_dir(a.run, create=False),
                                                  "50_fill",
                                                  "filled_occlusion.png"))
        ww = warp_mod.warp_view(cams, a.src_cam, a.dst_cam, filled_ref, occ_depth)
        got = ww["warped_depth"]
        mm = dis & (got > 0) & (gt_depth > 0)
        if mm.any():
            zg = calib.depth_from_P(got[mm])
            zt = calib.depth_from_P(gt_depth[mm])
            e = np.abs(np.log(zg) - np.log(zt))
            print(f"  OURS  predicted layer depth on disocclusion ({int(mm.sum())} px): "
                  f"median log-error {np.median(e):.4f}, p90 {np.percentile(e, 90):.4f}")
            # what the depth error means in pixels for this baseline
            base = abs(calib.baseline(cams, a.src_cam, a.dst_cam))
            fx = cams[a.dst_cam]["K"][0, 0]
            px = np.exp(np.median(e)) * 0
            print(f"  (baseline {base:.2f}, fx {fx:.0f}: a log-depth error of 0.05 displaces "
                  f"a patch by roughly {fx * base * 0.05 / 80:.1f} px at z=80)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
