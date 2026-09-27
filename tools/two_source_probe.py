"""Is the current two-source composition optimal?

Step 6 already uses the warped occlusion layer where it has a sample and postprocessing
inpainting elsewhere.  The alternative is to inpaint the disocclusion DIRECTLY in the virtual
view (what the comparison methods do) and then choose per pixel.  If direct inpainting were
better on the pixels where the occlusion layer exists, a per-pixel selector could win; this
measures both sources on exactly the same masks.

    python tools/two_source_probe.py --src_cam 5 --dst_cam 4 --frames 0,1,2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, inpaint as I, io_utils, metrics, warp as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2")
    ap.add_argument("--run_prefix", default="ba54_seq")
    a = ap.parse_args()
    cams = calib.load_calib()
    print(f"{'frame':6s} {'occ px':>8s} {'rest px':>8s} {'occ PSNR':>9s} {'rest: post':>11s} "
          f"{'rest: direct':>13s} {'best of 2':>10s}")
    tot = {"occ": [], "post": [], "direct": [], "best": []}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        base = os.path.join(io_utils.run_dir(f"{a.run_prefix}__cam{a.src_cam}-{a.dst_cam}-{frame}",
                                            create=False))
        if not os.path.isdir(base):
            print(f"[skip] {frame}: no {base}")
            continue
        gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, a.dst_cam, frame)["color"]
        occ = io_utils.imread(os.path.join(base, "50_fill", "filled_occlusion.png"))
        occ_d = io_utils.imread(os.path.join(base, "50_fill",
                                             "filled_occlusion_depth.png"), gray=True)
        final = io_utils.imread(os.path.join(base, "60_final", "final.png"))
        wd = os.path.join(base, "20_warp")
        warp_c = io_utils.imread(os.path.join(wd, "warped_color.png"))
        dis = io_utils.imread(os.path.join(wd, "hole_disocc.png"), gray=True) > 0
        cr = io_utils.imread(os.path.join(wd, "hole_crack.png"), gray=True) > 0
        oofa = io_utils.imread(os.path.join(wd, "hole_oofa.png"), gray=True) > 0
        wdep = np.load(os.path.join(wd, "warped_depth.npy")).astype(np.float32)
        r = W.warp_view(cams, a.src_cam, a.dst_cam, occ, occ_d)
        take = dis & ~r["hole"]
        rest = dis & r["hole"]

        # direct inpainting of the virtual view on the whole fillable region
        todo = (dis | cr) & ~oofa
        dep = np.clip(np.where(wdep < 0, 0, wdep), 0, 255).astype(np.uint8)
        res = I.inpaint(warp_c, todo, dep, patch_size=9, search_w=160, search_h=120,
                        use_bg_term=False, use_depth_term=True, use_depth_limit=False,
                        return_meta=True)
        direct = res["filled"]
        # the two sources on their own masks
        def sub_to(img, mask):
            out = np.array(final, copy=True)
            out[mask] = img[mask]
            return out
        p_occ = metrics.psnr(final, gt, mask=take) if take.any() else float("nan")
        p_post = metrics.psnr(final, gt, mask=rest) if rest.any() else float("nan")
        # replace the postprocessed pixels with the direct-inpainted ones
        cand = sub_to(direct, rest)
        p_dir = metrics.psnr(cand, gt, mask=rest) if rest.any() else float("nan")
        # oracle selector: per pixel take whichever of the two is closer to the GT
        gtd = np.abs(final.astype(np.float32) - gt.astype(np.float32)).mean(2)
        dd = np.abs(direct.astype(np.float32) - gt.astype(np.float32)).mean(2)
        pick_direct = rest & (dd < gtd)
        best = sub_to(direct, pick_direct)
        p_best = metrics.psnr(best, gt, mask=rest) if rest.any() else float("nan")
        print(f"{frame:6s} {int(take.sum()):8d} {int(rest.sum()):8d} {p_occ:9.2f} "
              f"{p_post:11.2f} {p_dir:13.2f} {p_best:10.2f}")
        tot["occ"].append(p_occ)
        tot["post"].append(p_post)
        tot["direct"].append(p_dir)
        tot["best"].append(p_best)
    if tot["occ"]:
        m = lambda k: float(np.nanmean(tot[k]))
        print(f"\nMEAN  occlusion-layer px {m('occ'):.2f} | postprocessed px "
              f"{m('post'):.2f} | same px with direct inpaint {m('direct'):.2f} | "
              f"oracle per-pixel pick {m('best'):.2f}")
        print(f"=> a perfect selector on the postprocessed pixels would gain "
              f"{m('best') - m('post'):+.2f} dB THERE, i.e. "
              f"{(m('best') - m('post')) * 0.14:+.3f} dB over the whole disocclusion "
              f"(those pixels are ~14% of it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
