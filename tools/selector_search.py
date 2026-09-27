"""Can a GROUND-TRUTH-FREE selector capture the 3-way oracle gain?

tools/source_select_oracle.py shows a per-pixel choice among {paper-literal, temporal,
reference-paste} would be worth +0.97 dB over the temporal result, but that oracle peeks at
the ground truth.  This evaluates selectors that only use signals available at run time:

  always-temporal        what the project ships now
  always-ref-undiluted   pick reference-paste when eligible, no confidence test
  conf-paste             reference-paste only where the temporal model has weak evidence
  conf-paste+agree       as above, but also require the pasted colour to agree with the
                         current value within a tolerance
  blend                  evidence-weighted blend of temporal and paper-literal

    python tools/selector_search.py --run ba54_seq --temporal_run ba54_temporal `
        --src_cam 5 --dst_cam 4 --frames 0,1,2,3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, io_utils, metrics, temporal


def build(run, trun, src, dst, frame, cams, n_frames=100):
    base = os.path.join(io_utils.run_dir(run, create=False), f"cam{src}-cam{dst}-{frame}")
    ref = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, src, frame)
    gt = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, dst, frame)["color"]
    dis = io_utils.imread(os.path.join(base, "20_warp", "hole_disocc.png"), gray=True) > 0
    cr = io_utils.imread(os.path.join(base, "20_warp", "hole_crack.png"), gray=True) > 0
    oofa = io_utils.imread(os.path.join(base, "20_warp", "hole_oofa.png"), gray=True) > 0
    region = (dis | cr) & ~oofa
    A = io_utils.imread(os.path.join(base, "60_final", "final.png"))
    tp = os.path.join(io_utils.run_dir(trun, create=False),
                      f"cam{src}-cam{dst}-{frame}", "60_final", "final.png")
    B = io_utils.imread(tp) if os.path.isfile(tp) else np.array(A, copy=True)
    rm = io_utils.imread(os.path.join(base, "40_removal", "removed_mask.png"), gray=True) > 0
    # temporal evidence count (recomputed; it is not stored by step5)
    dis_root = io_utils.DATASET_ROOT_DEFAULT
    ys0, xs0 = np.nonzero(rm)
    P = np.stack([io_utils.imread(os.path.join(dis_root, f"cam{src}",
                                                f"depth-cam{src}-f{f:03d}.png"),
                                  gray=True)[ys0, xs0]
                  for f in range(n_frames)])
    z = np.percentile(P, 10, axis=0)
    n_s = ((P <= (z[None, :] + 4.0)) & (P > 0)).sum(0)
    ns = np.zeros(rm.shape, np.int32)
    ns[ys0, xs0] = n_s
    comp = np.load(os.path.join(base, "60_final", "final_depth.npy")).astype(np.float64)
    fg = io_utils.imread(os.path.join(base, "30_class", "fg_mask.png"), gray=True) > 0
    dref = ref["depth"]
    sep = float(np.percentile(dref[dref > 0], 60))
    ys, xs = np.nonzero(region)
    zz = comp[ys, xs]
    with np.errstate(invalid="ignore"):
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), zz,
                                       cams[dst], cams[src])
    bad = ~np.isfinite(ut) | ~np.isfinite(vt)
    ur = np.where(bad, -1, np.rint(np.nan_to_num(ut))).astype(np.int64)
    vr = np.where(bad, -1, np.rint(np.nan_to_num(vt))).astype(np.int64)
    inb = ok & ~bad & (ur >= 0) & (ur < ur.shape[0] * 0 + ref["color"].shape[1]) \
        & (vr >= 0) & (vr < ref["color"].shape[0]) & (zz > 0)
    elig = np.zeros(ys.size, bool)
    jj = np.nonzero(inb)[0]
    if jj.size:
        elig[jj] = (~fg[vr[jj], ur[jj]]) & (dref[vr[jj], ur[jj]] <= sep) \
            & (dref[vr[jj], ur[jj]] > 0)
    C = np.array(A, copy=True)
    selc = np.zeros(region.shape, bool)
    jk = np.nonzero(elig)[0]
    selc[ys[jk], xs[jk]] = True
    if jk.size:
        C[ys[jk], xs[jk]] = ref["color"][vr[jk], ur[jk]]
    return dict(region=region, gt=gt, A=A, B=B, C=C, ns=ns, elig=selc,
                agree=np.abs(A.astype(np.int16) - C.astype(np.int16)).mean(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--temporal_run", required=True)
    ap.add_argument("--src_cam", type=int, required=True)
    ap.add_argument("--dst_cam", type=int, required=True)
    ap.add_argument("--frames", default="0,1,2,3")
    ap.add_argument("--n_frames", type=int, default=100)
    a = ap.parse_args()
    cams = calib.load_calib()
    names = ["temporal(shipped)", "paper-literal", "ref-undiluted", "ref@ns=0",
             "ref@ns<=4", "ref@ns<=4+agree<=40", "ref@ns<=4+agree<=25",
             "oracle(3-way)"]
    vals = {n: [] for n in names}
    for fi in [int(x) for x in a.frames.split(",")]:
        frame = io_utils.frame_name(fi)
        d = build(a.run, a.temporal_run, a.src_cam, a.dst_cam, frame, cams, a.n_frames)
        reg, gt = d["region"], d["gt"]
        if not reg.any():
            continue
        cand = {}
        cand["temporal(shipped)"] = d["B"]
        cand["paper-literal"] = d["A"]
        cand["ref-undiluted"] = d["C"]
        for tag, cond in (("ref@ns=0", d["ns"] == 0),
                          ("ref@ns<=4", d["ns"] <= 4),
                          ("ref@ns<=4+agree<=40", (d["ns"] <= 4) & (d["agree"] <= 40)),
                          ("ref@ns<=4+agree<=25", (d["ns"] <= 4) & (d["agree"] <= 25))):
            img = np.array(d["B"], copy=True)
            take = d["elig"] & cond
            img[take] = d["C"][take]
            cand[tag] = img
        err = np.stack([np.abs(d["A"].astype(np.float32) - gt.astype(np.float32)).mean(2),
                        np.abs(d["B"].astype(np.float32) - gt.astype(np.float32)).mean(2),
                        np.abs(d["C"].astype(np.float32) - gt.astype(np.float32)).mean(2)], 0)
        pick = np.argmin(err, 0)[..., None]
        cand["oracle(3-way)"] = np.where(pick == 0, d["A"], np.where(pick == 1, d["B"], d["C"]))
        row = "  ".join(f"{n.split('(')[0][:9]} {metrics.psnr(cand[n], gt, mask=reg):6.2f}"
                        for n in names if n in cand)
        print(f"{frame}: {row}")
        for n in names:
            if n in cand:
                vals[n].append(metrics.psnr(cand[n], gt, mask=reg))
    print()
    base = float(np.mean(vals["temporal(shipped)"]))
    for n in names:
        if vals[n]:
            v = float(np.mean(vals[n]))
            print(f"  {n:24s} {v:6.2f} dB   ({v - base:+.2f} vs 当前 shipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
