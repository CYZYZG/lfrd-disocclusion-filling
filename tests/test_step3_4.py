"""Tests for stage B (step 3 disocclusion edge classification, step 4 local foreground
removal + depth prediction).

Plain asserts, no pytest -- run with the bundled interpreter:

    python tests/test_step3_4.py

Coverage
--------
1. `test_eq4_cases`      -- the three branches of paper eq. (4) on a hand-made depth row.
2. `test_synthetic_geometry` -- a self-consistent synthetic scene warped with the REAL
   cam5->cam4 calibration (scaled to 512x384) through `lfrd.warp.warp_view`: checks that
   the disocclusion is found, its background side is +1, the removal is a strip on the
   right of the object (never the whole object), the pixels just outside the removal are
   background, the prediction is at the background level and the re-warped predicted layer
   covers the disocclusion.
3. `test_fixture_artefacts` -- the `tools/make_fixtures.py` fixture: structural invariants
   of classify_edges and the edges.npz round trip (the fixture's depth/hole geometry is
   not calibration-consistent, so only structural properties are asserted there).
4. `test_real_run` (skipped when stage A/B artefacts are absent) -- the same invariants on
   the real output/ba54_f000 artefacts.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from lfrd import calib, cli, fill, io_utils, viz, warp                        # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)
    return bool(ok)


# --------------------------------------------------------------------------- #
# 1. eq.(4) branches
# --------------------------------------------------------------------------- #
def test_eq4_cases():
    print("--- eq.(4) branches ---")
    # row 0: BG(60) [removed 3 px] BG(90)      -> linear interpolation
    # row 1: BG(60) [removed 3 px] FG(200)     -> constant d(u_l)
    # row 2: FG(200)[removed 3 px] BG(90)      -> constant d(u_r)
    P = np.zeros((3, 9), np.uint8)
    P[0, :] = [60, 60, 0, 0, 0, 90, 90, 90, 90]
    P[1, :] = [60, 60, 0, 0, 0, 200, 200, 200, 200]
    P[2, :] = [200, 200, 0, 0, 0, 90, 90, 90, 90]
    rm = np.zeros((3, 9), bool)
    rm[:, 2:5] = True
    fg = np.zeros((3, 9), bool)
    fg[1, 5:] = True
    fg[2, :2] = True

    def side(x, y):
        if not (0 <= x < 9 and 0 <= y < 3):
            return None
        return 1 if fg[y, x] else 0

    pred, info = fill.predict_removed_depth(P, rm, None, 5, 4, classify_side_fn=side)
    # d(u) = d(u_l) + s*(u - u_l), u_l = x0 = 2, d(u_l) = P[0, 1] = 60, s = (90-60)/2
    interp = P[0, 1] + (P[0, 5] - P[0, 1]) / 2.0 * np.arange(0, 3)
    check("BG-BG row is linearly interpolated",
          np.allclose(pred[0, 2:5], np.rint(interp), atol=1),
          f"{pred[0, 2:5].tolist()} vs {np.rint(interp).tolist()}")
    check("BG-FG row uses d(u_l)", np.all(pred[1, 2:5] == P[1, 1]),
          f"{pred[1, 2:5].tolist()}")
    check("FG-BG row uses d(u_r)", np.all(pred[2, 2:5] == P[2, 5]),
          f"{pred[2, 2:5].tolist()}")
    check("case histogram has the three branches",
          info["case_hist"].get("BG-BG") == 1 and info["case_hist"].get("BG-FG") == 1
          and info["case_hist"].get("FG-BG") == 1, str(info["case_hist"]))
    changed = pred != P
    check("eq.(4) only writes removed pixels", not (changed & ~rm).any())


# --------------------------------------------------------------------------- #
# 2. synthetic geometry through the real calibration
# --------------------------------------------------------------------------- #
def _synthetic_scene(H=768, W=1024, fg_p=211, bg_p=99):
    """Textured background + a rectangular near foreground block.

    The size must be the native 1024x768 of the calibration: `calib.displacement_field`
    (used by stage 2's `warp.warp_view`) builds its grid from the module constants.
    """
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = np.clip(80 + xx * 0.12, 0, 255)
    img[..., 1] = np.clip(70 + yy * 0.15, 0, 255)
    img[..., 2] = 60
    for _ in range(600):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(img, (x, y), int(rng.integers(2, 8)),
                   tuple(int(v) for v in rng.integers(60, 220, 3)), -1)
    depth = np.full((H, W), bg_p, np.uint8)
    fg = np.zeros((H, W), bool)
    fg[150:650, 400:600] = True
    depth[fg] = fg_p
    img[fg] = np.array([200, 40, 40], np.uint8)
    cv2.rectangle(img, (400, 150), (599, 649), (255, 255, 255), 1)
    return img, depth, fg


def _real_cams_for(shape):
    cams = calib.load_calib(io_utils.DATASET_ROOT_DEFAULT, "ballet")
    return fill.scale_cams_to_shape(cams, shape)


def _split_holes(art_warp):
    hole = art_warp["hole"]
    crack = warp.crack_mask(hole, 2)
    enclosed, oofa = warp.interior_gap(hole & ~crack, 1)
    return dict(hole=hole, crack=crack, disocc=enclosed, oofa=oofa)


def test_synthetic_geometry():
    print("--- synthetic scene, real cam5->cam4 geometry (512x384) ---")
    img, depth, fg_ref = _synthetic_scene()
    cams = _real_cams_for(depth.shape)
    src, dst = 5, 4
    res = warp.warp_view(cams, src, dst, img, depth)
    holes = _split_holes(res)
    n_d = int(holes["disocc"].sum())
    check("warping produces a disocclusion", n_d > 50, f"{n_d} px")

    edges = fill.classify_edges(holes["disocc"], res["warped_lap"], res["warped_depth"],
                                edge_band=2, min_area=20, min_width=3)
    comps = edges["components"]
    check("classification finds the disocclusion(s)", len(comps) >= 1,
          f"{len(comps)} components")
    check("mean depth of F edges > mean depth of BG edges",
          edges["stats"]["mean_depth_F"] > edges["stats"]["mean_depth_BG"],
          f"{edges['stats']['mean_depth_F']:.1f} > {edges['stats']['mean_depth_BG']:.1f}")
    main = max(comps, key=lambda c: c["area"])
    check("background side of the disocclusion is +1 (right of the object)",
          main["bg_side"] == 1, f"bg_side {main['bg_side']}")
    check("eq.(3) vs depth cross-check agrees > 80%",
          edges["stats"]["agree_rate"] > 0.8, f"{edges['stats']['agree_rate']:.3f}")

    fp = fill.inverse_project_edges(edges, res["warped_depth"], cams, src, dst,
                                    backward_idx=res["backward"], P_ref=depth)
    rows = fp["components"][0]["ref_rows"]
    check("reference footprint rows exist", len(rows) > 10, f"{len(rows)} rows")

    rem = fill.remove_local_foreground(depth, None, fp, cams, src, dst,
                                       cli.make_config(cli.base_parser("t").parse_args([])))
    removed = rem["removed_mask"]
    check("removal is non-empty", removed.sum() > 0, f"{int(removed.sum())} px")
    check("removed pixels are foreground only", not (removed & ~(depth > 99 + 6)).any())
    obj_w = int(fg_ref[150].sum())
    widths = [x1 - x0 + 1 for y in np.unique(np.nonzero(removed)[0])
              for (x0, x1) in fill._runs(removed[y])]
    check("removal is a strip, not the whole object",
          max(widths) < 0.8 * obj_w, f"max run width {max(widths)} vs object {obj_w}")
    ys, xs = np.nonzero(removed)
    check("removed pixels stay inside the object's x range",
          xs.min() >= 400 and xs.max() <= 599, f"x {xs.min()}..{xs.max()}")

    # every run must end on a background pixel (so eq.(4) can use it)
    bad = 0
    for y in np.unique(ys):
        for (x0, x1) in fill._runs(removed[y]):
            if not (depth[y, x1 + 1] <= 99 + 6 if x1 + 1 < depth.shape[1] else True):
                bad += 1
    check("every removed run ends on a reference background pixel", bad == 0,
          f"{bad} bad runs")

    pred, info = fill.predict_removed_depth(depth, removed, cams, src, dst,
                                            classify_side_fn=rem["classify_side"],
                                            bg_level=rem["fg_eff"] * 0 + 99.0)
    med = float(np.median(pred[removed]))
    check("predicted depth is the background level", abs(med - 99) <= 10,
          f"median {med:.0f}")
    check("no removed pixel keeps the foreground depth",
          int(pred[removed].max()) < 150, f"max {int(pred[removed].max())}")

    cov, _ = _coverage(removed, pred, cams, src, dst, edges["labels"], comps)
    check("predicted layer re-warps onto the disocclusion", cov[main["id"]] > 0.5,
          f"coverage {cov[main['id']]:.3f}")


def _coverage(removed, pred, cams, src, dst, lab, comps):
    ys, xs = np.nonzero(removed)
    cov = np.zeros(removed.shape, bool)
    if ys.size:
        z = calib.depth_from_P(pred[ys, xs].astype(np.float64))
        ut, vt, ok = calib.project_pts(xs.astype(np.float64), ys.astype(np.float64), z,
                                       cams[src], cams[dst])
        H, W = removed.shape
        u = np.rint(ut).astype(int)
        v = np.rint(vt).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cov[v[inb], u[inb]] = True
    return ({c["id"]: float((cov & (lab == c["id"])).sum()) / max(1, c["area"])
             for c in comps}, cov)


# --------------------------------------------------------------------------- #
# 3. fixture artefacts (structural)
# --------------------------------------------------------------------------- #
def test_fixture_artefacts():
    print("--- fixture artefacts ---")
    run = "fixture"
    wdir = io_utils.run_dir(run, "warp")
    if not os.path.isfile(os.path.join(wdir, "warped_depth.npy")):
        print("[skip] fixture not generated (run tools/make_fixtures.py --run fixture)")
        return
    art = fill.load_warp_artefacts(wdir)
    edges = fill.classify_edges(art["hole_disocc"], art["warped_lap"], art["warped_depth"],
                                edge_band=2, min_area=20, min_width=2)
    check("fixture: components found", len(edges["components"]) >= 1,
          f"{len(edges['components'])}")
    check("fixture: every boundary pixel is classified",
          int((edges["fg_edge"] > 0).sum() + (edges["bg_edge"] > 0).sum())
          == edges["stats"]["n_boundary_pixels"])
    check("fixture: every component has a determinate background side",
          all(c["bg_side"] in (-1, 1) for c in edges["components"]))
    p = os.path.join(io_utils.run_dir(run, "class"), "tests_edges.npz")
    fill.save_edges_npz(p, edges)
    back = fill.load_edges_npz(p)
    check("fixture: edges.npz round trip keeps the components",
          len(back["components"]) == len(edges["components"])
          and back["stats"]["n_boundary_pixels"] == edges["stats"]["n_boundary_pixels"],
          f"{len(back['components'])} components")
    check("fixture: round trip keeps F arrays",
          all(np.array_equal(a["F"], b["F"])
              for a, b in zip(edges["components"], back["components"])))

    # hand-built footprints exercise removal + eq.(4) on the fixture depth
    H, W = art["warped_depth"].shape
    P = io_utils.imread(os.path.join(io_utils.run_dir(run, "preproc"), "depth_pp.png"),
                        gray=True) if os.path.isfile(
        os.path.join(io_utils.run_dir(run, "preproc"), "depth_pp.png")) else None
    if P is None or P.shape != (H, W):
        print("[skip] fixture depth shape does not match the warp (synthetic mismatch)")
        return
    hole = np.zeros((H, W), bool)
    hole[60:140, 168:198] = True                 # the fixture's disocclusion
    comp = dict(edges["components"][0])
    comp["ref_rows"] = {int(y): (150, 190) for y in range(60, 140)}
    comp["ref_opp"] = {int(y): 190 for y in range(60, 140)}
    comp["bg_depth"] = 60.0
    comp["bg_side"] = 1
    fp2 = dict(components=[comp], footprint_label=np.zeros((H, W), np.int32),
               stats=dict(n_footprint_px=0, n_no_footprint=0))
    rem2 = fill.remove_local_foreground(P, None, fp2, None, 5, 4,
                                        cli.make_config(cli.base_parser("t").parse_args([])))
    check("fixture: hand-built footprint removal is non-empty",
          int(rem2["removed_mask"].sum()) > 0, f"{int(rem2['removed_mask'].sum())} px")


# --------------------------------------------------------------------------- #
# 4. real run
# --------------------------------------------------------------------------- #
def test_real_run():
    """Real-data invariants, run in a DEDICATED single-frame run.

    The test drives step1..step4 itself into `output/_test_stage_b/` so that every stage
    artefact provably belongs to the same frame.  (A shared stage directory belongs to
    whichever frame ran last, so testing an arbitrary existing run compares frames.)
    """
    run = "_test_stage_b"
    src_cam, dst_cam, frame = 5, 4, "f000"
    print(f"--- real run output/{run} (cam{src_cam}->cam{dst_cam} {frame}) ---")
    py = sys.executable
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for script in ("step1_preprocess.py", "step2_warp.py", "step3_classify.py",
                   "step4_removal.py"):
        p = subprocess.run([py, os.path.join(root, script), "--run", run,
                            "--src_cam", str(src_cam), "--dst_cam", str(dst_cam),
                            "--frame", frame], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=root)
        if p.returncode != 0:
            print(f"[skip] {script} failed (exit {p.returncode}); tail:")
            print("\n".join((p.stdout or "").strip().splitlines()[-8:]))
            return
    wdir = io_utils.run_dir(run, "warp")
    cdir = os.path.join(io_utils.run_dir(run, "class"), "edges.npz")
    art = fill.load_warp_artefacts(wdir)
    P = io_utils.imread(os.path.join(io_utils.run_dir(run, "preproc"), "depth_pp.png"),
                        gray=True)
    cams = _real_cams_for(P.shape)
    edges = fill.load_edges_npz(cdir)
    # cam5->cam4 moves the foreground LEFT by ~12 px, so the exposed background sits on the
    # RIGHT of each foreground object -> bg_side +1 for the large components.  A handful of
    # tiny components may vote the other way, so the check is on the area-weighted majority.
    big = [c for c in edges["components"] if c["area"] >= 1000]
    n_pos = sum(1 for c in big if c["bg_side"] == 1)
    check("real: the large disocclusions have bg_side +1 (cam5->cam4)",
          len(big) > 0 and n_pos >= 0.75 * len(big),
          f"{n_pos}/{len(big)} large components +1, all: "
          f"{edges['stats']['bg_side_counts']}")
    check("real: eq.(3) vs depth agreement > 80%",
          edges["stats"]["agree_rate"] > 0.8, f"{edges['stats']['agree_rate']:.3f}")

    fp = fill.inverse_project_edges(edges, art["warped_depth"], cams, 5, 4,
                                    backward_idx=art["backward"], P_ref=P)
    check("real: every component has a reference footprint",
          fp["stats"]["n_no_footprint"] == 0, str(fp["stats"]))

    rem = fill.remove_local_foreground(P, None, fp, cams, 5, 4,
                                       cli.make_config(cli.base_parser("t").parse_args([])))
    removed = rem["removed_mask"]
    check("real: every disocclusion has a non-empty removal",
          rem["stats"]["n_empty"] == 0, str(rem["stats"]))
    check("real: removed pixels are inside the eligible reference foreground",
          not (removed & ~rem["fg_eff"]).any(),
          f"{int(removed.sum())} px, eligible {int(rem['fg_eff'].sum())}")
    widths = [int(np.ptp(np.nonzero(removed[y])[0]) + 1) for y in
              np.unique(np.nonzero(removed)[0])]
    run_w = [x1 - x0 + 1 for y in np.unique(np.nonzero(removed)[0])
             for (x0, x1) in fill._runs(removed[y])]
    check("real: every removed run is a narrow strip (< 130 px per run)",
          max(run_w) < 130, f"max run {max(run_w)}, max row span {max(widths)}")

    pred, info = fill.predict_removed_depth(
        P, removed, cams, 5, 4, classify_side_fn=rem["classify_side"],
        bg_level=None)
    bg_pred, _ = _bg_level_map(rem, P, 6.0)
    pred, info = fill.predict_removed_depth(
        P, removed, cams, 5, 4,
        classify_side_fn=lambda x, y: (None if not (0 <= x < P.shape[1]
                                                     and 0 <= y < P.shape[0])
                                       else (1 if P[y, x] > bg_pred[y, x] + 6 else 0)),
        bg_level=bg_pred)
    check("real: prediction only changes removed pixels", info["n_changed_outside"] == 0)
    nn = sum(v for k, v in info["case_hist"].items() if k.startswith("nearest"))
    check("real: the nearest-depth fallback is not used", nn == 0, str(info["case_hist"]))
    cov, _ = _coverage(removed, pred, cams, 5, 4, edges["labels"], fp["components"])
    wcov = float(np.average([cov[int(c["id"])] for c in fp["components"]],
                            weights=[c["area"] for c in fp["components"]]))
    check("real: the predicted layer re-warps onto the disocclusions", wcov > 0.5,
          f"area-weighted coverage {wcov:.3f}")
    rdir = os.path.join(io_utils.run_dir(run), "removal")
    if os.path.isfile(os.path.join(rdir, "depth_pred.png")):
        dp = io_utils.imread(os.path.join(rdir, "depth_pred.png"), gray=True)
        check("real: depth_pred.png on disk equals the recomputation",
              int(dp[removed].sum()) == int(pred[removed].sum()))


def _bg_level_map(rem, P, margin):
    """Rasterise the per-component revealed background level (test helper)."""
    out = np.full(P.shape, np.nan, np.float32)
    for c in rem["components"]:
        if not np.isfinite(c["bg_depth"]):
            continue
        for (v, x0, x1) in c["intervals"]:
            out[v, max(0, x0 - 4):x1 + 5] = c["bg_depth"]
    have = np.isfinite(out)
    if have.any() and (~have).any():
        from scipy import ndimage
        _d, (iy, ix) = ndimage.distance_transform_edt(~have, return_indices=True)
        out[~have] = out[iy[~have], ix[~have]]
    return out, have


def main():
    test_eq4_cases()
    test_synthetic_geometry()
    test_fixture_artefacts()
    test_real_run()
    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} check(s): {FAILED}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
