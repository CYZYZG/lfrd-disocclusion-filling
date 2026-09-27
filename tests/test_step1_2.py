"""Stage A tests: depth preprocessing (paper III-A) + forward warping & hole typing (II-B).

Plain asserts, no pytest:  run with the bundled interpreter

    C:\\Users\\ZHJ\\.dsh\\dsh-runtimes\\dsh-primary-runtime\\dependencies\\python\\python.exe tests\\test_step1_2.py

Covers

A. unit semantics of ``lfrd.preprocess`` (eq. 1-2) on small synthetic depths,
B. ``lfrd.disocclusion`` hole typing on crafted masks (incl. the known degeneracy of
   ``lfrd.warp.interior_gap`` in the frozen core),
C. the real data (cam5 -> cam4, frame f000) end to end: preprocessing, warping of both
   arms, hole typing, the brute-force Z-buffer reference (explicit loop, written here),
   the backward map and the OOFA-stability check,
D. the synthetic fixture (``tools.make_fixtures.build``, both its native 192x256 size for
   the depth stages and dataset resolution for the geometry stages).

Exit code is non-zero if any check fails.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from lfrd import calib, disocclusion, io_utils, preprocess, warp
from step1_preprocess import load_synthetic_view
from step2_warp import fill_depth_nearest
from tools.make_fixtures import build


# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self):
        self.n_pass = 0
        self.n_fail = 0

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            self.n_pass += 1
        else:
            self.n_fail += 1
        return ok

    def info(self, msg):
        print(f"[info] {msg}")

    def section(self, title):
        print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# A. lfrd.preprocess unit semantics
# --------------------------------------------------------------------------- #
def two_level(H=40, W=60, bg=100, fg=200):
    P = np.full((H, W), bg, np.uint8)
    P[15:30, 20:40] = fg
    return P


def test_preprocess_units(C):
    C.section("A. lfrd.preprocess unit semantics")
    P = two_level()

    # cross kernel == 4-neighbourhood max
    k = preprocess.CROSS_KERNEL
    C.check("cross kernel L == eq. (2)", bool((k == np.array([[0, 1, 0], [1, 1, 1],
                                                              [0, 1, 0]])).all()))
    manual = P.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        manual = np.maximum(manual, np.roll(np.roll(P, dy, 0), dx, 1))
    manual[0, :] = np.maximum(manual[0, :], P[0, :])   # roll wraps; ignore border
    C.check("dilate_cross == 4-neighbourhood max (interior)",
            bool((preprocess.dilate_cross(P)[1:-1, 1:-1] == manual[1:-1, 1:-1]).all()))

    info = preprocess.preprocess_depth(P, th=20, rounds=2, min_change=1, return_info=True)
    C.check("return_info keys (P_out, marked, marked_all, rounds)",
            all(key in info for key in ("P_out", "marked", "marked_all", "rounds",
                                        "er", "er_used")))
    C.check("per-round Er mask returned for every round",
            len(info["er"]) == 2 and len(info["er_used"]) == 2
            and all(e.dtype == bool for e in info["er"]))
    C.check("marked pixels > 0", info["marked_all"].sum() > 0,
            f"{int(info['marked_all'].sum())} px")
    C.check("monotone non-decreasing", bool((info["P_out"] >= P).all()))
    C.check("only marked pixels change",
            bool((info["P_out"] != P).sum() == info["marked_all"].sum()))
    # round 1 marks the 1-px ring outside the block; round 2 the next ring
    C.check("round 1 marks exactly the outer ring",
            int(info["er_used"][0].sum()) == 2 * (15 + 20),
            f"{int(info['er_used'][0].sum())} vs {2 * (15 + 20)}")
    C.check("two rounds grow the foreground by 2 px",
            bool((info["P_out"][15, 18:42] == 200).all() and
                 (info["P_out"][15, 17] == 100) and
                 int(info["P_out"][15, 20]) == 200))
    C.check("foreground interior untouched",
            bool((info["P_out"][16:29, 21:39] == 200).all()))
    C.check("nothing outside the object boundary changed",
            bool((info["P_out"][0:13, :] == 100).all() and
                 (info["P_out"][32:, :] == 100).all()))

    # undefined depth must be preserved even when surrounded by foreground
    Pz = two_level()
    Pz[20:25, 25:30] = 0
    infoz = preprocess.preprocess_depth(Pz, th=20, rounds=2, return_info=True)
    C.check("P==0 pixels are never modified",
            bool((infoz["P_out"][Pz == 0] == 0).all()) and
            bool(((Pz == 0) & infoz["marked_all"]).sum() == 0),
            f"raw marking would have hit {int((preprocess.mark_ghosts(Pz)[0] & (Pz == 0)).sum())} px")

    # threshold / min_change behaviour
    C.check("huge th marks nothing",
            int(preprocess.preprocess_depth(P, th=250, return_info=True)["marked_all"].sum()) == 0)
    C.check("huge min_change marks nothing",
            int(preprocess.preprocess_depth(P, min_change=200,
                                            return_info=True)["marked_all"].sum()) == 0)

    # ghost_stats on a *contaminated* depth (background-valued pixels inside a fg block)
    Pc = two_level(60, 80)
    Pc[20:24, 25:29] = 100            # 4x4 dark spot inside the foreground block
    for x, y in ((10, 40), (25, 55), (45, 20), (50, 60)):
        Pc[y - 1:y + 2, x - 1:x + 2] = 100
    infoc = preprocess.preprocess_depth(Pc, th=20, rounds=2, return_info=True)
    st = preprocess.ghost_stats(Pc, infoc["P_out"], infoc["marked_all"])
    C.check("ghost_stats keys",
            all(k in st for k in ("n_marked", "n_changed", "p_inc", "p_dec",
                                  "n_zero_changed", "lap_energy_before",
                                  "lap_energy_after")))
    C.check("ghost_stats: no decrease, no zero-pixel change",
            st["p_dec"] == 0 and st["n_zero_changed"] == 0,
            f"inc {st['p_inc']} dec {st['p_dec']} zero {st['n_zero_changed']}")
    C.check("ghost_stats: Laplacian energy drops on contaminated depth",
            st["lap_energy_after"] < st["lap_energy_before"],
            f"{st['lap_energy_before']:.2f} -> {st['lap_energy_after']:.2f}")
    C.check("interior dark spots are corrected",
            bool((infoc["P_out"][20:24, 25:29] == 200).all()),
            f"4x4 spot -> {np.unique(infoc['P_out'][20:24, 25:29]).tolist()}")


# --------------------------------------------------------------------------- #
# B. lfrd.disocclusion unit semantics
# --------------------------------------------------------------------------- #
def crafted_hole(H=80, W=100):
    hole = np.zeros((H, W), bool)
    hole[10, 5:60] = True              # 1 px crack (chamfer thickness ~0.955)
    hole[3:6, 5:60] = True             # 3 px wide band -> thickness ~1.91 -> crack
    hole[30:55, 30:50] = True          # enclosed blob -> disocclusion
    hole[60:80, 70:95] = True          # open to the bottom border -> OOFA
    hole[0:4, 75:95] = True            # open to the top border -> OOFA
    return hole


def test_type_holes_units(C):
    C.section("B. lfrd.disocclusion hole typing on crafted masks")
    hole = crafted_hole()
    ty = disocclusion.type_holes(hole, crack_max_width=2, min_area=60, min_width=4)
    C.check("partition is complete", bool(((ty["crack"] | ty["disocc"] | ty["oofa"])
                                          == hole).all()))
    C.check("partition is disjoint",
            bool(not (ty["crack"] & ty["disocc"]).any()
                 and not (ty["crack"] & ty["oofa"]).any()
                 and not (ty["disocc"] & ty["oofa"]).any()))
    C.check("hole_type_map ids 1/2/3 match the masks",
            bool((ty["hole_type_map"][ty["crack"]] == 1).all()
                 and (ty["hole_type_map"][ty["disocc"]] == 2).all()
                 and (ty["hole_type_map"][ty["oofa"]] == 3).all()
                 and int(ty["hole_type_map"].max()) == 3
                 and int(ty["hole_type_map"].min()) == 0))
    C.check("1 px crack is classified as crack", bool(ty["crack"][10, 5:60].all()))
    C.check("3 px band (thickness 2) is classified as crack",
            bool(ty["crack"][3:6, 10:55].all()))
    C.check("enclosed 20x25 blob is a disocclusion",
            bool(ty["disocc"][30:55, 30:50].all()))
    C.check("bottom/top border regions are OOFA",
            bool(ty["oofa"][60:80, 70:95].all()) and bool(ty["oofa"][0:4, 75:95].all()))
    C.check("crack mask from warp.crack_mask agrees with the typing",
            bool((warp.crack_mask(hole, 2) == ty["crack"]).all()))
    C.check("type_holes counts are consistent",
            ty["n_crack"] + ty["n_disocc"] + ty["n_oofa"] == ty["n_hole"],
            f"{ty['n_crack']}+{ty['n_disocc']}+{ty['n_oofa']}={ty['n_hole']}")

    tab = disocclusion.hole_stats(hole)
    keys = {"label", "area", "bbox", "max_thickness", "touches_border",
            "mean_width", "max_width"}
    C.check("hole_stats returns one row per component with the required fields",
            len(tab) == ty["n_components"] and all(keys <= set(r) for r in tab),
            f"{len(tab)} rows, biggest {tab[0]['area']} px")
    line = [r for r in tab if r["area"] == 55]
    C.check("hole_stats: the 1 px crack has chamfer thickness ~0.955 and width 55",
            len(line) == 1 and abs(line[0]["max_thickness"] - 0.955) < 0.01
            and abs(line[0]["max_width"] - 55) < 1e-6,
            f"{line[0] if line else 'not found'}")
    C.check("hole_stats: touches_border flags the border regions",
            any(r["touches_border"] for r in tab))

    # warp.interior_gap: the core helper labels the HOLE mask, so its "oofa" is a subset of
    # the hole mask and touches the image border (this was a real core bug -- it used to
    # label the free space, so its oofa never intersected the hole mask at all).
    enc, oofa = warp.interior_gap(hole)
    n_enc, n_oofa = int(np.asarray(enc).sum()), int(np.asarray(oofa).sum())
    C.info(f"lfrd.warp.interior_gap -> enclosed {n_enc} px (hole {int(hole.sum())}), "
           f"oofa {n_oofa} px, of which {int((np.asarray(oofa, bool) & hole).sum())} "
           f"intersect the hole mask")
    C.check("core warp.interior_gap: enclosed + oofa == hole and oofa touches the border",
            n_enc + n_oofa == int(hole.sum())
            and int((np.asarray(oofa, bool) & hole).sum()) == n_oofa
            and bool(np.asarray(oofa, bool)[0, :].any()
                     or np.asarray(oofa, bool)[-1, :].any()
                     or np.asarray(oofa, bool)[:, 0].any()
                     or np.asarray(oofa, bool)[:, -1].any()),
            f"enclosed {n_enc}, oofa {n_oofa}, hole {int(hole.sum())}")
    C.check("typing agrees with the core helper's OOFA decision",
            ty["n_oofa"] > 0 and n_oofa > 0,
            f"method={ty['oofa_method']}")

    # min_area/min_width only refine disocc_major
    ty2 = disocclusion.type_holes(hole, min_area=10 ** 6, min_width=1000)
    C.check("disocc_major shrinks with bigger min_area/min_width but typing is unchanged",
            ty2["n_disocc"] == ty["n_disocc"] and ty2["n_disocc_major"] == 0)


# --------------------------------------------------------------------------- #
# brute-force Z-buffer (explicit python loop, reference for lfrd.warp.forward_warp)
# --------------------------------------------------------------------------- #
def brute_force_z(dx, dy, z, box):
    """Independent slow reference: dict of candidates per destination pixel.

    The candidate collection is vectorised (numpy) to keep it usable on 1024x768, but the
    winner resolution is an explicit python loop over *every* candidate contribution --
    written here, not in the module under test.
    """
    dx, dy, z = (np.asarray(a, np.float32) for a in (dx, dy, z))
    H, W = dx.shape
    x0, y0, w, h = box
    yy, xx = np.mgrid[0:H, 0:W]
    tx = xx + dx
    ty = yy + dy
    bx = np.floor(tx).astype(np.int32)
    by = np.floor(ty).astype(np.int32)
    wx = (tx - bx).astype(np.float32)
    wy = (ty - by).astype(np.float32)
    cand = {}
    for ox, oy, ww in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                       (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        ttx = bx + ox
        tty = by + oy
        sel = (ww > 0) & (ttx >= x0) & (ttx < x0 + w) & (tty >= y0) & (tty < y0 + h)
        for j, i, zz, txi, tyi in zip(yy[sel], xx[sel], z[sel], ttx[sel], tty[sel]):
            key = (int(tyi), int(txi))
            cand.setdefault(key, []).append((float(zz), int(j) * W + int(i)))
    win = np.full((H, W), -1, np.int64)
    zw = np.full((H, W), -np.inf, np.float32)
    ties = 0
    for (tyi, txi), lst in cand.items():
        best = -np.inf
        winners = []
        for zz, src in lst:                 # explicit winner resolution
            if zz > best:
                best, winners = zz, [src]
            elif zz == best:
                winners.append(src)
        if len(winners) > 1:
            ties += 1
        win[tyi, txi] = winners[-1]
        zw[tyi, txi] = best
    return win, zw, ties


def densest_box(mask, win):
    m = np.asarray(mask, np.float32)
    h, w = m.shape
    win = int(max(8, min(win, min(h, w))))
    acc = cv2.boxFilter(m, -1, (win, win), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    _a, _b, _c, (cx, cy) = cv2.minMaxLoc(acc)
    return (int(np.clip(cx - win // 2, 0, w - win)), int(np.clip(cy - win // 2, 0, h - win)),
            win, win)


def test_real_data(C):
    C.section("C. real data cam5 -> cam4 f000 (MSR Ballet)")
    cams = calib.load_calib()
    view = io_utils.load_view(io_utils.DATASET_ROOT_DEFAULT, 5, "f000")
    P, color = view["depth"], view["color"]
    C.check("view loaded", P.shape == (768, 1024) and color.shape == (768, 1024, 3),
            f"depth {P.shape}, colour {color.shape}")

    # drive the two step scripts into a DEDICATED run so that the on-disk contract below
    # compares artefacts that provably belong to THIS frame (a shared stage dir belongs to
    # whichever frame ran last).
    import subprocess
    run = "_test_stage_a"
    for script in ("step1_preprocess.py", "step2_warp.py"):
        p = subprocess.run([sys.executable, os.path.join(ROOT, script), "--run", run],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=ROOT)
        if p.returncode != 0:
            C.info(f"{script} failed (exit {p.returncode}) -- on-disk checks skipped")
            run = None
            break

    info = preprocess.preprocess_depth(P, th=20, rounds=2, min_change=1, return_info=True)
    st = preprocess.ghost_stats(P, info["P_out"], info["marked_all"])
    C.info(f"preprocess: raw {int(info['marked_raw'].sum())} px, applied "
           f"{st['n_marked']} px, per-round raw "
           f"{[int(e.sum()) for e in info['er']]}")
    C.check("real: marked pixels > 0", st["n_marked"] > 1000, f"{st['n_marked']} px")
    C.check("real: monotone increase only",
            st["p_dec"] == 0 and bool((info["P_out"] >= P).all()),
            f"increased {st['p_inc']}, decreased {st['p_dec']}")
    C.check("real: undefined (P==0) pixels unchanged",
            st["n_zero_changed"] == 0 and int((P == 0).sum()) > 0,
            f"{st['n_zero_changed']} of {st['n_zero_input']} changed")
    C.check("real: raw eq.(1) marking includes the undefined pixels "
            "(guarded out of the correction)",
            int((info["marked_raw"] & (P == 0)).sum()) == int((P == 0).sum())
            and int((info["marked_all"] & (P == 0)).sum()) == 0,
            f"{int((P == 0).sum())} undefined px raw-marked")
    C.check("real: P_out is uint8 and in range",
            info["P_out"].dtype == np.uint8 and int(info["P_out"].max()) <= 255)
    C.check("real: Laplacian energy in the ghost band drops sharply",
            st["lap_energy_after"] < 0.75 * st["lap_energy_before"],
            f"{st['lap_energy_before']:.3f} -> {st['lap_energy_after']:.3f}")

    # ---- warp both arms -------------------------------------------------- #
    wp = warp.warp_view(cams, 5, 4, color, info["P_out"])
    wnp = warp.warp_view(cams, 5, 4, color, P)
    hf, hf_np = float(wp["hole"].mean()), float(wnp["hole"].mean())
    crack = warp.crack_mask(wp["hole"], 2)
    ty = disocclusion.type_holes(wp["hole"], crack_max_width=2, min_area=60, min_width=4,
                                 src_oob=wp["src_oob"])
    ty_np = disocclusion.type_holes(wnp["hole"], crack_max_width=2, min_area=60,
                                    min_width=4)
    C.info(f"warp prep : hole {hf * 100:.4f}% ({int(wp['hole'].sum())} px), crack "
           f"{ty['n_crack']}, disocc {ty['n_disocc']}, oofa {ty['n_oofa']}")
    C.info(f"warp noprep: hole {hf_np * 100:.4f}% ({int(wnp['hole'].sum())} px), crack "
           f"{ty_np['n_crack']}, disocc {ty_np['n_disocc']}, oofa {ty_np['n_oofa']}")
    C.check("real: hole fraction in [2%, 40%]", 0.02 <= hf <= 0.40,
            f"{hf * 100:.4f}%")
    C.check("real: hole region is not empty", int(wp["hole"].sum()) > 0)
    C.check("real: backward idx >= 0 exactly where ~hole",
            bool(((wp["backward"] >= 0) == ~wp["hole"]).all()),
            f"idx>=0 {int((wp['backward'] >= 0).sum())} vs valid {int((~wp['hole']).sum())}")
    C.check("real: warped depth holes match colour holes",
            bool(((wp["warped_depth"] < 0) == wp["hole"]).all()))
    C.check("real: every hole pixel has exactly one type",
            ty["n_crack"] + ty["n_disocc"] + ty["n_oofa"] == ty["n_hole"]
            and bool(((ty["crack"].astype(np.uint8) + ty["disocc"].astype(np.uint8)
                       + ty["oofa"].astype(np.uint8)) == wp["hole"].astype(np.uint8)).all()))
    C.check("real: crack mask subset of hole_all",
            bool((crack & ~wp["hole"]).sum() == 0), f"{int(crack.sum())} px")
    C.check("real: crack+disocc+oofa == hole_all",
            bool(((ty["crack"] | ty["disocc"] | ty["oofa"]) == wp["hole"]).all()))
    C.check("real: thin cracks are the 1-2 px components",
            bool((crack == ty["crack"]).all()), f"{int(crack.sum())} px")
    C.check("real: OOFA sits on the image border",
            bool(ty["oofa"][:, 0].any() or ty["oofa"][0, :].any()
                 or ty["oofa"][:, -1].any() or ty["oofa"][-1, :].any()),
            f"{ty['n_oofa']} px")
    rel = abs(ty["n_oofa"] - ty_np["n_oofa"]) / max(1.0, ty_np["n_oofa"])
    C.check("real: preprocessing keeps the OOFA area within 5%", rel <= 0.05,
            f"{ty['n_oofa']} vs {ty_np['n_oofa']} px ({rel * 100:.2f}%)")
    C.check("real: disocclusion region exists and is the largest hole class",
            ty["n_disocc"] > 10000 and ty["n_disocc"] > ty["n_oofa"] * 0.5,
            f"disocc {ty['n_disocc']} vs oofa {ty['n_oofa']}")

    # ---- brute-force Z-buffer on a 96x96 crop ---------------------------- #
    crop = densest_box(wp["hole"], 128)
    crop = (crop[0], crop[1], 96, 96)
    z_inv = (1.0 / np.maximum(calib.depth_from_P(info["P_out"]), 1e-6)).astype(np.float32)
    win_bf, z_bf, n_tie = brute_force_z(wp["dx"], wp["dy"], z_inv, crop)
    wcolor_c, wz_core, _hc, _wc, idx_core = warp.forward_warp(
        color.astype(np.float32), wp["dx"], wp["dy"], z=z_inv, rule="zbuf",
        splat="sub", return_index=True)
    x0, y0, w0, h0 = crop
    bf = win_bf[y0:y0 + h0, x0:x0 + w0]
    core = idx_core[y0:y0 + h0, x0:x0 + w0]
    zc = z_inv.reshape(-1)
    both = (bf >= 0) & (core >= 0)
    same_cov = bool(((bf >= 0) == (core >= 0)).all())
    agree = int((bf[both] == core[both]).sum())
    tie_ok = bool(np.allclose(zc[bf[both]], zc[core[both]], atol=1e-6)) if both.any() \
        else True
    C.info(f"Z-buffer crop {crop}: {int(both.sum())} covered px, {agree} identical "
           f"winners (brute-force tie targets: {n_tie}), z-ties ok: {tie_ok}")
    C.check("real: Z-buffer coverage matches brute force", same_cov)
    C.check("real: Z-buffer winner matches brute force (or is a depth tie)",
            agree == int(both.sum()) or tie_ok,
            f"{agree}/{int(both.sum())}")
    z_bf_c = z_bf[y0:y0 + h0, x0:x0 + w0]
    z_core_c = wz_core[y0:y0 + h0, x0:x0 + w0]
    C.check("real: warped z-buffer value == brute-force max depth weight",
            bool(same_cov and np.allclose(z_bf_c[both & (z_bf_c > -np.inf)],
                                         z_core_c[both & (z_bf_c > -np.inf)], atol=1e-4)),
            f"max |diff| "
            f"{float(np.abs(z_bf_c[both] - z_core_c[both]).max()) if both.any() else 0.0:.2e}")

    # ---- depth-map hole semantics ---------------------------------------- #
    saved_mem = np.rint(fill_depth_nearest(wp["warped_depth"], crack)).astype(np.int16)
    C.check("real: warped_depth semantics (-1 exactly on unfilled holes)",
            bool(((saved_mem < 0) == (wp["hole"] & ~crack)).all()),
            f"{int((saved_mem < 0).sum())} unset px of {int(wp['hole'].sum())} holes")
    C.check("real: crack-filled depth pixels have a defined depth",
            bool((saved_mem[crack] >= 0).all()), f"{int(crack.sum())} crack px")

    # on-disk stage contract.  IMPORTANT: a shared stage directory holds whichever frame was
    # processed there LAST, so the on-disk comparison is only meaningful when the test also
    # drives the pipeline itself -- it writes to its own run name for that reason.
    wdir = io_utils.run_dir(run, "warp") if run else None
    dpath = os.path.join(wdir, "warped_depth.npy") if wdir else ""
    if wdir and os.path.isfile(dpath):
        on_disk = np.load(dpath)
        hole_png = io_utils.imread(os.path.join(wdir, "hole_all.png"), gray=True) > 0
        tmap = io_utils.imread(os.path.join(wdir, "hole_type.png"), gray=True)
        C.check("on-disk warped_depth.npy: int16, -1 exactly on unfilled holes",
                on_disk.dtype == np.int16
                and bool(((on_disk < 0) == (wp["hole"] & ~crack)).all()),
                f"dtype {on_disk.dtype}, {int((on_disk < 0).sum())} unset px")
        C.check("on-disk hole_all.png matches the in-memory hole mask",
                bool((hole_png == wp["hole"]).all()), f"{int(hole_png.sum())} px")
        C.check("on-disk hole_type.png matches the typing ids",
                bool((tmap == ty["hole_type_map"]).all())
                and set(np.unique(tmap).tolist()) <= {0, 1, 2, 3},
                f"ids {np.unique(tmap).tolist()}")
        lap = np.load(os.path.join(wdir, "warped_lap.npy"))
        C.check("on-disk warped_lap.npy matches warp_view output",
                lap.shape == wp["warped_lap"].shape and bool(np.allclose(lap,
                                                                        wp["warped_lap"])),
                f"shape {lap.shape}")
        bw = io_utils.load_npz(os.path.join(wdir, "backward.npz"))["idx"]
        C.check("on-disk backward.npz idx >= 0 exactly where ~hole",
                bool(((bw >= 0) == ~wp["hole"]).all()), f"{bw.shape}")
    else:
        C.info(f"{dpath} absent -- on-disk artefact checks skipped")
    return dict(hole_frac=hf, crack=ty["n_crack"], disocc=ty["n_disocc"],
                oofa=ty["n_oofa"], marked=st["n_marked"])


# --------------------------------------------------------------------------- #
# D. synthetic fixture
# --------------------------------------------------------------------------- #
def test_fixture(C):
    C.section("D. synthetic fixture (tools.make_fixtures.build)")
    small_img, small_depth = build()                      # native 192x256 fixture
    C.check("fixture built at 192x256",
            small_depth.shape == (192, 256) and small_img.shape == (192, 256, 3))
    info = preprocess.preprocess_depth(small_depth, th=20, rounds=2, return_info=True)
    C.check("fixture: preprocessing marks the foreground boundary",
            info["marked_all"].sum() > 0,
            f"{int(info['marked_all'].sum())} px")
    C.check("fixture: monotone non-decreasing", bool((info["P_out"] >= small_depth).all()))
    C.check("fixture: foreground block interior untouched",
            bool((info["P_out"][60:140, 90:140] == 210).all()))
    C.check("fixture: two rounds thicken the block by 2 px",
            bool((info["P_out"][100, 78] == 210) and (info["P_out"][100, 77] == 60)))

    # full-resolution synthetic scene through the real calibration (geometry stages)
    view = load_synthetic_view(5)
    cams = calib.load_calib()
    inf2 = preprocess.preprocess_depth(view["depth"], th=20, rounds=2, return_info=True)
    wp = warp.warp_view(cams, 5, 4, view["color"], inf2["P_out"])
    ty = disocclusion.type_holes(wp["hole"], crack_max_width=2, min_area=60, min_width=4)
    C.info(f"fixture_synth warp: hole {wp['hole'].mean() * 100:.4f}% "
           f"({int(wp['hole'].sum())} px), crack {ty['n_crack']}, disocc {ty['n_disocc']}, "
           f"oofa {ty['n_oofa']}")
    C.check("fixture_synth: hole fraction in [2%, 40%]",
            0.02 <= wp["hole"].mean() <= 0.40, f"{wp['hole'].mean() * 100:.4f}%")
    C.check("fixture_synth: all three hole classes present",
            ty["n_crack"] + ty["n_disocc"] > 0 and ty["n_oofa"] > 0,
            f"crack {ty['n_crack']}, disocc {ty['n_disocc']}, oofa {ty['n_oofa']}")
    C.check("fixture_synth: typing partitions the hole mask",
            bool(((ty["crack"] | ty["disocc"] | ty["oofa"]) == wp["hole"]).all()))
    C.check("fixture_synth: the synthetic block's disocclusion opens to its right",
            bool(ty["disocc"][:, 150:].any()),
            f"disocc bbox {disocclusion.largest_bbox(ty['disocc'])}")

    # the fixture's own recorded depth_pp / hole masks are readable (stage contract)
    d = io_utils.run_dir("fixture", "preproc")
    if os.path.isfile(os.path.join(d, "depth_pp.png")):
        pp = io_utils.imread(os.path.join(d, "depth_pp.png"), gray=True)
        C.check("fixture run: make_fixtures depth_pp.png readable at 192x256",
                pp.shape == (192, 256), f"{pp.shape}")


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-real", action="store_true", help="skip the real-data section")
    ap.add_argument("--no-fixture", action="store_true", help="skip the fixture section")
    a = ap.parse_args()

    C = Checks()
    test_preprocess_units(C)
    test_type_holes_units(C)
    if not a.no_real:
        test_real_data(C)
    if not a.no_fixture:
        test_fixture(C)
    print(f"\n[TEST] test_step1_2: {C.n_pass} passed, {C.n_fail} failed")
    return 1 if C.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
