"""Stage C tests -- modified Criminisi inpainting (paper III-D) and render (paper III-E).

Plain python asserts (no pytest), runnable with the bundled interpreter::

    & $py tests/test_step5_6.py            # everything that can run in this workspace
    & $py tests/test_step5_6.py --quick    # skip the 1024x768 speed run and the CLI runs

Two fixtures are used:

* an *in-memory* synthetic scene (foreground block + textured background + a hole on the
  right of the block) that reproduces exactly the failure mode the paper's depth term is
  meant to prevent -- foreground texture leaking into the removed region;
* ``output/fixture_c/`` -- a full-size (1024x768) synthetic stand-in for stages 2 and 4:
  the real camera-5 frame, warped to camera 4 with ``lfrd.warp.warp_view`` (that part is
  stage A/B functionality, used here only to build a fixture), plus a removed-region mask
  and a predicted depth built with the eq. (4) "background side" rule.  This exercises the
  real ``step5_inpaint.py`` and ``step6_render.py`` command lines end to end.

If ``output/ba54_f000/40_removal/`` and ``output/ba54_f000/20_warp/`` exist (stage A/B
finished), the last test also runs both CLIs on the real data.
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
cv2.setNumThreads(1)

from lfrd import calib, inpaint as I, io_utils, render, warp

PY = sys.executable
NAMES = []
FAILS = []


def check(name, ok, detail=""):
    ok = bool(ok)
    NAMES.append(name)
    if not ok:
        FAILS.append(f"{name}  ({detail})")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def synth_scene(seed=11, H=240, W=320):
    """Foreground block on the left, textured background, hole on the block's right."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = np.clip(60 + xx * 0.5, 0, 255)
    bg[..., 1] = np.clip(50 + yy * 0.6, 0, 255)
    bg[..., 2] = 90
    for _ in range(160):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(bg, (x, y), int(rng.integers(2, 7)),
                   tuple(int(v) for v in rng.integers(60, 190, 3)), -1)
    fg = np.zeros((H, W), bool)
    fg[70:190, 60:140] = True
    img = bg.copy()
    img[fg] = np.array([25, 30, 210], np.uint8)
    depth = np.full((H, W), 60, np.uint8)
    depth[fg] = 210
    hole = np.zeros((H, W), bool)
    hole[80:180, 120:150] = True          # strip over/right of the block
    hole[40, 20:300] = True               # thin crack
    depth_pred = depth.copy()
    depth_pred[hole] = 60                 # stage 5.1 predicts the background layer
    colour_h = img.copy()
    colour_h[hole] = 0
    return dict(bg=bg, img=img, depth=depth, hole=hole, depth_pred=depth_pred,
                colour=colour_h, fg=fg)


def bg_continuation(img, hole):
    """Nearest valid pixel to the right in the same row (the background side)."""
    H, W = hole.shape
    idx = np.where(~hole, np.arange(W)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    return img[np.arange(H)[:, None], np.clip(idx, 0, W - 1)]


# --------------------------------------------------------------------------- #
# 1. the vectorised masked SSD equals the brute-force masked SSD
# --------------------------------------------------------------------------- #
def test_ssd_matches_bruteforce():
    section("1. vectorised masked SSD == brute force (eq. 9)")
    rng = np.random.default_rng(3)
    H, W = 40, 48
    img = rng.integers(0, 255, (H, W, 3)).astype(np.uint8)
    hole = np.zeros((H, W), bool)
    hole[12:28, 16:34] = True
    ps, r = 9, 4
    Hp, Wp = H + 2 * r, W + 2 * r
    imgp = np.zeros((Hp, Wp, 3), np.float32)
    imgp[r:r + H, r:r + W] = img
    sq_p = (imgp ** 2).sum(-1)
    known = np.ones((Hp, Wp), bool)
    known[r:r + H, r:r + W] = ~hole
    py, px = 12 + r, 20 + r
    win = (slice(py - r, py + r + 1), slice(px - r, px + r + 1))
    M = known[win]
    maskf = M.astype(np.float32)
    tpm = np.where(M[:, :, None], imgp[win], 0.0)
    A = float((tpm * imgp[win]).sum())
    bq = np.zeros((H, W), np.float32)
    for c in range(3):
        bq += cv2.filter2D(imgp[:, :, c], cv2.CV_32F, tpm[:, :, c])[r:H + r, r:W + r]
    cq = cv2.filter2D(sq_p, cv2.CV_32F, maskf)[r:H + r, r:W + r]
    ssd_vec = A - 2 * bq + cq
    tp = imgp[win].astype(np.float64)
    ssd_bf = np.zeros((H, W), np.float64)
    for yy in range(H):
        for xx in range(W):
            q = imgp[yy:yy + ps, xx:xx + ps].astype(np.float64)
            ssd_bf[yy, xx] = (M[:, :, None] * (tp - q) ** 2).sum()
    err = float(np.abs(ssd_vec - ssd_bf).max())
    rel = err / max(float(ssd_bf.max()), 1.0)
    check("masked SSD correlation == brute force (rel err < 1e-4)", rel < 1e-4,
          f"max abs err {err:.4f}, rel {rel:.2e}")


# --------------------------------------------------------------------------- #
# 2. inpaint contract: only holes written, everything filled
# --------------------------------------------------------------------------- #
def test_inpaint_contract():
    section("2. inpaint contract (writes only holes, fills everything, eq. 8 priority)")
    s = synth_scene()
    fg_edge = I.laplacian_fg(s["depth_pred"], dilate=1)[0]
    m = I.inpaint(s["colour"], s["hole"], s["depth_pred"], fg_mask=fg_edge,
                  return_meta=True, record_source=True)
    check("hole pixels: all filled", m["n_unfilled"] == 0 and m["fully_filled"],
          f"filled {m['n_filled']}/{m['n_hole']}, unfilled {m['n_unfilled']}")
    check("no pixel outside the hole changed",
          bool(np.array_equal(m["filled"][~s["hole"]], s["colour"][~s["hole"]])),
          f"{int((~s['hole']).sum())} px compared")
    check("source map defined exactly on the hole",
          bool(((m["src_map"] >= 0) == s["hole"]).all()),
          f"src_map>=0 outside hole: {int((m['src_map'][~s['hole']] >= 0).sum())}")
    check("FG front pixels have priority 0 (eq. 8)",
          m["fg_front_at_first"] > 0 and m["fg_front_priority_max_first"] == 0.0,
          f"{m['fg_front_at_first']} FG front px, max P0 "
          f"{m['fg_front_priority_max_first']:.6f}")
    check("matching used only the strict rung of the ladder",
          set(m["branch_hist"]) <= {0, 1}, f"{m['branch_hist']}")
    d = np.asarray(m["filled_depth"], np.float32)
    check("filled depth written (not left at 0)",
          bool((d[s["hole"]] > 0).all()), f"min filled depth {d[s['hole']].min():.0f}")
    return m


# --------------------------------------------------------------------------- #
# 3. no foreground sampling + the ablation switches change behaviour
# --------------------------------------------------------------------------- #
def test_no_fg_sampling_and_ablations():
    section("3. no foreground sampling; eq. (7)/(8)/(9)-(10) ablations")
    s = synth_scene()
    fg_edge = I.laplacian_fg(s["depth_pred"], dilate=1)[0]
    fg_region, thr = I.layer_fg_mask(s["depth"], valid=s["depth"] > 0)
    bg_ref = bg_continuation(s["bg"], s["hole"])
    H, W = s["hole"].shape

    def run(**over):
        kw = dict(fg_mask=fg_edge, return_meta=True, record_source=True)
        kw.update(over)
        r = I.inpaint(s["colour"], s["hole"], s["depth_pred"], **kw)
        src = r["src_map"][s["hole"]]
        sy, sx = np.divmod(src, W)
        leak = float(fg_region[sy, sx].mean())
        dev = float(np.abs(np.asarray(r["filled"], np.float32)[s["hole"]]
                           - bg_ref[s["hole"]].astype(np.float32)).mean())
        diff = float((np.asarray(r["filled"]) != np.asarray(_ours["filled"])
                      ).any(axis=2)[s["hole"]].mean())
        return r, leak, dev, diff

    _ours = None
    r0 = I.inpaint(s["colour"], s["hole"], s["depth_pred"], fg_mask=fg_edge,
                   return_meta=True, record_source=True)
    _ours = r0
    src = r0["src_map"][s["hole"]]
    sy, sx = np.divmod(src, W)
    leak = float(fg_region[sy, sx].mean())
    dev = float(np.abs(np.asarray(r0["filled"], np.float32)[s["hole"]]
                       - bg_ref[s["hole"]].astype(np.float32)).mean())
    check("foreground sampling fraction ~ 0 (ours)", leak < 0.005,
          f"leak = {leak:.5f} (FG layer threshold {thr})")
    no_bg, leak_b, dev_b, diff_b = run(use_bg_term=False)
    check("ablation B(p)=1 (eq. 8) changes the result", diff_b > 1e-4,
          f"{diff_b:.2%} of the removed px differ")
    check("ablation B(p)=1 is worse in background similarity",
          dev_b > dev, f"bg dev {dev_b:.2f} vs ours {dev:.2f} (lower better)")
    check("ablation B(p)=1 has a worse exemplar match cost",
          no_bg["match_cost_mean"] > r0["match_cost_mean"],
          f"{no_bg['match_cost_mean']:.1f} vs {r0['match_cost_mean']:.1f}")
    no_dl, leak_dl, dev_dl, diff_dl = run(use_depth_limit=False)
    check("ablation DD off (eq. 9-10) lets foreground be sampled",
          leak_dl > leak + 0.05,
          f"leak {leak_dl:.4f} vs ours {leak:.5f}; bg dev {dev_dl:.2f} vs {dev:.2f}")
    no_dt, leak_dt, dev_dt, diff_dt = run(use_depth_term=False)
    dp = float(np.abs(no_dt["priority_at_first"] - r0["priority_at_first"]).max())
    check("ablation Z(p) off (eq. 7) changes the priority field or the result",
          dp > 0.0 or diff_dt > 1e-4,
          f"max |dP0| = {dp:.5f}; resulting image diff {diff_dt:.2%}")
    lit, leak_lit, dev_lit, diff_lit = run(layer_ref="valid")
    check("literal eq. (9) 'valid part' reference increases foreground sampling "
          "(justifies the documented deviation)",
          leak_lit > leak + 1e-3, f"leak {leak_lit:.4f} vs ours {leak:.5f}")
    return dict(leak=leak, dev=dev)


# --------------------------------------------------------------------------- #
# 4. render.postprocess semantics (OOFA untouched, holes filled)
# --------------------------------------------------------------------------- #
def test_render_postprocess():
    section("4. lfrd.render.postprocess: fills holes, leaves OOFA")
    s = synth_scene(seed=5)
    hole = s["hole"].copy()
    oofa = np.zeros_like(hole)
    oofa[:, 0:12] = hole[:, 0:12]
    hole2 = hole | oofa
    out, outd, st = render.postprocess(s["colour"], hole2, s["depth_pred"], oofa=oofa,
                                       fill_oofa=False)
    changed = (np.asarray(out) != np.asarray(s["colour"])).any(axis=2)
    check("postprocess fills every non-OOFA hole",
          bool(changed[hole2 & ~oofa].all()),
          f"{int((hole2 & ~oofa).sum())} px, unfilled {st.get('unfilled')}")
    check("postprocess leaves OOFA untouched",
          bool(np.array_equal(np.asarray(out)[oofa], np.asarray(s["colour"])[oofa])),
          f"{int(oofa.sum())} OOFA px")
    check("postprocess does not touch valid pixels",
          bool(np.array_equal(np.asarray(out)[~hole2], np.asarray(s["colour"])[~hole2])),
          "")
    out2, _, st2 = render.postprocess(s["colour"], hole2, s["depth_pred"], oofa=oofa,
                                      fill_oofa=True)
    check("fill_oofa=True fills OOFA as well",
          bool((( np.asarray(out2) != np.asarray(s["colour"])).any(axis=2)[oofa]).all()),
          f"{st2['filled_px']} px filled")


# --------------------------------------------------------------------------- #
# 5. speed on a full-size frame with a large hole area
# --------------------------------------------------------------------------- #
def test_speed_full_frame(max_seconds=600.0):
    section("5. speed: full 1024x768 frame, large hole area, paper search window")
    rng = np.random.default_rng(11)
    H, W = 768, 1024
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = np.clip(50 + xx * 0.15, 0, 255)
    bg[..., 1] = np.clip(60 + yy * 0.2, 0, 255)
    bg[..., 2] = 100
    for _ in range(3000):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(bg, (x, y), int(rng.integers(2, 9)),
                   tuple(int(v) for v in rng.integers(50, 200, 3)), -1)
    fg = np.zeros((H, W), bool)
    for cx, cy, w, h in ((300, 420, 90, 300), (600, 380, 70, 260), (800, 450, 60, 200)):
        fg[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = True
    img = bg.copy()
    img[fg] = np.array([30, 25, 200], np.uint8)
    depth = np.full((H, W), 99, np.uint8)
    depth[fg] = 211
    hole = np.zeros((H, W), bool)
    for cx, cy, w, h in ((345, 420, 40, 300), (635, 380, 36, 260), (830, 450, 26, 200)):
        hole[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = True
    for y in range(20, H, 37):
        hole[y, 40:1000] = True
    depth_pred = depth.copy()
    depth_pred[hole] = 99
    colour = img.copy()
    colour[hole] = 0
    fg_edge = I.laplacian_fg(depth_pred, dilate=1)[0]
    t0 = time.perf_counter()
    m = I.inpaint(colour, hole, depth_pred, fg_mask=fg_edge, return_meta=True)
    dt = time.perf_counter() - t0
    print(f"    hole {m['n_hole']} px, {m['n_iters']} iterations, {dt:.1f}s, "
          f"{1000 * dt / max(m['n_iters'], 1):.2f} ms/iter, "
          f"{m['n_hole'] / max(dt, 1e-9):.0f} px/s")
    fg_region, _ = I.layer_fg_mask(depth)
    src = m["src_map"]
    check("large-hole frame fully filled", m["n_unfilled"] == 0,
          f"{m['n_filled']}/{m['n_hole']} px")
    check(f"speed within budget (< {max_seconds:.0f}s)", dt < max_seconds,
          f"{dt:.1f}s, {1000 * dt / max(m['n_iters'], 1):.2f} ms/iter")
    return dt


# --------------------------------------------------------------------------- #
# 6. build the full-size fixture and run step5 + step6 through their CLIs
# --------------------------------------------------------------------------- #
def _step5_module():
    """Import step5_inpaint.py as a module (for its eq. (4)-(5) helper)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("step5_inpaint_mod",
                                                  os.path.join(ROOT, "step5_inpaint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_fixture(run="fixture_c"):
    """Synthetic stage-2 / stage-4 artefacts at the real resolution.

    Only this *test fixture* reproduces the forward warp; the deliverable stages are
    exercised through their command lines exactly as the user runs them.  The removed
    region is a band along a real foreground component of the reference frame and its
    predicted depth follows the eq. (4) "take the background-side value" rule, so the
    fixture is geometrically meaningful (a pure synthetic rectangle would not be).
    """
    root = io_utils.DATASET_ROOT_DEFAULT
    cams = calib.load_calib(root)
    ref = io_utils.load_view(root, 5, "f000")
    colour, depth = ref["color"], ref["depth"]
    H, W = depth.shape

    # ---- stage 2 stand-in: plain warp cam5 -> cam4 (Z-buffer, sub-pixel) ---- #
    res = warp.warp_view(cams, 5, 4, colour, depth)
    hole = res["hole"]
    crack = warp.crack_mask(hole, 2)
    enclosed, oofa = warp.interior_gap(hole, dilate=1)
    disocc = enclosed & ~crack
    d20 = io_utils.run_dir(run, "warp")
    io_utils.imwrite(os.path.join(d20, "warped_color.png"), res["warped_color"])
    io_utils.imwrite(os.path.join(d20, "warped_depth.png"),
                     np.clip(np.rint(res["warped_depth"]), 0, 255).astype(np.uint8))
    for name, mm in (("hole_all", hole), ("hole_crack", crack),
                     ("hole_disocc", disocc), ("hole_oofa", oofa)):
        io_utils.imwrite(os.path.join(d20, f"{name}.png"), (mm * 255).astype(np.uint8))

    # ---- stage 4 stand-in: a strip of a real foreground component ---- #
    nz = depth[depth > 0]
    person = (depth >= np.percentile(nz, 92))
    n, lab, st, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), 8)
    if n > 1:
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        x0, y0, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                        st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        rem = np.zeros((H, W), bool)
        ww = max(8, int(0.35 * w))
        rem[y0:y0 + h, x0:x0 + ww] = (lab[y0:y0 + h, x0:x0 + ww] == i)
    else:
        rem = np.zeros((H, W), bool)
        rem[250:560, 470:530] = True
    rem[400, 100:900] = True                      # a thin crack-like removal
    rem_color = colour.copy()
    rem_color[rem] = 0
    rem_depth = depth.copy()
    rem_depth[rem] = 0
    # eq. (4) "the removed region belongs to the background": nearest same-row pixel of
    # the visible background layer (same helper step5 uses for the auto fallback)
    depth_pred, bg_level, _ = _step5_module()._background_layer(depth.astype(np.float32),
                                                                rem)
    depth_pred = np.clip(np.rint(depth_pred), 0, 255).astype(np.uint8)
    d40 = io_utils.run_dir(run, "removal")
    io_utils.imwrite(os.path.join(d40, "removed_mask.png"), (rem * 255).astype(np.uint8))
    io_utils.imwrite(os.path.join(d40, "removed_color.png"), rem_color)
    io_utils.imwrite(os.path.join(d40, "removed_depth.png"), rem_depth)
    io_utils.imwrite(os.path.join(d40, "depth_pred.png"), depth_pred)
    ul = np.where(rem.any(1), rem.argmax(1), -1).astype(np.int32)
    ur = np.where(rem.any(1), W - 1 - rem[:, ::-1].argmax(1), -1).astype(np.int32)
    fg = np.zeros((H, W), np.uint8)
    fg[rem] = 255
    io_utils.save_npz(os.path.join(d40, "removal_meta.npz"), u_l=ul, u_r=ur,
                      mask=(rem * 255).astype(np.uint8), case=np.zeros((H,), np.int8),
                      fg_mask=fg, depth_pred=depth_pred,
                      ref_depth=depth.astype(np.uint8), bg_level=np.float32(bg_level))
    print(f"    fixture: warp holes {int(hole.sum())} px (disocc {int(disocc.sum())}, "
          f"crack {int(crack.sum())}, oofa {int(oofa.sum())}); removed {int(rem.sum())} px, "
          f"predicted background level P={bg_level:.0f}, "
          f"pred median {np.median(depth_pred[rem]) if rem.any() else 0:.0f}, "
          f"removed foreground level {np.median(depth[rem]) if rem.any() else 0:.0f}")
    return dict(hole=hole, disocc=disocc, oofa=oofa, rem=rem)


def run_cli(script, args, timeout=2400):
    cmd = [PY, os.path.join(ROOT, script)] + args
    print("    $ " + " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        rc = -9
        out = ((exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes)
               else (exc.stdout or ""))
    dt = time.perf_counter() - t0
    for line in out.splitlines():
        if "[CHECK]" in line or "inpaint (eq" in line or "III-E" in line:
            print("      " + line.strip())
    print(f"    exit={rc}  wall={dt:.1f}s")
    return rc, out, dt


def _parse_summary(out, tag):
    for line in out.splitlines():
        if line.startswith(f"[CHECK] {tag}:") and "passed" in line:
            parts = line.split()
            try:                       # '[CHECK] step5: 10 passed, 0 failed (..)'
                return int(parts[2]), int(parts[4])
            except Exception:                                      # noqa: BLE001
                pass
    return None, None


def test_pipeline_fixture(run="fixture_c"):
    section("6. step5_inpaint.py + step6_render.py on the full-size fixture")
    fx = build_fixture(run)
    rc5, out5, dt5 = run_cli("step5_inpaint.py", ["--run", run])
    p5, f5 = _parse_summary(out5, "step5")
    check("step5 exits 0 on the fixture", rc5 == 0, f"exit {rc5}")
    check("step5 [CHECK] all pass", p5 is not None and f5 == 0,
          f"passed={p5} failed={f5}")
    d_fill = io_utils.run_dir(run, "fill")
    need = ["filled_occlusion.png", "filled_occlusion_depth.png", "depth_pred.png",
            "inpaint_meta.json", "inpaint_meta.npz", "panel_inpaint.png"]
    missing = [f for f in need if not os.path.isfile(os.path.join(d_fill, f))]
    check("step5 wrote all artefacts", not missing, f"missing {missing}")
    if missing:
        return None
    filled = io_utils.imread(os.path.join(d_fill, "filled_occlusion.png"))
    fmeta = io_utils.read_json(os.path.join(d_fill, "inpaint_meta.json"))
    rem_color = io_utils.imread(os.path.join(io_utils.run_dir(run, "removal"),
                                             "removed_color.png"))
    check("filled_occlusion: untouched pixels outside the removed region",
          bool(np.array_equal(filled[~fx["rem"]], rem_color[~fx["rem"]])), "")
    check("inpaint_meta.json records iters/seconds/pixels/rejections",
          all(k in fmeta for k in ("n_iters", "seconds", "filled_px", "removed_px",
                                   "candidates_rejected_no_depth",
                                   "candidates_rejected_none", "branch_hist")),
          f"n_iters={fmeta.get('n_iters')} seconds={fmeta.get('seconds', 0):.1f} "
          f"filled={fmeta.get('filled_px')}")
    check("inpaint speed budget (< 600 s)", fmeta["seconds"] < 600.0,
          f"{fmeta['seconds']:.1f}s for {fmeta['removed_px']} px, "
          f"{fmeta['ms_per_iteration']:.2f} ms/iter")
    check("fixture fill: no foreground sampling",
          fmeta["fg_source_fraction"] is None or fmeta["fg_source_fraction"] < 0.02,
          f"fg_source_fraction={fmeta['fg_source_fraction']}")

    rc6, out6, dt6 = run_cli("step6_render.py", ["--run", run])
    p6, f6 = _parse_summary(out6, "step6")
    check("step6 exits 0 on the fixture", rc6 == 0, f"exit {rc6}")
    check("step6 [CHECK] all pass", p6 is not None and f6 == 0, f"passed={p6} failed={f6}")
    d_fin = io_utils.run_dir(run, "final")
    need6 = ["final.png", "final_pre_post.png", "final_mask_overlay.png", "valid_mask.png",
             "final_meta.json", "panel_final.png"]
    missing6 = [f for f in need6 if not os.path.isfile(os.path.join(d_fin, f))]
    check("step6 wrote all artefacts", not missing6, f"missing {missing6}")
    if missing6:
        return fmeta
    final = io_utils.imread(os.path.join(d_fin, "final.png"))
    wcolor = io_utils.imread(os.path.join(io_utils.run_dir(run, "warp"), "warped_color.png"))
    fmeta6 = io_utils.read_json(os.path.join(d_fin, "final_meta.json"))
    check("final == plain warp where the warp was valid",
          bool(np.array_equal(final[~fx["hole"]], wcolor[~fx["hole"]])), "")
    check("OOFA untouched (--fill_oofa off)",
          bool(np.array_equal(final[fx["oofa"]], wcolor[fx["oofa"]]))
          if fx["oofa"].any() else True, f"{int(fx['oofa'].sum())} OOFA px")
    check("leftovers are (almost exactly) the OOFA pixels",
          abs(fmeta6["leftover_px"] - fmeta6["hole_oofa_px"])
          <= max(50, int(0.01 * fmeta6["hole_oofa_px"])),
          f"leftover {fmeta6['leftover_px']} px, OOFA {fmeta6['hole_oofa_px']} px")
    check("no valid pixel overwritten", fmeta6["checks"]["failed"] == 0,
          f"failed checks: {fmeta6['checks']['failed']}")
    return fmeta


# --------------------------------------------------------------------------- #
# 7. real data (only if stages A/B have finished)
# --------------------------------------------------------------------------- #
def test_real_data(run="ba54_f000"):
    section(f"7. real data run ({run}) -- needs stage A/B artefacts")
    d_rem = io_utils.run_dir(run, "removal", create=False)
    d_warp = io_utils.run_dir(run, "warp", create=False)
    need = [os.path.join(d_rem, "removed_mask.png"),
            os.path.join(d_rem, "removed_color.png"),
            os.path.join(d_warp, "warped_color.png"),
            os.path.join(d_warp, "hole_disocc.png")]
    have = [p for p in need if os.path.isfile(p)]
    if len(have) < len(need):
        print("  [SKIP] real data not ready; missing:")
        for p in need:
            if not os.path.isfile(p):
                print("         " + p)
        check("real-data run skipped (stage A/B artefacts absent)", True,
              f"{len(have)}/{len(need)} inputs present")
        return None
    rc5, out5, dt5 = run_cli("step5_inpaint.py", ["--run", run])
    p5, f5 = _parse_summary(out5, "step5")
    check("real step5 exits 0", rc5 == 0, f"exit {rc5}")
    check("real step5 [CHECK] all pass", p5 is not None and f5 == 0,
          f"passed={p5} failed={f5}")
    fmeta = io_utils.read_json(os.path.join(io_utils.run_dir(run, "fill"),
                                            "inpaint_meta.json"))
    print(f"    real removed region {fmeta['removed_px']} px, {fmeta['n_iters']} iters, "
          f"{fmeta['seconds']:.1f}s, {fmeta['ms_per_iteration']:.2f} ms/iter, "
          f"leak {fmeta['fg_source_fraction']}")
    check("real inpaint speed budget (< 600 s)", fmeta["seconds"] < 600.0,
          f"{fmeta['seconds']:.1f}s")
    check("real fill: no foreground sampling",
          fmeta["fg_source_fraction"] is None or fmeta["fg_source_fraction"] < 0.02,
          f"fg_source_fraction={fmeta['fg_source_fraction']}")
    rc6, out6, dt6 = run_cli("step6_render.py", ["--run", run])
    p6, f6 = _parse_summary(out6, "step6")
    check("real step6 exits 0", rc6 == 0, f"exit {rc6}")
    check("real step6 [CHECK] all pass", p6 is not None and f6 == 0,
          f"passed={p6} failed={f6}")
    return fmeta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the 1024x768 speed run and the CLI pipeline runs")
    ap.add_argument("--run", default="ba54_f000")
    a = ap.parse_args()
    t0 = time.perf_counter()
    test_ssd_matches_bruteforce()
    test_inpaint_contract()
    test_no_fg_sampling_and_ablations()
    test_render_postprocess()
    if not a.quick:
        test_speed_full_frame()
        test_pipeline_fixture()
    test_real_data(a.run)
    print("\n" + "=" * 78)
    print(f"tests/test_step5_6.py: {len(NAMES) - len(FAILS)}/{len(NAMES)} passed, "
          f"{len(FAILS)} failed in {time.perf_counter() - t0:.1f}s")
    for f in FAILS:
        print("  FAILED: " + f)
    print("=" * 78)
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
