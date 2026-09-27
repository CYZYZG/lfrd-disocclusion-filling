"""Camera calibration, MSR inverse-depth model, projection and displacement fields.

Everything here mirrors the conventions of the MSR 3D Video distribution and of the
sibling reproduction project (D:\\项目\\空洞填补\\dibr\\calib.py), which this paper shares:

  * depth PNG stores INVERSE depth:  z = 1 / ((P/255)*(1/MinZ - 1/MaxZ) + 1/MaxZ)
    MinZ = 42.0, MaxZ = 130.0  (same length unit as the calibration translations)
  * the (0,0) coordinate of an image is the BOTTOM LEFT corner
  * therefore P LARGER  ==  z SMALLER  ==  NEARER  ==  FOREGROUND
"""
import os

import numpy as np

from . import io_utils

MINZ, MAXZ = 42.0, 130.0
IMG_H, IMG_W = 768, 1024

# the horizontal shift of the *content* when going from a to b, in pixels:
#   shifts are per-depth: the disocclusion opens on the side the FOREGROUND moved away
#   from.  Measured for cam5->cam4: background +48.9 px, foreground -12.0 px, so the gap
#   is on the RIGHT of the foreground (confirmed visually).  See `disocclusion_side`.
# computed from the geometry of a mid-depth point, see `content_shift`.
_MID_Z = 0.5 * (MINZ + MAXZ)


# --------------------------------------------------------------------------- #
# calibration file
# --------------------------------------------------------------------------- #
def parse_calib(path):
    """Parse calibParams-*.txt -> {cam_index: dict(K, R, t, C)}."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().split("\n")
    cams, i = {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        idx = int(lines[i].strip())
        K = np.array([[float(x) for x in lines[i + 1].split()],
                      [float(x) for x in lines[i + 2].split()],
                      [float(x) for x in lines[i + 3].split()]])
        Rt = np.array([[float(x) for x in lines[i + 5].split()],
                       [float(x) for x in lines[i + 6].split()],
                       [float(x) for x in lines[i + 7].split()]])
        R, t = Rt[:, :3], Rt[:, 3]
        cams[idx] = {"K": K, "R": R, "t": t, "C": -R.T @ t}
        i += 8
    return cams


def load_calib(dataset_root=io_utils.DATASET_ROOT_DEFAULT, name="ballet"):
    path = os.path.join(dataset_root, f"calibParams-{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"calibration not found: {path}")
    return parse_calib(path)


# --------------------------------------------------------------------------- #
# official inverse depth model
# --------------------------------------------------------------------------- #
def depth_from_P(P):
    """8-bit inverse-depth intensity -> z (same unit as the calibration translations)."""
    P = np.asarray(P, dtype=np.float64)
    return 1.0 / ((P / 255.0) * (1.0 / MINZ - 1.0 / MAXZ) + 1.0 / MAXZ)


def invdepth_from_z(z):
    """z -> 8-bit intensity P."""
    z = np.asarray(z, dtype=np.float64)
    return (1.0 / z - 1.0 / MAXZ) / (1.0 / MINZ - 1.0 / MAXZ) * 255.0


def baseline(cams, a, b, axis=0):
    return float(cams[b]["C"][axis] - cams[a]["C"][axis])


# --------------------------------------------------------------------------- #
# projection  (u,v are top-down pixel coordinates)
# --------------------------------------------------------------------------- #
def _unproject(u, v, z, K, y_origin="bottom"):
    cy = (IMG_H - 1) - K[1, 2] if y_origin == "bottom" else K[1, 2]
    sy = -1.0 if y_origin == "bottom" else 1.0
    yc = sy * (v - cy) / K[1, 1] * z
    xc = (u - K[0, 2] - K[0, 1] * yc / K[1, 1] / z) / K[0, 0] * z
    return xc, yc, z


def _project(xc, yc, zc, K, y_origin="bottom"):
    cy = (IMG_H - 1) - K[1, 2] if y_origin == "bottom" else K[1, 2]
    sy = -1.0 if y_origin == "bottom" else 1.0
    u = K[0, 0] * xc / zc + K[0, 1] * yc / zc + K[0, 2]
    v = cy + sy * K[1, 1] * yc / zc
    return u, v


def unproject(u, v, z, cam, y_origin="bottom"):
    """Pixel (u,v)+depth -> camera-frame 3D point. Accepts scalars or arrays."""
    return _unproject(u, v, z, cam["K"], y_origin)


def project(p, cam, y_origin="bottom"):
    """Camera-frame 3D point(s) -> pixel. `p` is (...,3) or a 3-tuple."""
    p = np.asarray(p, dtype=np.float64)
    return _project(p[..., 0], p[..., 1], p[..., 2], cam["K"], y_origin)


def cam_to_cam(p, cam_a, cam_b):
    """3D point(s) in camera-a frame -> camera-b frame."""
    p = np.asarray(p, dtype=np.float64)
    q = p - cam_a["t"]
    q = q @ cam_a["R"]
    return q @ cam_b["R"].T + cam_b["t"]


def project_pts(u, v, z, cam_a, cam_b, y_origin="bottom"):
    """Reference-camera pixel+depth -> target-camera pixel. Arrays, any shape.

    Returns (u_t, v_t, valid) with valid = point is in front of cam_b.
    """
    xc, yc, zc = unproject(u, v, z, cam_a, y_origin)
    p = np.stack([xc, yc, zc], -1)
    q = cam_to_cam(p, cam_a, cam_b)
    ut, vt = project(q, cam_b, y_origin)
    return ut, vt, q[..., 2] > 1e-6


def displacement_field(cams, src, dst, P, y_origin="bottom", step=1):
    """Exact per-pixel (dx, dy) taking src pixels to the dst camera image.

    target_xy = source_xy + (dx, dy).  Returns (dx, dy, u_t, v_t, valid).
    """
    Ks = cams[src]["K"]
    z = depth_from_P(np.asarray(P, dtype=np.float64)[::step, ::step])
    v, u = np.mgrid[0:IMG_H:step, 0:IMG_W:step]
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    xc, yc, zc = _unproject(u, v, z, Ks, y_origin)
    p = np.stack([xc, yc, zc], -1) - cams[src]["t"]
    p = p @ cams[src]["R"]
    p = p @ cams[dst]["R"].T + cams[dst]["t"]
    u_t, v_t = _project(p[..., 0], p[..., 1], p[..., 2], cams[dst]["K"], y_origin)
    valid = p[..., 2] > 1e-6
    return u_t - u, v_t - v, u_t, v_t, valid


def disparity_scale(cams, src, dst):
    """Signed (dst_u - src_u) per unit of (P/255) at mid depth -- the 1D disparity slope."""
    z = _MID_Z
    ut, _, ok = project_pts(np.array(512.0), np.array(384.0), np.array(z),
                            cams[src], cams[dst])
    if not ok:
        return 0.0
    Pmid = invdepth_from_z(z) / 255.0
    return float((ut - 512.0) / Pmid)


def content_shift(cams, src, dst, z_bg=_MID_Z, z_fg=MINZ * 1.2):
    """(shift_bg, shift_fg): horizontal pixel shift of background / foreground content."""
    out = []
    for z in (z_bg, z_fg):
        ut, _, ok = project_pts(np.array(512.0), np.array(384.0), np.array(float(z)),
                                cams[src], cams[dst])
        out.append(float(ut - 512.0) if ok else 0.0)
    return tuple(out)


def disocclusion_side(cams, src, dst):
    """Which side of the foreground object the disocclusion opens on.

    Returns +1 (the gap is on the object's RIGHT) or -1 (on its LEFT).
    The gap opens on the side the FOREGROUND moved away from: a foreground shift
    of +x means the object left a gap behind it on its left (-1).

    NOTE: the paper says "for the right synthesized virtual view, disocclusion appears
    on the right side of foreground", but the MSR Ballet camera geometry gives
    foreground shift = -12.04 px for cam5->cam4, i.e. the gap is on the RIGHT of the
    foreground -- the paper's sign convention for "right/left synthesized view" is
    therefore inverted relative to the reference camera's viewing direction.  We follow
    the geometry (verified visually on the data), and the per-hole background side is in
    any case decided from the depth classification, not from this helper.
    """
    _, fg_shift = content_shift(cams, src, dst)
    return +1 if fg_shift <= 0 else -1


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def describe(cams, src, dst, P, out_dir=None):
    """Calibration report for one camera pair; optionally writes artefacts."""
    dx, dy, u_t, v_t, valid = displacement_field(cams, src, dst, P)
    z = depth_from_P(P)
    m = valid
    lines = []

    def log(s=""):
        lines.append(s)

    shift_bg, shift_fg = content_shift(cams, src, dst)
    side = disocclusion_side(cams, src, dst)
    log(f"--- step 0 calibration report: cam{src} -> cam{dst} ---")
    log(f"camera centres : cam{src} = {np.round(cams[src]['C'], 4)}")
    log(f"                 cam{dst} = {np.round(cams[dst]['C'], 4)}")
    log(f"baseline       : |dC| = {np.linalg.norm(cams[dst]['C'] - cams[src]['C']):.4f} "
        f"(dX = {baseline(cams, src, dst, 0):+.4f})")
    log(f"focal (px)     : fx = {cams[src]['K'][0, 0]:.2f}, fy = {cams[src]['K'][1, 1]:.2f}")
    log(f"depth z        : {z[P > 0].min():.2f} .. {z[P > 0].max():.2f} "
        f"(mean {z[P > 0].mean():.2f});  P nonzero: {int((P > 0).sum())} px")
    log(f"content shift  : background {shift_bg:+.2f} px, foreground {shift_fg:+.2f} px")
    log(f"disocclusion opens on the {'LEFT' if side < 0 else 'RIGHT'} side of the foreground")
    log(f"dx (px)        : min {dx[m].min():+.2f} max {dx[m].max():+.2f} "
        f"mean {dx[m].mean():+.2f} std {dx[m].std():.2f}")
    log(f"dy (px)        : min {dy[m].min():+.2f} max {dy[m].max():+.2f} "
        f"mean {dy[m].mean():+.2f} std {dy[m].std():.2f}")
    log(f"vertical residual (pure-horizontal assumption): median |dy| = "
        f"{np.median(np.abs(dy[m])):.2f} px, p95 = {np.percentile(np.abs(dy[m]), 95):.2f} px")

    if out_dir:
        from . import viz
        viz.imwrite_field(os.path.join(out_dir, f"dx_cam{src}_to_cam{dst}.png"), dx, valid)
        viz.imwrite_field(os.path.join(out_dir, f"dy_cam{src}_to_cam{dst}.png"), dy, valid)
        io_utils.save_npz(os.path.join(out_dir, f"field_cam{src}_cam{dst}.npz"),
                          dx=dx.astype(np.float32), dy=dy.astype(np.float32),
                          u_t=u_t.astype(np.float32), v_t=v_t.astype(np.float32),
                          valid=valid)
        io_utils.write_text(os.path.join(out_dir, f"report_cam{src}_to_cam{dst}.txt"),
                            "\n".join(lines) + "\n")
    return dict(dx=dx, dy=dy, u_t=u_t, v_t=v_t, valid=valid, lines=lines,
                shift_bg=shift_bg, shift_fg=shift_fg, disocc_side=side)
