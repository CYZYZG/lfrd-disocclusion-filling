"""Objective metrics: PSNR, SSIM, FSIMc, VSI + small reporting helpers.

The paper evaluates PSNR / SSIM / FSIMc / VSI on the whole synthesized frame against the
real capture of the virtual camera.  We implement PSNR and SSIM exactly (eq. 11-13) and
provide FSIMc / VSI as faithful implementations of the published feature-based indices
(used by the paper but not needed to validate the filling itself).

All functions take RGB uint8 (H,W,3); masks are bool.
"""
import numpy as np

from . import io_utils, viz


# --------------------------------------------------------------------------- #
# PSNR / SSIM
# --------------------------------------------------------------------------- #
def _as_float(a):
    a = np.asarray(a, np.float64)
    return a


def mse(a, b, mask=None):
    a, b = _as_float(a), _as_float(b)
    d = (a - b) ** 2
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        if not m.any():
            return float("nan")
        return float(d[m].mean())
    return float(d.mean())


def psnr(a, b, mask=None, peak=255.0):
    e = mse(a, b, mask)
    if not np.isfinite(e):
        return float("nan")
    if e == 0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / e))


def ssim(a, b, mask=None, data_range=255.0):
    """Mean SSIM (Wang et al. 2004) with an 11x11 Gaussian window, sigma=1.5."""
    from scipy.ndimage import gaussian_filter
    a = _as_float(a)
    b = _as_float(b)
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]
    sigma = 1.5
    k = (11, 11)
    win = lambda x: gaussian_filter(x, sigma, mode="reflect", truncate=3.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    out = []
    for c in range(a.shape[2]):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        mux, muy = win(x), win(y)
        xx, yy, xy = win(x * x), win(y * y), win(x * y)
        nxx = xx - mux * mux
        nyy = yy - muy * muy
        nxy = xy - mux * muy
        s = ((2 * mux * muy + c1) * (2 * nxy + c2)) / \
            ((mux ** 2 + muy ** 2 + c1) * (nxx + nyy + c2) + 1e-12)
        out.append(s)
    s = np.mean(out, axis=0)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        if not m.any():
            return float("nan")
        return float(s[m].mean())
    return float(s.mean())


# --------------------------------------------------------------------------- #
# FSIMc (Zhang et al. 2011) and VSI (Zhang et al. 2014)
# --------------------------------------------------------------------------- #
def _grad_mag_phase(img):
    """Scharr gradient magnitude + phase congruency proxy (PC via 2-scale log-Gabor)."""
    from scipy.ndimage import gaussian_filter
    img = img.astype(np.float64)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    # Scharr
    kx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], np.float64) / 16.0
    ky = kx.T
    from scipy.ndimage import convolve
    gx = convolve(img, kx, mode="reflect")
    gy = convolve(img, ky, mode="reflect")
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag, gx, gy


def _phase_congruency(img, nscale=4, norient=4):
    """A compact phase-congruency implementation (Kovesi-style, log-Gabor filter bank)."""
    from scipy.ndimage import gaussian_filter
    img = img.astype(np.float64)
    rows, cols = img.shape
    # frequency grid
    u1, u2 = np.meshgrid(np.arange(-cols // 2, cols - cols // 2),
                         np.arange(-rows // 2, rows - rows // 2))
    u1 = u1 / cols
    u2 = u2 / rows
    radius = np.sqrt(u1 ** 2 + u2 ** 2)
    theta = np.arctan2(-u2, u1)
    radius[rows // 2, cols // 2] = 1.0
    sintheta, costheta = np.sin(theta), np.cos(theta)
    PC = np.zeros((rows, cols), np.float64)
    epsilon = 1e-4
    for s in range(nscale):
        wavelength = 3.0 * (1.6 ** s)
        fo = 1.0 / wavelength
        logGabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(0.65) ** 2))
        logGabor[rows // 2, cols // 2] = 0.0
        for o in range(norient):
            angl = o * np.pi / norient
            ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
            dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            spread = np.exp(-(dtheta ** 2) / (2 * (np.pi / norient / 1.2) ** 2))
            G = logGabor * spread
            F = np.fft.ifft2(np.fft.ifftshift(G * np.fft.fftshift(np.fft.fft2(img))))
            an = np.abs(F)
            PC += an
    PC = PC / (PC.max() + 1e-12)
    # small blur to reduce ringing
    return gaussian_filter(PC, 1.0, mode="reflect")


def fsimc(a, b, mask=None, T1=170.0, T2=200.0, lamb=0.03):
    """FSIM/FSIMc: gradient magnitude similarity + phase congruency + colour similarity."""
    a = _as_float(a)
    b = _as_float(b)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
        b = np.stack([b] * 3, -1)
    y1 = a.mean(axis=2)
    y2 = b.mean(axis=2)
    m1, _, _ = _grad_mag_phase(y1)
    m2, _, _ = _grad_mag_phase(y2)
    gm = (2 * m1 * m2 + T1) / (m1 ** 2 + m2 ** 2 + T1)
    pc1 = _phase_congruency(y1)
    pc2 = _phase_congruency(y2)
    pc = (2 * pc1 * pc2 + T2) / (pc1 ** 2 + pc2 ** 2 + T2)
    pcm = np.maximum(pc1, pc2)
    # colour similarity (YIQ chroma)
    def chroma(x):
        r, g, bl = x[..., 0], x[..., 1], x[..., 2]
        i = 0.596 * r - 0.274 * g - 0.322 * bl
        q = 0.211 * r - 0.523 * g + 0.312 * bl
        return i, q
    i1, q1 = chroma(a)
    i2, q2 = chroma(b)
    sc = (2 * i1 * i2 + T1) / (i1 ** 2 + i2 ** 2 + T1) * \
         (2 * q1 * q2 + T1) / (q1 ** 2 + q2 ** 2 + T1)
    # SC is a product of two bounded-positive ratios; clip for the fractional power
    sc = np.clip(sc, 1e-6, None)
    s = (gm * (sc ** lamb)) * pc
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        w = pcm[m]
        if w.size == 0 or w.sum() == 0:
            return float("nan")
        return float((s[m] * w).sum() / w.sum())
    return float((s * pcm).sum() / (pcm.sum() + 1e-12))


def vsi(a, b, mask=None):
    """VSI: visual saliency (SDSP) weighted feature similarity.  Uses FSIMc's features."""
    a = _as_float(a)
    b = _as_float(b)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
        b = np.stack([b] * 3, -1)
    m1, gx1, gy1 = _grad_mag_phase(a.mean(axis=2))
    m2, gx2, gy2 = _grad_mag_phase(b.mean(axis=2))
    # feature similarity: gradient magnitude + chroma
    T1 = 170.0
    def chroma(x):
        r, g, bl = x[..., 0], x[..., 1], x[..., 2]
        return 0.596 * r - 0.274 * g - 0.322 * bl, 0.211 * r - 0.523 * g + 0.312 * bl
    i1, q1 = chroma(a)
    i2, q2 = chroma(b)
    sc = (2 * i1 * i2 + T1) / (i1 ** 2 + i2 ** 2 + T1) * \
         (2 * q1 * q2 + T1) / (q1 ** 2 + q2 ** 2 + T1)
    sg = (2 * m1 * m2 + T1) / (m1 ** 2 + m2 ** 2 + T1)
    S = sc * sg
    # saliency: SDSP (spectral residual style) -- cheap approximation via local variance
    from scipy.ndimage import gaussian_filter
    sal = gaussian_filter(np.abs(gx1) + np.abs(gy1), 8.0, mode="reflect")
    sal = sal / (sal.max() + 1e-12)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.ndim == 3:
            m = m[..., 0]
        w = sal[m]
        if w.size == 0 or w.sum() == 0:
            return float("nan")
        return float((S[m] * w).sum() / w.sum())
    return float((S * sal).sum() / (sal.sum() + 1e-12))


# --------------------------------------------------------------------------- #
# report helpers
# --------------------------------------------------------------------------- #
def eval_pair(pred, gt, mask=None, do_fsimc=False, do_vsi=False):
    """One comparison -> dict of metrics."""
    out = {"psnr": psnr(pred, gt, mask), "ssim": ssim(pred, gt, mask)}
    if do_fsimc:
        out["fsimc"] = fsimc(pred, gt, mask)
    if do_vsi:
        out["vsi"] = vsi(pred, gt, mask)
    out["n_px"] = int(mask.sum()) if mask is not None else int(pred.shape[0] * pred.shape[1])
    return out


def progress_chart(series, path, keys=None, title="PSNR per frame"):
    """Line chart of one or more PSNR series.

    `series` may be either a list of per-frame dicts (keys = arm names) or a dict
    {arm: [values]}.  Pure numpy/cv2, no matplotlib.
    """
    import cv2
    if isinstance(series, dict):
        data = {k: [v for v in vals] for k, vals in series.items()}
    else:
        data = {}
        for r in series:
            for k, v in r.items():
                data.setdefault(k, []).append(v)
    keys = list(data.keys()) if keys is None else [k for k in keys if k in data]
    W, H, M = 1000, 360, 50
    img = np.full((H, W, 3), 255, np.uint8)
    all_v = [v for k in keys for v in data[k] if v is not None and np.isfinite(v)]
    if not all_v or not keys:
        io_utils.imwrite(path, img)
        return img
    lo, hi = min(all_v) - 0.5, max(all_v) + 0.5
    n = max(len(data[k]) for k in keys)
    colors = {"warp": (200, 0, 0), "ghost": (0, 140, 0), "inpaint_direct": (0, 120, 220),
              "ours": (0, 0, 200)}
    for k in keys:
        vals = data[k]
        pts = []
        for i, v in enumerate(vals):
            if v is None or not np.isfinite(v):
                continue
            x = M + int((W - 2 * M) * i / max(1, len(vals) - 1))
            y = H - M - int((H - 2 * M) * (v - lo) / max(1e-9, hi - lo))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, colors.get(k, (0, 0, 0)), 2)
        if pts:
            cv2.putText(img, f"{k}", (pts[-1][0] + 4, pts[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(k, (0, 0, 0)), 1)
    cv2.line(img, (M, H - M), (W - M, H - M), (0, 0, 0), 1)
    cv2.line(img, (M, M), (M, H - M), (0, 0, 0), 1)
    cv2.putText(img, f"{title}   y: {lo:.1f}..{hi:.1f} dB", (M, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    io_utils.imwrite(path, img)
    return img
