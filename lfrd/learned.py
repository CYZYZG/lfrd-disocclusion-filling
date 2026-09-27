"""Learned inpainting by sparse representation: what it is and how it is applied here.

The environment has no torch / tensorflow / onnxruntime and no inpainting weights anywhere on
disk (tools/probe_learned_env.py, tools/probe_models_deep.py), and pip is offline, so a
pretrained network is not an option.  What IS available is numpy/cv2/scipy, which is enough for
the pre-deep-learning family of *learned* inpainting: sparse representation over a dictionary
learned from the data.

Why it can beat our patch-copy fill
-----------------------------------
Our occlusion layer is built by copying reference patches, chosen one at a time by SSD, with a
per-run constant depth.  A dictionary attacks the two things that costs us:

* **the flat-patch problem.**  A hole patch has to be matched against patches that are fully
  known; where only a few neighbours are known the SSD is unconstrained and the fill degenerates
  towards whatever is cheap.  Sparse coding solves for the whole patch from the known pixels
  only, so the unknown part is *inferred jointly* rather than copied.
* **no notion of "what this kind of surface looks like".**  A dictionary trained on the scene's
  own background patches encodes that; a single patch copy does not.

Method
------
1. Collect patches from the reference image where they are entirely BACKGROUND
   (outside the foreground mask, depth on the background level).  These are the training set --
   the model learns the appearance of this scene's background, nothing else.
2. Build a dictionary by SVD/PCA of those patches.  It is an orthonormal basis of background
   appearance; `n_atoms` controls capacity.
3. For a hole patch with a known/unknown mask: project the KNOWN pixels onto the basis restricted
   to the known support.  If B is the (P x K) sub-basis for the known pixels, solve
   min ||B c - y||^2 with Tikhonov damping, then reconstruct the UNKNOWN pixels as A c, where A is
   the sub-basis for the unknown pixels.  This is linear regression from what is seen to what is
   hidden -- the learned part.
4. Iterate over the hole with a priority order (as Criminisi does) and re-solve, so each patch
   can also lean on the pixels filled earlier in the same pass.

This module provides steps 1-3 as a reusable estimator plus the iteration of step 4.
"""
from __future__ import annotations

import numpy as np


def collect_patches(img, allowed, patch=9, stride=2, max_patches=40000, rng=None):
    """All `patch`x`patch` patches of `img` that lie entirely inside `allowed`.

    Returns (M, C, patch) where M is (n, patch*patch) for ONE channel; the dictionary is built
    per channel so that there is no ambiguity about how channels are stacked.
    """
    H, W = allowed.shape[:2]
    r = patch // 2
    ys = range(r, H - r, stride)
    xs = range(r, W - r, stride)
    out = []
    for y in ys:
        row_ok = allowed[y - r:y + r + 1]
        for x in xs:
            if not row_ok[:, x - r:x + r + 1].all():
                continue
            out.append(img[y - r:y + r + 1, x - r:x + r + 1])
    C = img.shape[2] if img.ndim == 3 else 1
    if not out:
        return np.zeros((0, patch * patch), np.float32), C, patch
    arr = np.stack(out).astype(np.float32)              # (n, p, p, C)
    n = arr.shape[0]
    if max_patches and n > max_patches:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, max_patches, replace=False)
        arr = arr[idx]
        n = arr.shape[0]
    M = np.transpose(arr, (3, 0, 1, 2)).reshape(C, n, patch * patch)
    return M, C, patch


def fit_dictionary(M, n_atoms=128):
    """PCA/SVD basis fitted per channel.  Returns dict(basis (C,P,K), mean (C,P), eigvals)."""
    C, n, P = M.shape
    means = M.mean(1)                                    # (C, P)
    X = M - means[:, None, :]
    k = int(max(1, min(n_atoms, n - 1, P)))
    basis = np.zeros((C, P, k), np.float32)
    eigvals = np.zeros((C, k), np.float32)
    for c in range(C):
        Xc = X[c]                                        # (n, P)
        if n >= P:
            Cov = (Xc.T @ Xc) / max(1, n - 1)
            w, V = np.linalg.eigh(Cov)
            order = np.argsort(-w)[:k]
            basis[c] = V[:, order].astype(np.float32)
            eigvals[c] = w[order].astype(np.float32)
        else:
            Cov = (Xc @ Xc.T) / max(1, n - 1)
            w, V = np.linalg.eigh(Cov)
            order = np.argsort(-w)[:k]
            B = Xc.T @ V[:, order]
            B /= np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-8)
            basis[c] = B.astype(np.float32)
            eigvals[c] = w[order].astype(np.float32)
    return dict(basis=basis, mean=means, eigvals=eigvals, C=C, P=P)


def solve_patch(dic, y_patch, known_mask, lam=1.0, n_atoms=None):
    """Reconstruct a patch from its KNOWN pixels using the per-channel dictionary.

    y_patch    : (patch, patch, C)
    known_mask : (patch, patch) bool, True where the pixel is trustworthy
    Returns (reconstruction, coefficients (C,K), residual)
    """
    patch = y_patch.shape[0]
    C = y_patch.shape[2] if y_patch.ndim == 3 else 1
    K = dic["basis"].shape[2] if n_atoms is None else min(n_atoms, dic["basis"].shape[2])
    if K < 1:
        return np.array(y_patch, copy=True), None, np.nan
    km = known_mask.reshape(-1)
    if km.sum() < max(3, K // 4):
        return np.array(y_patch, copy=True), None, np.nan
    yv = y_patch.reshape(-1, C) if y_patch.ndim == 3 else y_patch.reshape(-1, 1)
    rec = np.empty((patch * patch, C), np.float32)
    coefs = np.zeros((C, K), np.float64)
    res = []
    for c in range(C):
        A = dic["basis"][c][:, :K]                       # (P, K)
        mu = dic["mean"][c]                              # (P,)
        A_k = A[km]
        y_k = (yv[km, c] - mu[km]).astype(np.float64)
        G = A_k.T @ A_k + lam * np.eye(K)
        rhs = A_k.T @ y_k
        try:
            coef = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(G, rhs, rcond=None)[0]
        rec[:, c] = mu + A @ coef
        coefs[c] = coef
        res.append(float(np.sqrt(((A_k @ coef - y_k) ** 2).mean())))
    out = rec.reshape(patch, patch, C) if y_patch.ndim == 3 else rec.reshape(patch, patch)
    return out.astype(np.float32), coefs, float(np.mean(res))


def dictionary_fill(color, hole, valid, dic, patch=9, n_atoms=None, lam=1.0,
                    max_iters=400_000, verbose=False):
    """Fill `hole` by iterating sparse reconstructions over a priority order.

    The priority follows Criminisi: fill where the most neighbours are known first, so early
    reconstructions are well constrained and later ones can use them.
    """
    H, W = hole.shape[:2]
    out = np.array(color, copy=True)
    known = (valid & ~hole).copy()
    r = patch // 2
    todo = hole & ~known
    n_filled = 0
    it = 0
    ksum = np.ones((patch, patch), np.float32)
    while todo.any() and it < max_iters:
        it += 1
        # confidence = fraction of known pixels in the patch
        cnt = _conv(known.astype(np.float32), ksum)
        prio = np.where(todo, cnt, -1.0)
        y, x = np.unravel_index(int(np.argmax(prio)), prio.shape)
        if y < r or x < r or y >= H - r or x >= W - r:
            todo[y, x] = False
            continue
        kwin = known[y - r:y + r + 1, x - r:x + r + 1]
        twin = todo[y - r:y + r + 1, x - r:x + r + 1]
        if not twin.any():
            todo[y, x] = False
            continue
        rec, coef, resid = solve_patch(dic, out[y - r:y + r + 1, x - r:x + r + 1],
                                       kwin, lam=lam, n_atoms=n_atoms)
        if coef is None:
            # no usable dictionary fit: fall back to the mean of the known neighbours
            m = out[y - r:y + r + 1, x - r:x + r + 1][kwin].mean(0)
            out[y - r:y + r + 1, x - r:x + r + 1][twin] = m
        else:
            out[y - r:y + r + 1, x - r:x + r + 1][twin] = rec[twin]
        known[y - r:y + r + 1, x - r:x + r + 1][twin] = True
        todo[y - r:y + r + 1, x - r:x + r + 1] = False
        todo[y, x] = False
        n_filled += int(twin.sum())
    if verbose:
        print(f"dictionary fill: {n_filled} px in {it} iterations, leftover {int(todo.sum())}")
    return out, dict(filled_px=n_filled, iters=it, leftover=int(todo.sum()))


def _conv(a, k):
    """Valid-ish 2-D convolution via FFT-free shifted sums (small kernels only)."""
    import cv2
    return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE)
