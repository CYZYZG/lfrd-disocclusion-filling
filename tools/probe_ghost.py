"""Probe: (1) OOFA side, (2) ghost-contamination sign for the morphology preprocessing."""
import sys, os
import numpy as np
sys.path.insert(0, r'D:\项目\空洞填补')
sys.stdout.reconfigure(encoding='utf-8')
from dibr import io_utils, calib, warp as W

ROOT = io_utils.DATASET_ROOT_DEFAULT
CAMS = calib.load_calib()


def prep(P, th=20, rounds=2, mode='A'):
    d = P.astype(np.int32)
    for _ in range(rounds):
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dil = cv2.dilate(d.astype(np.uint8), k).astype(np.int32)
        er = (dil - d) > th
        if mode == 'A':
            new = np.where(er, np.maximum(d, cv2.dilate(d.astype(np.uint8), k).astype(np.int32)), d)
            # equation (1)+(4-nbhd fg copy) = dilate-mark, then copy 4-nbhd maximum
            new = np.where(er, cv2.dilate(d.astype(np.uint8), k).astype(np.int32), d)
        else:
            dif2 = d - cv2.erode(d.astype(np.uint8), k).astype(np.int32)
            new = np.where(dif2 > th, cv2.erode(d.astype(np.uint8), k).astype(np.int32), d)
        changed = int((new != d).sum())
        d = new.astype(np.uint8)
    return d, changed


import cv2

def run(src, dst, frame, mode):
    v = io_utils.load_view(ROOT, src, frame)
    gt = io_utils.load_view(ROOT, dst, frame)
    P = v['depth']
    Pp, nch = prep(P, 20, 2, mode)
    z = calib.depth_from_P(P)
    zp = calib.depth_from_P(Pp)
    dx, dy, ut, vt, valid = calib.displacement_field(CAMS, src, dst, P)
    dxp, dyp, _, _, _ = calib.displacement_field(CAMS, src, dst, Pp)
    C = v['color'].astype(np.float32)
    wc, wz, hole, _ = W.forward_warp(C, dx, dy, z=z, rule='zbuf', splat='sub')
    wcp, wzp, holep, _ = W.forward_warp(C, dxp, dyp, z=zp, rule='zbuf', splat='sub')
    m = ~hole
    # ghost band: pixels near the depth silhouette in the SOURCE
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edge = (cv2.dilate(P, k).astype(int) - P.astype(int)) > 20
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band = m & edge
    G = gt['color'].astype(np.float32)
    def mse(a, b, sel):
        d = (a - b)[sel]
        return float((d ** 2).mean()) ** 0.5
    print(f'--- cam{src}->cam{dst} {frame} mode={mode} changed={nch}')
    print(f'   whole valid: raw {mse(wc,G,m):.3f}  prep {mse(wcp,G,m):.3f}')
    print(f'   ghost band : raw {mse(wc,G,band):.3f}  prep {mse(wcp,G,band):.3f}   (band {band.sum()} px)')
    print(f'   hole frac  : raw {hole.mean()*100:.2f}%  prep {holep.mean()*100:.2f}%')
    return wc, wcp, G, band, hole, holep


for mode in ('A', 'B'):
    run(5, 4, 'f000', mode)
    print()
