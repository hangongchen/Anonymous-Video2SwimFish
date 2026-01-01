#!/usr/bin/env python
"""3D-ZeF bend recovery -- Phase 0 (calibrate + validate triangulation) & Phase 1 (count clean 3D frames).

Phase 0: solvePnP for the top (camT) and front (camF) cameras from intrinsics + tank-corner references,
         then re-triangulate the GT head pixels and compare to the GIVEN 3D head -> calibration error.
Phase 1: filter one fish to frames unoccluded in BOTH views AND roughly BROADSIDE to the front cam
         (a fish swimming toward/away is a head-on blob with no side-midline) -> usable-second count.
"""
import json, re, os, glob
import numpy as np
import cv2

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


ROOT = _P("${ZEF_ROOT}")
FPS = 60.0


def load_jsonc(path):
    s = open(path).read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)          # strip /* */ comments
    s = re.sub(r"//.*", "", s)                            # strip // comments
    return json.loads(s)


def load_cam(tag):
    intr = load_jsonc(f"{ROOT}/cam{tag}_intrinsic.json")
    refs = load_jsonc(f"{ROOT}/cam{tag}_references.json")
    K = np.array(intr["K"], float)
    dist = np.array(intr["Distortion"][0], float)
    world = np.array([[p["world"]["x"], p["world"]["y"], p["world"]["z"]] for p in refs], float)
    cam = np.array([[p["camera"]["x"], p["camera"]["y"]] for p in refs], float)
    return K, dist, world, cam


def calibrate(tag):
    K, dist, world, cam = load_cam(tag)
    ok, rvec, tvec = cv2.solvePnP(world, cam, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])
    # reprojection error of the reference points
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, dist)
    reproj = np.linalg.norm(proj.reshape(-1, 2) - cam, axis=1)
    print(f"[cam{tag}] {len(world)} ref pts, PnP reproj err mean {reproj.mean():.1f} px max {reproj.max():.1f} px")
    return dict(K=K, dist=dist, R=R, tvec=tvec, P=P, rvec=rvec)


def undistort(pt, cam):
    p = cv2.undistortPoints(np.asarray(pt, float).reshape(1, 1, 2), cam["K"], cam["dist"], P=cam["K"])
    return p.reshape(2)


def triangulate(ptT, ptF, camT, camF):
    uT = undistort(ptT, camT); uF = undistort(ptF, camF)
    X = cv2.triangulatePoints(camT["P"], camF["P"], uT.reshape(2, 1), uF.reshape(2, 1))
    X = (X[:3] / X[3]).reshape(3)
    return X


def main():
    print("=== PHASE 0: calibrate + validate ===")
    camT = calibrate("T"); camF = calibrate("F")

    gt = np.loadtxt(f"{ROOT}/gt/gt.txt", delimiter=",")
    # cols: 0 fr,1 id,2-4 3d,5-6 camT head,7-10 camT bbox,11 camT occ,12-13 camF head,14-17 camF bbox,18 camF occ
    ids = np.unique(gt[:, 1]).astype(int)
    print(f"\nframes {int(gt[:,0].min())}-{int(gt[:,0].max())}, fish ids {list(ids)}, {len(gt)} rows")

    # validate triangulation on rows unoccluded in BOTH views
    clean = gt[(gt[:, 11] == 0) & (gt[:, 18] == 0)]
    errs = []
    for r in clean[:: max(1, len(clean) // 400)]:
        X = triangulate(r[5:7], r[12:14], camT, camF)
        errs.append(np.linalg.norm(X - r[2:5]))
    errs = np.array(errs)
    print(f"\n[validate] triangulated GT head vs given 3D: mean {errs.mean():.3f} cm, median {np.median(errs):.3f} cm, "
          f"p90 {np.percentile(errs,90):.3f} cm  (over {len(errs)} clean rows)  -- tank is 29x29 cm")

    print("\n=== PHASE 1: clean + broadside frame yield per fish ===")
    # broadside to front cam: heading in the xy-plane pointing along x (front cam depth axis = y).
    # proxy A: |heading_x|/|heading| ; proxy B: camF bbox width (broadside fish -> wide front bbox).
    for fid in ids:
        f = gt[gt[:, 1] == fid]
        f = f[np.argsort(f[:, 0])]
        both_clean = (f[:, 11] == 0) & (f[:, 18] == 0)
        # heading from 3D xy trajectory
        xy = f[:, 2:4]
        vel = np.gradient(xy, axis=0)
        spd = np.linalg.norm(vel, axis=1) + 1e-9
        head_x_frac = np.abs(vel[:, 0]) / spd                      # 1 => moving along x (broadside to front)
        camF_w = f[:, 16]
        broadside = head_x_frac > 0.7                              # heading mostly along x
        usable = both_clean & broadside & (spd > 0.05)            # moving, clean, broadside
        print(f" fish {fid}: {len(f)} frames | both-view clean {int(both_clean.sum())} "
              f"({100*both_clean.mean():.0f}%) | +broadside(hx>0.7) {int(usable.sum())} "
              f"= {usable.sum()/FPS:.1f}s | camF bbox w median {np.median(camF_w):.0f}px "
              f"(broadside frames {np.median(camF_w[usable]) if usable.any() else float('nan'):.0f}px)")
    # dump one clean+broadside frame index per fish for Phase 2 viz
    picks = {}
    for fid in ids:
        f = gt[gt[:, 1] == fid]; f = f[np.argsort(f[:, 0])]
        xy = f[:, 2:4]; vel = np.gradient(xy, axis=0); spd = np.linalg.norm(vel, axis=1) + 1e-9
        hx = np.abs(vel[:, 0]) / spd
        m = (f[:, 11] == 0) & (f[:, 18] == 0) & (hx > 0.7) & (spd > 0.05)
        if m.any():
            picks[int(fid)] = [int(x) for x in f[m, 0][:6]]
    print("\n[phase2-picks] clean+broadside example frames per fish:", picks)


main()
