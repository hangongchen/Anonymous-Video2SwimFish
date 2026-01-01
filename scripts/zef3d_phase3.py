#!/usr/bin/env python
"""3D-ZeF Phase 3: triangulate the top + front midlines into a real 3D body line, and show the
two-number-per-point shape (left/right AND up/down) for the real fish.

For each clean+broadside frame: extract the head->tail midline in BOTH views, resample each to K points
by arc length, triangulate the matching points -> a 3D midline. Then express it in the body's own frame
(head at 0, anterior tangent = forward, world-up = up, left = up x forward) and read off, per station:
  lateral(s) = left/right offset,  vertical(s) = up/down offset,  both / TRUE 3D body length (tilt-free).
"""
import json, re
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


ROOT = _P("${ZEF_ROOT}")
SEG = _P("${FISH_ROOT}/demo_out/zef05_seg/ZebraFish-05")
K = 20


def load_jsonc(p):
    s = re.sub(r"/\*.*?\*/", "", open(p).read(), flags=re.S)
    return json.loads(re.sub(r"//.*", "", s))


def calib(tag):
    intr = load_jsonc(f"{ROOT}/cam{tag}_intrinsic.json")
    refs = load_jsonc(f"{ROOT}/cam{tag}_references.json")
    K_ = np.array(intr["K"], float)
    dist = np.array(intr["Distortion"][0], float)
    world = np.array([[p["world"]["x"], p["world"]["y"], p["world"]["z"]] for p in refs], float)
    cam = np.array([[p["camera"]["x"], p["camera"]["y"]] for p in refs], float)
    ok, rvec, tvec = cv2.solvePnP(world, cam, K_, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return dict(K=K_, dist=dist, P=K_ @ np.hstack([R, tvec]))


def clean_mask(mask, head, bbox, margin=0.20):
    m = (mask > 127).astype(np.uint8)
    l, t, w, h = [float(v) for v in bbox]
    mx, my = int(margin * w), int(margin * h)
    keep = np.zeros_like(m)
    keep[max(0, int(t - my)):int(t + h + my), max(0, int(l - mx)):int(l + w + mx)] = 1
    m = m * keep
    n, lab, st, ct = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    hx, hy = int(round(head[0])), int(round(head[1]))
    L = lab[hy, hx] if (0 <= hy < lab.shape[0] and 0 <= hx < lab.shape[1]) else 0
    if L == 0:
        L = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    return (lab == L).astype(np.uint8)


def midline_pixels(mask, head, bbox, nbins=20):
    """ordered head->tail midline in image pixels, resampled to K points by arc length."""
    m = clean_mask(mask, head, bbox)
    ys, xs = np.nonzero(m)
    if len(xs) < 40:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    axis = P.mean(0) - head
    axis /= np.linalg.norm(axis) + 1e-9
    along = (P - head) @ axis
    lo, hi = np.percentile(along, 1), np.percentile(along, 99)
    edges = np.linspace(lo, hi, nbins + 1)
    pts = []
    for i in range(nbins):
        sel = (along >= edges[i]) & (along <= edges[i + 1])
        if sel.sum() > 3:
            pts.append(P[sel].mean(0))
    if len(pts) < 6:
        return None
    pts = np.array(pts)                                    # head->tail order
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = cum / cum[-1]
    tgt = np.linspace(0, 1, K)
    return np.stack([np.interp(tgt, s, pts[:, 0]), np.interp(tgt, s, pts[:, 1])], 1)   # (K,2)


def undist(pts, cam):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(float), cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)


def triangulate_midline(mT, mF, camT, camF):
    uT = undist(mT, camT).T; uF = undist(mF, camF).T                 # (2,K)
    X = cv2.triangulatePoints(camT["P"], camF["P"], uT, uF)          # (4,K)
    return (X[:3] / X[3]).T                                          # (K,3) world cm


def body_frame_feature(M3d):
    """express the 3D midline in the body frame -> lateral & vertical per station / true body length."""
    head = M3d[0]
    rel = M3d - head
    fwd = rel[max(1, K // 4)] - rel[0]                               # anterior tangent (head quarter)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up = np.array([0.0, 0.0, 1.0])                                   # world up (fish don't roll) -> fixes roll
    up = up - up.dot(fwd) * fwd; up /= np.linalg.norm(up) + 1e-9     # make perpendicular to fwd
    left = np.cross(up, fwd)
    along = rel @ fwd; lateral = rel @ left; vertical = rel @ up
    body_len = np.linalg.norm(np.diff(M3d, axis=0), axis=1).sum()    # TRUE 3D arc length (tilt-free)
    return along / body_len, lateral / body_len, vertical / body_len, body_len


def main():
    camT, camF = calib("T"), calib("F")
    gt = np.loadtxt(f"{ROOT}/gt/gt.txt", delimiter=",")
    f = gt[np.argsort(gt[:, 0])]
    xy = f[:, 2:4]; vel = np.gradient(xy, axis=0); spd = np.linalg.norm(vel, axis=1) + 1e-9
    hx = np.abs(vel[:, 0]) / spd
    broad = f[(f[:, 11] == 0) & (f[:, 18] == 0) & (hx > 0.7) & (spd > 0.05), 0].astype(int)

    all_lat, all_ver, blens = [], [], []
    show = broad[np.linspace(0, len(broad) - 1, 5).astype(int)]
    show_curves = {}
    for fr in broad:
        r = gt[gt[:, 0] == fr][0]
        mT = midline_pixels(cv2.imread(f"{SEG}/imgT/mask/{fr:06d}.png", 0), r[5:7], r[7:11])
        mF = midline_pixels(cv2.imread(f"{SEG}/imgF/mask/{fr:06d}.png", 0), r[12:14], r[14:18])
        if mT is None or mF is None:
            continue
        M3d = triangulate_midline(mT, mF, camT, camF)
        al, la, ve, bl = body_frame_feature(M3d)
        if not (2.0 < bl < 6.0):                       # sanity: adult zebrafish ~3-4 cm
            continue
        all_lat.append(la); all_ver.append(ve); blens.append(bl)
        if fr in show:
            show_curves[fr] = (M3d, al, la, ve)
    all_lat = np.array(all_lat); all_ver = np.array(all_ver); blens = np.array(blens)
    print(f"[phase3] recovered 3D midline for {len(all_lat)}/{len(broad)} broadside frames; "
          f"body length {blens.mean():.2f}+-{blens.std():.2f} cm")
    np.savez("demo_out/zef05_amp_ref/bend3d_raw.npz", lateral=all_lat, vertical=all_ver, body_len_cm=blens)

    # ---- visualize ----
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Real fish 3D body line — TWO numbers per point (left/right AND up/down)", fontsize=13)
    s = np.linspace(0, 1, K)
    ax = fig.add_subplot(2, 2, 1)
    for fr, (M, al, la, ve) in show_curves.items():
        ax.plot(al, la, "-o", ms=3, lw=1.5, label=f"f{fr}")
    ax.axhline(0, color="k", lw=0.4, ls=":"); ax.set_title("(a) TOP view of the 3D line: LEFT/RIGHT bend")
    ax.set_xlabel("along body (head→tail)"); ax.set_ylabel("left / right (body-lengths)"); ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 2, 2)
    for fr, (M, al, la, ve) in show_curves.items():
        ax.plot(al, ve, "-o", ms=3, lw=1.5)
    ax.axhline(0, color="k", lw=0.4, ls=":"); ax.set_title("(b) SIDE view of the 3D line: UP/DOWN bend")
    ax.set_xlabel("along body (head→tail)"); ax.set_ylabel("up / down (body-lengths)")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(s, np.abs(all_lat).mean(0) * np.sqrt(2), "b-o", ms=3, label="left/right amplitude")
    ax.plot(s, np.abs(all_ver).mean(0) * np.sqrt(2), "r-s", ms=3, label="up/down amplitude")
    ax.set_title("(c) how big each bend is, per station (over all frames)")
    ax.set_xlabel("body station (head→tail)"); ax.set_ylabel("amplitude (body-lengths)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for fr, (M, al, la, ve) in show_curves.items():
        Mc = M - M[0]
        ax.plot(Mc[:, 0], Mc[:, 1], Mc[:, 2], "-o", ms=2)
    ax.set_title("(d) the recovered 3D body lines"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("demo_out/zef3d_bend3d.png", dpi=110)
    print("[phase3] saved demo_out/zef3d_bend3d.png")
    print(f"[phase3] left/right amplitude (tail) {np.abs(all_lat).mean(0)[-1]*np.sqrt(2):.3f} BL  |  "
          f"up/down amplitude (tail) {np.abs(all_ver).mean(0)[-1]*np.sqrt(2):.3f} BL")


main()
