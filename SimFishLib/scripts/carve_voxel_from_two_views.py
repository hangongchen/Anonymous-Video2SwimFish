#!/usr/bin/env python
"""Two-view silhouette carving -> coarse 64^3 voxel.

Takes the FRONT (imgF) and TOP (imgT) fish masks of a SINGLE frame (one pose)
and carves an orthographic visual hull:

  world axes:  X = body length, Y = width (left-right), Z = height (dorsal-ventral)
  top  view (looking down -Z)  gives the (X, Y) silhouette  -> width  profile
  front view (looking along -Y) gives the (X, Z) silhouette -> height profile
  occ(x,y,z) = top_silhouette(x, y) AND front_silhouette(x, z)

Both masks are PCA-rotated so the body long axis is horizontal, flipped so the
head (from gt) points +X, and rescaled to a common body length so the shared
X axis lines up between the two views. This is a canonical-frame, single-pose
coarse prior (high recall) meant to be handed to the flow refiner to carve down.

Usage:
  python scripts/carve_voxel_from_two_views.py --seq ZebraFish-05 --frame 532
"""
import argparse, os
import cv2
import numpy as np

VIEW_COLS = {  # gt.txt 0-based cols: head x/y per view
    "imgF": dict(cx=12, cy=13),
    "imgT": dict(cx=5, cy=6),
}


def load_mask(path):
    m = cv2.imread(path, 0)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def pca_angle(mask):
    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys]).astype(np.float64).T
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]                     # unit major-axis vector (x,y)
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return angle, c


def _warp(mask, c, angle):
    H, W = mask.shape
    diag = int(np.hypot(H, W))
    canvas = np.zeros((2 * diag, 2 * diag), np.uint8)
    off = (diag - int(c[1]), diag - int(c[0]))     # place centroid at canvas center
    canvas[off[0]:off[0] + H, off[1]:off[1] + W] = mask
    cc = (diag, diag)
    M = cv2.getRotationMatrix2D(cc, angle, 1.0)
    rot = cv2.warpAffine(canvas, M, (2 * diag, 2 * diag), flags=cv2.INTER_NEAREST)
    return rot, M, off


def rotate_upright(mask, head_xy):
    """Rotate so the body long axis is horizontal and the HEAD (from gt) points
    +X. The head side is anchored to the ground-truth head position of THIS view,
    so front and top views agree on which end is the head -> the carve matches
    head-to-head / tail-to-tail across views."""
    angle, c = pca_angle(mask)
    best = None
    for a in (angle, -angle, angle + 180, -angle + 180):
        rot, M, off = _warp(mask, c, a)
        resid = abs(pca_angle(rot)[0])              # residual angle of result
        resid = min(resid, abs(resid - 180))        # fold to [0,90]
        if best is None or resid < best[0]:
            best = (resid, rot, M, off)
    _, rot, M, off = best
    # track the gt head point through the same warp: (x,y) -> M @ [x+off, y+off, 1]
    hx, hy = head_xy[0] + off[1], head_xy[1] + off[0]
    hv = M @ np.array([hx, hy, 1.0])
    hc, hr = float(hv[0]), float(hv[1])             # head col, row in rotated canvas
    ys, xs = np.where(rot > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = rot[y0:y1 + 1, x0:x1 + 1]
    hc -= x0; hr -= y0
    # guarantee horizontal (long axis along width)
    if crop.shape[0] > crop.shape[1]:
        W = crop.shape[1]
        crop = np.rot90(crop).copy()                # new[i,j]=old[j, W-1-i]
        hr, hc = (W - 1 - hc), hr                   # (row,col) of head after rot90
    # flip so the gt head is on the RIGHT (+X)
    if hc < crop.shape[1] / 2.0:
        crop = crop[:, ::-1].copy()
    return crop


def resize_to_length(mask, L):
    h, w = mask.shape
    s = L / w
    out = cv2.resize(mask, (L, max(1, int(round(h * s)))), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def carve(front, top, grid=64, length=None):
    # length=None -> fill the long (head-tail) axis to the full grid, matching the
    # training coarse frame (mesh normalized so its longest axis fills [0,1], i.e.
    # the whole 64 grid; Y/Z centered). This makes the inference coarse land in the
    # SAME unit-cube frame the refiner was retrained on. (Old default was 54, a
    # ~15% scale mismatch vs training.)
    if length is None:
        length = grid
    fr = resize_to_length(front, length)   # (Hf, L)  rows->Z
    tp = resize_to_length(top, length)     # (Wt, L)  rows->Y
    Hf, L = fr.shape
    Wt, _ = tp.shape
    Hf = min(Hf, grid - 2); Wt = min(Wt, grid - 2)
    fr = fr[:Hf]; tp = tp[:Wt]
    occ = np.zeros((grid, grid, grid), bool)        # [X, Y, Z]
    x0 = (grid - L) // 2
    y0 = (grid - Wt) // 2
    z0 = (grid - Hf) // 2
    for i in range(L):
        fcol = fr[:, i].astype(bool)                # (Hf,) over Z
        tcol = tp[:, i].astype(bool)                # (Wt,) over Y
        if not fcol.any() or not tcol.any():
            continue
        block = np.outer(tcol, fcol)                # (Wt, Hf)
        occ[x0 + i, y0:y0 + Wt, z0:z0 + Hf] = block
    return occ


def _tile(a, lbl, sz=256):
    """Pad a binary image to square (no aspect distortion), upscale, label."""
    h, w = a.shape
    s = int(max(h, w))
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    sq = cv2.resize(sq * 255 if sq.max() <= 1 else sq, (sz, sz), interpolation=cv2.INTER_NEAREST)
    sq = cv2.cvtColor(sq, cv2.COLOR_GRAY2BGR)
    cv2.putText(sq, lbl, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return sq


def render_projections(occ, out_path, extra=None):
    xy = occ.any(2).T.astype(np.uint8)              # top  -> X horizontal
    xz = occ.any(1).T.astype(np.uint8)              # front-> X horizontal
    yz = occ.any(0).astype(np.uint8)                # head-on (Y,Z)
    row = np.hstack([_tile(xy, "proj top (X-Y)"), _tile(xz, "proj front (X-Z)"), _tile(yz, "proj head-on")])
    if extra is not None:
        row = np.vstack([extra, row])
    cv2.imwrite(out_path, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ZebraFish-05")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--mask-root", default="/work/yzha1/SimFishLib/outputs/zef_fish_seg")
    ap.add_argument("--gt-root", default="/work/yzha1/SimFishLib/Dataset/3D-ZeF/data")
    ap.add_argument("--out", default="/work/yzha1/SimFishLib/outputs/zef_carve")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--length", type=int, default=None,
                    help="head-tail axis length in voxels; default None = fill the grid "
                         "(unit-cube frame matching the retrained refiner)")
    args = ap.parse_args()

    fr = args.frame
    gt = np.loadtxt(os.path.join(args.gt_root, args.seq, "gt", "gt.txt"), delimiter=",")
    row = gt[gt[:, 0] == fr][0]
    mroot = os.path.join(args.mask_root, args.seq)
    mF = load_mask(os.path.join(mroot, "imgF", "mask", f"{fr:06d}.png"))
    mT = load_mask(os.path.join(mroot, "imgT", "mask", f"{fr:06d}.png"))
    headF = (row[VIEW_COLS["imgF"]["cx"]], row[VIEW_COLS["imgF"]["cy"]])
    headT = (row[VIEW_COLS["imgT"]["cx"]], row[VIEW_COLS["imgT"]["cy"]])

    front = rotate_upright(mF, headF)
    top = rotate_upright(mT, headT)
    print(f"front crop {front.shape}  top crop {top.shape}")
    occ = carve(front, top, grid=args.grid, length=args.length)
    ex = [np.ptp(np.where(occ)[i]) + 1 if occ.any() else 0 for i in range(3)]
    print(f"occ extents  X={ex[0]} Y={ex[1]} Z={ex[2]}")

    os.makedirs(args.out, exist_ok=True)
    npy = os.path.join(args.out, f"{args.seq}_f{fr:04d}_voxel{args.grid}.npy")
    np.save(npy, occ)

    pad = np.zeros((256, 256, 3), np.uint8)
    top_row = np.hstack([_tile(front, "front upright"), _tile(top, "top upright"), pad])
    render_projections(occ, os.path.join(args.out, f"{args.seq}_f{fr:04d}_check.png"), extra=top_row)
    print(f"occupied {occ.sum()} / {occ.size}  ({100*occ.mean():.2f}%)")
    print("saved", npy)
    print("check", os.path.join(args.out, f"{args.seq}_f{fr:04d}_check.png"))


if __name__ == "__main__":
    main()
