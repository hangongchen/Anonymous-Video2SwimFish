#!/usr/bin/env python
"""3D-ZeF Phase 2: extract the per-view MIDLINE (top + front) and overlay on both images -- PAUSE point.

Midline = per-station centroid of the silhouette along the head->body axis (same construction the
bend profile is built from). Head anchored by the GT head pixel per view. Shows several clean+broadside
frames as [top view + midline | front view + midline] so the extraction quality can be judged before
committing to triangulation.
"""
import numpy as np, cv2, os
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
OUT = _P("${FISH_ROOT}/demo_out/zef3d_midlines.png")


def clean_mask(mask, head_px, bbox=None, margin=0.20):
    """Reflection reject: (1) intersect with the GT bbox+margin -- reflections/shadows extend OUTSIDE the
    annotated box (the frame-395 case: mask reached y=1362 but the bbox ended at 1319, a MERGED blob CC
    splitting can't cut); (2) then keep the component containing the GT head (separate-blob reflections)."""
    m = (mask > 127).astype(np.uint8)
    if bbox is not None:
        l, t, w, h = [float(v) for v in bbox]
        mx, my = int(margin * w), int(margin * h)
        keep = np.zeros_like(m)
        y0, y1 = max(0, int(t - my)), min(m.shape[0], int(t + h + my))
        x0, x1 = max(0, int(l - mx)), min(m.shape[1], int(l + w + mx))
        keep[y0:y1, x0:x1] = 1
        m = m * keep
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return (m * 255).astype(np.uint8)
    hx, hy = int(round(head_px[0])), int(round(head_px[1]))
    lab = labels[hy, hx] if (0 <= hy < labels.shape[0] and 0 <= hx < labels.shape[1]) else 0
    if lab == 0:
        cand = [(np.linalg.norm(cents[i] - head_px), i) for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] > 30]
        lab = min(cand)[1] if cand else int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return ((labels == lab) * 255).astype(np.uint8)


def midline(mask, head_px, bbox=None, nbins=18):
    mask = clean_mask(mask, head_px, bbox)                  # reflection-reject before extracting
    ys, xs = np.nonzero(mask > 127)
    if len(xs) < 40:
        return None, None
    P = np.stack([xs, ys], 1).astype(float)
    ctr = P.mean(0)
    axis = ctr - head_px
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None, None
    axis = axis / n
    along = (P - head_px) @ axis
    lo, hi = np.percentile(along, 1), np.percentile(along, 99)
    edges = np.linspace(lo, hi, nbins + 1)
    mids = []
    for i in range(nbins):
        sel = (along >= edges[i]) & (along <= edges[i + 1])
        if sel.sum() > 3:
            mids.append(P[sel].mean(0))
    return (np.array(mids) if mids else None), P


gt = np.loadtxt(f"{ROOT}/gt/gt.txt", delimiter=",")
f = gt[np.argsort(gt[:, 0])]
xy = f[:, 2:4]; vel = np.gradient(xy, axis=0); spd = np.linalg.norm(vel, axis=1) + 1e-9
hx = np.abs(vel[:, 0]) / spd
broad = f[(f[:, 11] == 0) & (f[:, 18] == 0) & (hx > 0.7) & (spd > 0.05), 0].astype(int)
picks = broad[np.linspace(0, len(broad) - 1, 5).astype(int)]
print(f"[phase2] {len(broad)} broadside frames; visualizing {list(picks)}")

fig, axs = plt.subplots(len(picks), 2, figsize=(11, 3.0 * len(picks)))
fig.suptitle("3D-ZeF per-view MIDLINE extraction (top | front) — clean+broadside frames", fontsize=13)
for row, fr in enumerate(picks):
    r = gt[gt[:, 0] == fr][0]
    for col, (tag, headcol, bboxcol, imgdir) in enumerate([
            ("TOP (x,y)", (5, 6), (7, 8, 9, 10), "imgT"),
            ("FRONT (x,z)", (12, 13), (14, 15, 16, 17), "imgF")]):
        ax = axs[row, col] if len(picks) > 1 else axs[col]
        img = cv2.imread(f"{ROOT}/{imgdir}/{fr:06d}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        head = np.array([r[headcol[0]], r[headcol[1]]])
        l, t, w, h = r[list(bboxcol)].astype(int)
        mpath = f"{SEG}/{imgdir}/mask/{fr:06d}.png"
        mids = None
        if os.path.exists(mpath):
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            mids, _ = midline(mask, head, (l, t, w, h))
        # crop around bbox with margin
        mx = int(0.6 * w); my = int(0.9 * h)
        x0, y0 = max(0, l - mx), max(0, t - my)
        x1, y1 = min(img.shape[1], l + w + mx), min(img.shape[0], t + h + my)
        ax.imshow(img[y0:y1, x0:x1])
        if mids is not None and len(mids) > 2:
            ax.plot(mids[:, 0] - x0, mids[:, 1] - y0, "-o", color="lime", ms=4, lw=1.8)
            ax.plot(mids[0, 0] - x0, mids[0, 1] - y0, "s", color="red", ms=7)     # head end
            ax.plot(mids[-1, 0] - x0, mids[-1, 1] - y0, "^", color="cyan", ms=7)  # tail end
        ax.plot(head[0] - x0, head[1] - y0, "*", color="yellow", ms=13, mec="k")  # GT head
        ax.set_title(f"frame {fr}  {tag}  (mask {'ok' if mids is not None else 'MISSING'})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=115)
print(f"[phase2] saved {OUT}")
