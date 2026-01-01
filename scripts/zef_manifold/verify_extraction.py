#!/usr/bin/env python
"""Step 2: VERIFY the curvature extraction (visual + quantitative), before any PCA.

  A. Midline overlays on raw frames: 8 uniform + the 4 largest-|kappa| frames + the two known
     reflection-merge frames (386, 395). Midline colored head->tail (blue->red).
  B. Body-length time series (stability; depth-parallax breathing is expected, jumps are not).
  C. kappa(s,t) heatmap -- a real undulation must show diagonal stripes (a traveling wave);
     horizontal stripes = standing wave, salt-and-pepper = extraction noise.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


DATA = _P("${ZEF_ROOT}")
OUT = _P("${FISH_ROOT}/demo_out/zef_manifold")

d = np.load(f"{OUT}/curvature_dataset.npz")
kap, mid, valid = d["kappa_bl"], d["midline_px"], d["valid"]
frames, Lpx, t = d["frame"], d["bodylen_px"], d["t_sec"]
gt = np.loadtxt(f"{DATA}/gt/gt.txt", delimiter=",")
rows = {int(r[0]): r for r in gt}

# --- A. overlays ---
peak = np.nanmax(np.abs(kap), axis=1)
peak[~valid] = -1
big = frames[np.argsort(peak)[-4:]]
sample = sorted(set(np.linspace(1, 900, 8, dtype=int)) | set(big.tolist()) | {386, 395})
tiles = []
for fr in sample:
    i = int(fr - frames[0])
    r = rows[int(fr)]
    img = cv2.imread(f"{DATA}/imgT/{int(fr):06d}.jpg")
    if valid[i]:
        m = mid[i]
        for j in range(len(m) - 1):
            c = int(255 * j / (len(m) - 1))
            cv2.line(img, tuple(np.round(m[j]).astype(int)), tuple(np.round(m[j + 1]).astype(int)),
                     (255 - c, 64, c), 2)   # BGR: blue head -> red tail
    cv2.circle(img, (int(r[5]), int(r[6])), 4, (0, 255, 0), -1)
    cx, cy = int(r[5]), int(r[6])
    h = 120
    x0, y0 = max(0, cx - h), max(0, cy - h)
    crop = img[y0:y0 + 2 * h, x0:x0 + 2 * h]
    crop = cv2.copyMakeBorder(crop, 0, 2 * h - crop.shape[0], 0, 2 * h - crop.shape[1],
                              cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
    tag = f"f{fr}" + ("" if valid[i] else " INVALID") + (f" |k|={peak[i]:.1f}" if valid[i] else "")
    cv2.putText(crop, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    tiles.append(crop)
cols = 5
rows_n = (len(tiles) + cols - 1) // cols
grid = np.zeros((rows_n * 360, cols * 360, 3), np.uint8)
for i, tl in enumerate(tiles):
    rr, cc = divmod(i, cols)
    grid[rr * 360:(rr + 1) * 360, cc * 360:(cc + 1) * 360] = tl
cv2.imwrite(f"{OUT}/midline_overlay_grid.png", grid)
print(f"overlays -> {OUT}/midline_overlay_grid.png  (frames {sample})")

# --- B+C: length series + heatmap ---
fig, axes = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[1, 2.2],
                         constrained_layout=True)
ax = axes[0]
ax.plot(t, Lpx, lw=0.8, color="#0072B2")
bad = ~valid
if bad.any():
    ax.plot(t[bad], np.full(bad.sum(), np.nanmin(Lpx) - 4), "x", ms=5, color="#D55E00",
            label=f"rejected ({int(bad.sum())})")
    ax.legend(frameon=False)
ax.set_ylabel("body length (px)")
ax.set_title("Body length per frame (breathing = depth parallax; spikes would be extraction errors)")
ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
v = np.nanpercentile(np.abs(kap), 98)
imk = ax.imshow(kap.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                extent=[t[0], t[-1], 0, 1], interpolation="nearest")
ax.set_xlabel("time (s)")
ax.set_ylabel("body position s (0=head, 1=tail)")
ax.set_title(r"$\kappa(s,t)\cdot BL$ — diagonal stripes = head→tail traveling wave")
fig.colorbar(imk, ax=ax, label=r"$\kappa \cdot BL$", shrink=0.9)
fig.savefig(f"{OUT}/extraction_qc.png", dpi=130)
print(f"length + heatmap -> {OUT}/extraction_qc.png")
