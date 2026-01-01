"""Extract the ZeF REFERENCE TRAJECTORY for the locomotion-fidelity evaluation.

DATA CONSTRAINT (hard requirement): this script reads ONLY
`demo_out/zef_manifold/curvature_dataset.npz` -- the exact file the PCA basis was fitted
from. No other ZeF videos, frames, or trajectories are loaded. The reference trajectory is
therefore a strict subset of the data inside the learned representation.

Definitions (documented for the report):
  - source: ZebraFish-05 top-view video, the same 900-frame extraction (60 fps) used for PCA;
    only the `valid` frames are used, split into contiguous runs (gaps break segments).
  - position p_zef(t): CENTROID of the 100 midline points (~mid-body) -- chosen to match the
    agent's measured point (its articulation root is a mid-body bone). 2D top-view.
  - heading psi_zef(t): direction from the midline point at s=0.30 to the HEAD point (s=0),
    in the y-FLIPPED (right-handed) frame -- the same frame convention the curvature
    extraction used, so turning signs match the simulation.
  - units: positions divided by the per-frame body length in px (median over the segment for
    stability) -> everything in BL. fps = 60.
  - smoothing: 5-frame moving average on p (tracking pixel noise), heading from smoothed pts.

Waypoint pairs: for start frames on a uniform stride grid inside each valid run, and for each
displacement horizon D in {0.5, 1.0, 2.0} BL, the target frame is the FIRST future frame in
the same run with ||p(t_j) - p(t_i)|| >= D. Each (t_i, t_j) pair is one evaluation sample;
the ground-truth trajectory is p[t_i..t_j].

Output: demo_out/trajectory_fidelity/zef_reference.npz
"""

from pathlib import Path

import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = Path(_P("${FISH_ROOT}/demo_out/trajectory_fidelity"))
OUT.mkdir(parents=True, exist_ok=True)
SRC = _P("${FISH_ROOT}/demo_out/zef_manifold/curvature_dataset.npz")
d = np.load(SRC)
mid = d["midline_px"].astype(np.float64)          # (900, 100, 2) image px, head at index 0
valid = d["valid"].astype(bool)
fps = float(d["fps"])
blpx = d["bodylen_px"].astype(np.float64)

# y-flip to the right-handed frame used by the curvature extraction
mid = mid.copy()
mid[:, :, 1] = -mid[:, :, 1]

BLPX = float(np.nanmedian(blpx[valid]))
p = np.nanmean(mid, axis=1) / BLPX                # centroid, in BL units
# 5-frame moving average (valid-run-local smoothing applied below per segment)
head = mid[:, 0, :] / BLPX
s30 = mid[:, 30, :] / BLPX
psi = np.arctan2(head[:, 1] - s30[:, 1], head[:, 0] - s30[:, 0])

# contiguous valid runs
runs = []
i = 0
while i < len(valid):
    if valid[i]:
        j = i
        while j < len(valid) and valid[j]:
            j += 1
        if j - i >= 60:
            runs.append((i, j))
        i = j
    else:
        i += 1
print("valid runs:", runs)

def smooth(x, k=5):
    ker = np.ones(k) / k
    out = x.copy()
    for c in range(x.shape[1]):
        out[k // 2:-(k // 2), c] = np.convolve(x[:, c], ker, mode="valid")
    return out

HORIZONS = {"short": 0.5, "medium": 1.0, "long": 2.0}
STRIDE = 30                                        # start every 0.5 s
samples = []                                       # (t_i, t_j, horizon_idx)
for (a, b) in runs:
    p[a:b] = smooth(p[a:b])
    for ti in range(a, b - 30, STRIDE):
        dist = np.linalg.norm(p[ti:b] - p[ti], axis=1)
        for hi, (hname, D) in enumerate(HORIZONS.items()):
            hit = np.nonzero(dist >= D)[0]
            if len(hit) > 0:
                samples.append((ti, ti + int(hit[0]), hi))

samples = np.array(samples, dtype=np.int64)
disp = np.linalg.norm(p[samples[:, 1]] - p[samples[:, 0]], axis=1)
dur = (samples[:, 1] - samples[:, 0]) / fps
print(f"samples: {len(samples)} "
      f"(short={int((samples[:,2]==0).sum())}, medium={int((samples[:,2]==1).sum())}, "
      f"long={int((samples[:,2]==2).sum())})")
print(f"displacements BL: {np.round(np.percentile(disp, [0,50,100]), 2)}; "
      f"durations s: {np.round(np.percentile(dur, [0,50,100]), 2)}")
print(f"total ZeF travel run-wise:",
      [round(float(np.sum(np.linalg.norm(np.diff(p[a:b], axis=0), axis=1))), 1) for a, b in runs], "BL")

np.savez(OUT / "zef_reference.npz",
         p_bl=p.astype(np.float32), psi=psi.astype(np.float32), valid=valid,
         runs=np.array(runs), samples=samples, fps=fps, bl_px=BLPX,
         horizons=np.array(list(HORIZONS.values())),
         horizon_names=np.array(list(HORIZONS.keys())),
         source=SRC)
print(f"wrote {OUT/'zef_reference.npz'}")
