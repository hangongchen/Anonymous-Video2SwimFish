#!/usr/bin/env python
"""Build a REAL-fish point-cloud sequence FROM VIDEO for the Salmon-IL-Water task.

Runs the SimFishLib two-view silhouette carve (segment -> mask -> carve) for each frame of a
3D-ZeF sequence and turns the carved 64^3 occupancy voxel into a point cloud (occupied voxel
centres, centred and scaled to a real body length). The resulting per-frame `.npy` clouds are the
imitation target for the IL task's `real_pointcloud` (canonical-frame chamfer) mode -- i.e. the
policy is rewarded for reproducing the REAL fish's body shape / undulation as seen in the video.

Masks must already exist (run scripts/segment_zef_fish.py --save-mask first).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# import the carve stage from SimFishLib
sys.path.insert(0, str(Path(_P("${FISH_ROOT}/SimFishLib/scripts"))))
import carve_voxel_from_two_views as C  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ZebraFish-01-sub")
    ap.add_argument("--mask-root", default=_P("${FISH_ROOT}/demo_out/zef_video/seg"))
    ap.add_argument("--gt-root", default=_P("${FISH_ROOT}/demo_out/zef_video"))
    ap.add_argument("--frames", default="1000:1598:2", help="start:stop:step (inclusive stop)")
    ap.add_argument("--frames-file", default=None, help="file with one frame number per line "
                    "(non-contiguous; overrides --frames)")
    ap.add_argument("--prefix", default="zefvid", help="output cloud filename prefix")
    ap.add_argument("--out", default=_P("${FISH_ROOT}/demo_out/zef_video/pointclouds"))
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--body-length-m", type=float, default=0.5)
    args = ap.parse_args()

    if args.frames_file:
        frames = [int(x) for x in Path(args.frames_file).read_text().split()]
    else:
        a, b, s = (int(x) for x in args.frames.split(":"))
        frames = list(range(a, b + 1, s))
    gt = np.loadtxt(Path(args.gt_root) / args.seq / "gt" / "gt.txt", delimiter=",")
    mroot = Path(args.mask_root) / args.seq
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    carved_frames = []
    kept = 0
    for k, fr in enumerate(frames):
        row = gt[gt[:, 0] == fr]
        if len(row) == 0:
            continue
        row = row[0]
        try:
            mF = C.load_mask(str(mroot / "imgF" / "mask" / f"{fr:06d}.png"))
            mT = C.load_mask(str(mroot / "imgT" / "mask" / f"{fr:06d}.png"))
        except FileNotFoundError:
            continue
        if mF.sum() < 20 or mT.sum() < 20:   # segmentation missed the fish this frame
            continue
        headF = (row[C.VIEW_COLS["imgF"]["cx"]], row[C.VIEW_COLS["imgF"]["cy"]])
        headT = (row[C.VIEW_COLS["imgT"]["cx"]], row[C.VIEW_COLS["imgT"]["cy"]])
        front = C.rotate_upright(mF, headF)
        top = C.rotate_upright(mT, headT)
        occ = C.carve(front, top, grid=args.grid, length=None)
        if not occ.any():
            continue
        pts = np.argwhere(occ).astype(np.float32)     # (M,3) voxel indices [X,Y,Z]
        # centre, and scale so the long (X) axis spans the real body length
        pts -= pts.mean(0)
        long_span = float(pts[:, 0].max() - pts[:, 0].min()) or 1.0
        pts *= args.body_length_m / long_span
        np.save(out / f"{args.prefix}_{kept:04d}.npy", pts)
        carved_frames.append(int(fr))
        kept += 1
        if (k + 1) % 50 == 0:
            print(f"[carve] {k+1}/{len(frames)} frames done, {kept} clouds, "
                  f"last M={pts.shape[0]}", flush=True)

    np.save(out / "carved_frames.npy", np.array(carved_frames, dtype=np.int64))
    print(f"[build_zef_video_pointclouds] wrote {kept} clouds + carved_frames.npy -> {out}", flush=True)
    if kept:
        sample = np.load(sorted(out.glob(f'{args.prefix}_*.npy'))[0])
        print(f"  sample cloud shape {sample.shape}, bbox min {sample.min(0).round(3)} "
              f"max {sample.max(0).round(3)}", flush=True)


if __name__ == "__main__":
    main()
