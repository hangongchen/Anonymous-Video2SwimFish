#!/usr/bin/env python
"""Build an inference-consistent COARSE dataset for the flow refiner.

The refiner was trained on `degraded GT` coarse (corrupt-then-restore), but at
inference we feed a 2-view silhouette CARVE (rectangular cross-sections, no
roundness). This script closes that train/inference gap: for every asset it
takes the GT mesh (already realigned to canonical: X=length, Y=width, Z=height),
normalizes it to the unit cube [0,1]^3 (the EXACT frame of voxel.npy / the SDF
GT), and produces the 2-view visual hull at 64^3 -- the same operation as the
inference carve, but on the GT geometry so it's perfectly registered.

Head-tail axis: already canonicalized in asset.obj, so no PCA/flip needed.
Augmentation: also carve a few small yaw/pitch-tilted copies (like the fish not
being perfectly broadside at inference).

Output per asset -> <out>/<asset_id>/:
  qwen_voxel64.npy       base carve (plugs into the existing "qwen" coarse slot)
  coarse64_ag{k}.npy     tilt-augmented carves
  check.png              GT(green) vs base-carve(red) projection overlay

Usage:
  python scripts/experimental/build_carve_coarse_dataset.py \
      --data-root Dataset/Fish2VoxelMergedV2Processed \
      --out Dataset/Fish2VoxelMergedV2CarveCoarse
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import trimesh
from scipy.ndimage import binary_fill_holes
from simfishlib.experimental.voxel_refiner.mesh_target import load_normalized_mesh, mesh_to_occupancy

# base + small tilts: (yaw about Z / seen in top view, pitch about Y / tilts side view)
TILTS = [(0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (0.0, 8.0), (0.0, -8.0), (7.0, 6.0)]


def rotated(mesh, yaw_deg, pitch_deg):
    if yaw_deg == 0.0 and pitch_deg == 0.0:
        return mesh
    c = [0.5, 0.5, 0.5]
    m = mesh.copy()
    Rz = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1], c)
    Ry = trimesh.transformations.rotation_matrix(np.radians(pitch_deg), [0, 1, 0], c)
    m.apply_transform(Rz @ Ry)
    return m


def solid_occ3(mesh, G):
    """Solid G^3 occupancy of a (unit-cube) mesh via filled voxelization; falls
    back to dense surface points if the mesh isn't fillable (open fins)."""
    try:
        pts = np.asarray(mesh.voxelized(pitch=1.0 / G).fill().points)
        if len(pts) < 50:
            raise ValueError("empty voxelization")
    except Exception:
        pts, _ = trimesh.sample.sample_surface(mesh, 400_000)
    ix = np.clip((pts[:, 0] * G).astype(int), 0, G - 1)
    iy = np.clip((pts[:, 1] * G).astype(int), 0, G - 1)
    iz = np.clip((pts[:, 2] * G).astype(int), 0, G - 1)
    occ3 = np.zeros((G, G, G), bool)
    occ3[ix, iy, iz] = True
    return occ3


def hull_from_occ3(occ3):
    """2-view visual hull (unit-cube frame) from a solid occupancy.

    side silhouette = projection along Y (width) -> (X,Z)
    top  silhouette = projection along Z (height) -> (X,Y)
    occ(x,y,z) = top(x,y) AND side(x,z)   -- this is a proper visual hull, so it
    contains the source shape by construction (containment == 1.0 for the base,
    where occ3 is the GT voxelization). Returns (occ, side[X,Z], top[X,Y]).
    """
    side = binary_fill_holes(occ3.any(1))   # (X,Z)
    top = binary_fill_holes(occ3.any(2))     # (X,Y)
    occ = top[:, :, None] & side[:, None, :]
    return occ, side, top


def save_view_png(sil_xz_or_xy, path):
    """Save a black-bg, white-fish view PNG (rows = vertical axis, image top-up)."""
    img = np.flipud(sil_xz_or_xy.T.astype(np.uint8) * 255)  # (X,V) -> rows=V up
    cv2.imwrite(str(path), cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST))


def overlay(gt, carve, path):
    def proj(g, c, ax, lbl):
        G = g.any(ax).astype(np.uint8); C = c.any(ax).astype(np.uint8)
        img = np.zeros((*G.shape, 3), np.uint8)
        img[..., 1] = G * 255; img[..., 2] = C * 255
        img = img.transpose(1, 0, 2) if ax != 0 else img
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, lbl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img
    row = np.hstack([proj(gt, carve, 2, "top XY G=gt R=carve"),
                     proj(gt, carve, 1, "side XZ"), proj(gt, carve, 0, "head YZ")])
    cv2.imwrite(str(path), row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Dataset/Fish2VoxelMergedV2Processed")
    ap.add_argument("--out", default="Dataset/Fish2VoxelMergedV2CarveCoarse")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="process only first N assets")
    ap.add_argument("--assets", nargs="*", default=None, help="specific asset ids")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assets = sorted(p for p in root.glob("asset_*") if (p / "asset.obj").exists() and (p / "voxel.npy").exists())
    if args.assets:
        assets = [p for p in assets if p.name in set(args.assets)]
    if args.limit:
        assets = assets[:args.limit]
    print(f"{len(assets)} assets -> {out}")

    contain_stats, iou_stats = [], []
    for i, a in enumerate(assets):
        try:
            mesh = load_normalized_mesh(a / "asset.obj")
        except Exception as exc:
            print(f"[{i+1}/{len(assets)}] {a.name}: LOAD FAIL {exc}")
            continue
        d = out / a.name; d.mkdir(parents=True, exist_ok=True)
        # base occ3 = the GT voxelization itself -> hull contains GT exactly.
        gt = mesh_to_occupancy(a / "asset.obj", resolution=args.grid)
        for k, (yaw, pitch) in enumerate(TILTS):
            occ3 = gt if k == 0 else solid_occ3(rotated(mesh, yaw, pitch), args.grid)
            occ, side, top = hull_from_occ3(occ3)
            name = "qwen_voxel64.npy" if k == 0 else f"coarse64_ag{k}.npy"
            np.save(d / name, occ)
            if k == 0:  # save the black-bg side/top views that produced the base carve
                save_view_png(side, d / "side.png")
                save_view_png(top, d / "top.png")
        base = np.load(d / "qwen_voxel64.npy")
        contain = (base & gt).sum() / max(1, gt.sum())
        iou = (base & gt).sum() / max(1, (base | gt).sum())
        contain_stats.append(contain); iou_stats.append(iou)
        if i < 4:
            overlay(gt, base, d / "check.png")
        print(f"[{i+1}/{len(assets)}] {a.name}: base_occ={int(base.sum())} gt={int(gt.sum())} "
              f"contain={contain:.3f} iou={iou:.3f}", flush=True)

    if contain_stats:
        print(f"\n== containment mean={np.mean(contain_stats):.3f} min={np.min(contain_stats):.3f} "
              f"| IoU mean={np.mean(iou_stats):.3f} ==")


if __name__ == "__main__":
    main()
