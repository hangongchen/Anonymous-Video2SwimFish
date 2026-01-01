#!/usr/bin/env python3
"""Pre-compute the per-asset (query points, SDF) pool for the implicit refiner.

Single-process, serial. Each asset takes about a second with libigl, so the
full 85-asset run finishes in well under two minutes -- but it lifts the cost
out of the DataLoader hot path so training workers just np.load the cached
pools.

Cache layout (matches ImplicitFlowRefinerDataset._implicit_cache_path):

    <output-dir>/cache_implicit/implicit_v1_<hash>/<asset_id>.npz
        points : float32 [pool_size, 3]   in [0, 1]^3
        sdf    : float32 [pool_size]      unit-cube signed distance

Usage:
    python scripts/experimental/precompute_flow_implicit.py \
        --output-dir outputs/experimental/flow_transformer_refiner_implicit
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from simfishlib.experimental.flow_transformer_refiner import discover_samples
from simfishlib.experimental.flow_transformer_refiner.dataset import _implicit_cache_path_v2
from simfishlib.experimental.flow_transformer_refiner.target import (
    compute_implicit_cache_v2,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Dataset/Fish2VoxelMergedV2Processed")
    p.add_argument("--qwen-coarse-root", default="Dataset/Fish2VoxelMergedV2QwenCoarse")
    p.add_argument("--canonical-root", default="Dataset/Fish2VoxelMergedV2CanonicalViews")
    p.add_argument("--angled-root", default="Dataset/Fish2VoxelMergedV2AngledViews")
    p.add_argument("--video-root", default="Dataset/Fish2VoxelMergedV2Video")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-root", default=None,
                   help="override; defaults to <output-dir>/cache_implicit")
    p.add_argument("--max-assets", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=16384)
    p.add_argument("--num-near-surface", type=int, default=8192)
    p.add_argument("--num-uniform", type=int, default=8192)
    p.add_argument("--near-surface-sigma", type=float, default=0.03)
    p.add_argument("--coarse-resolution", type=int, default=64,
                   help="resolution of the dense TSDF grid baked into the v2 cache")
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--thicken-fins", action="store_true",
                   help="solidify paper-thin mesh components before SDF sampling")
    p.add_argument("--fin-thickness", type=float, default=0.015)
    p.add_argument("--thin-threshold", type=float, default=0.01)
    args = p.parse_args()

    if args.pool_size != args.num_near_surface + args.num_uniform:
        # Keep the cache hash consistent with what the dataset will look up.
        print(f"[precompute] note: pool_size ({args.pool_size}) != near+unif "
              f"({args.num_near_surface}+{args.num_uniform}); using near+unif as the actual count.")

    output_dir = Path(args.output_dir)
    cache_root = Path(args.cache_root) if args.cache_root else output_dir / "cache_implicit"
    cache_root.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        Path(args.data_root),
        resolution=64,
        max_assets=args.max_assets,
        qwen_coarse_root=Path(args.qwen_coarse_root) if args.qwen_coarse_root else None,
        canonical_root=Path(args.canonical_root) if args.canonical_root else None,
        angled_root=Path(args.angled_root) if args.angled_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
    )
    print(f"[precompute] {len(samples)} assets -> {cache_root}  thicken={args.thicken_fins}  "
          f"fin_thickness={args.fin_thickness}  thin_threshold={args.thin_threshold}")

    skipped = computed = 0
    t_start = time.time()
    for i, s in enumerate(samples, 1):
        path = _implicit_cache_path_v2(
            cache_root, s.asset_id,
            pool_size=args.pool_size,
            num_near=args.num_near_surface,
            num_unif=args.num_uniform,
            sigma=args.near_surface_sigma,
            coarse_resolution=args.coarse_resolution,
            truncation=args.truncation,
            thicken=bool(args.thicken_fins),
            fin_thickness=float(args.fin_thickness),
            thin_threshold=float(args.thin_threshold),
        )
        if path.exists():
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        points, sdf, tsdf_grid = compute_implicit_cache_v2(
            s.mesh_path,
            num_near_surface=args.num_near_surface,
            num_uniform=args.num_uniform,
            near_surface_sigma=args.near_surface_sigma,
            grid_resolution=args.coarse_resolution,
            truncation=args.truncation,
            seed=abs(hash(s.asset_id)) & 0xFFFFFFFF,
            thicken_fins=bool(args.thicken_fins),
            fin_thickness=float(args.fin_thickness),
            thin_threshold=float(args.thin_threshold),
        )
        np.savez(
            path,
            points=points.astype(np.float32),
            sdf=sdf.astype(np.float32),
            tsdf_grid=tsdf_grid.astype(np.float32),
        )
        computed += 1
        dt = time.time() - t0
        if i % 5 == 0 or i == len(samples):
            print(f"  [{i}/{len(samples)}] {s.asset_id[:60]}  {dt:.1f}s  total {time.time()-t_start:.0f}s")

    print(f"[precompute] done. computed={computed} skipped={skipped} elapsed={time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
