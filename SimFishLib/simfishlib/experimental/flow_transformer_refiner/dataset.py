"""Dataset for the flow-transformer refiner.

Yields per __getitem__:
  {
    "coarse":    FloatTensor [1, R, R, R]   # Qwen voxel64 (kept at native 64^3 by default)
    "target":    FloatTensor [1, R, R, R]   # GT TSDF in [-1, 1] at the same resolution
    "visual":    FloatTensor [3, H, W] or [K, 3, H, W]
    "asset_id":  str
    "coarse_source": "qwen" | "degraded" | "near_gt"
  }

We **reuse** discovery, asset-level split, visual loading, and degraded-GT
sampling helpers from the V2 module - they're stable utilities, not part of
the V2 model. The unique part here is the TSDF target, computed at the same
resolution as the coarse input (default 64^3, no downsampling).

Caching: GT TSDFs are computed from `asset.obj` once per asset and cached on
disk under `<data_root>/.flow_refiner_cache/tsdf_R{R}_T{T}/<asset>.npy` (path
is overridable via `cache_root`). The cache is plain numpy, so it's safe to
delete to force a recompute.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from simfishlib.experimental.voxel_refiner.dataset import (
    CoarseSourceMix,
    VoxelRefinerSample,
    discover_samples as _discover_samples_v2,
    split_by_asset as _split_by_asset_v2,
    light_degrade,
    _load_pil_image,
    _normalize_image,
    _sample_video_frames,
)
from simfishlib.experimental.voxel_refiner.degradation import DegradationConfig, degrade_voxel
from simfishlib.experimental.flow_transformer_refiner.target import (
    compute_implicit_cache_v2,
    downsample_voxel,
    mesh_to_signed_distance,
    sample_query_points_and_sdf,
    voxel_to_tsdf,
)


# Re-export so callers don't need to know they live in the V2 module.
discover_samples = _discover_samples_v2
split_by_asset = _split_by_asset_v2


@dataclass(frozen=True)
class FlowRefinerConfig:
    target_resolution: int = 64         # TSDF target & coarse-condition resolution
                                        # (64 matches Qwen voxel64 natively; set <64 to ablate)
    truncation: float = 4.0             # voxel-unit clamp for TSDF
    visual_mode: str = "video"
    image_size: int = 224
    num_video_frames: int = 4


def _cache_path(cache_root: Path, asset_id: str, resolution: int, truncation: float) -> Path:
    # Tag prefix "meshsdf_v1" invalidates the older EDT-on-binary caches
    # (which used "tsdf_RxxxxxxxxXX_T...."). Bump the version suffix if the
    # supervision computation changes again.
    tag = hashlib.md5(f"R{resolution}_T{truncation:.3f}".encode()).hexdigest()[:8]
    return cache_root / f"tsdf_meshsdf_v1_R{resolution}_T{truncation:.2f}_{tag}" / f"{asset_id}.npy"


class FlowRefinerDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[VoxelRefinerSample],
        config: FlowRefinerConfig | None = None,
        coarse_mix: CoarseSourceMix | None = None,
        degradation: DegradationConfig | None = None,
        cache_root: Path | None = None,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.config = config or FlowRefinerConfig()
        if self.config.visual_mode not in ("image", "video", "both", "none"):
            raise ValueError(f"unknown visual_mode {self.config.visual_mode!r}")
        self.coarse_mix = coarse_mix or CoarseSourceMix()
        # default degradation matches the source voxel64 grid.
        self.degradation = degradation or DegradationConfig(resolution=64)
        self.cache_root = cache_root
        self.seed = seed
        if cache_root is not None:
            cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    # ----- coarse selection -----------------------------------------------

    def _pick_coarse_source(self, rng: np.random.Generator, sample: VoxelRefinerSample) -> str:
        weights = self.coarse_mix.as_tuple()
        choice = str(rng.choice(("qwen", "degraded", "near_gt"), p=weights))
        if choice == "qwen" and sample.qwen_voxel_path is None:
            sub = np.array([weights[1], weights[2]], dtype=np.float64)
            if sub.sum() <= 0:
                return "degraded"
            return str(rng.choice(("degraded", "near_gt"), p=sub / sub.sum()))
        return choice

    def _load_coarse64(
        self,
        source: str,
        gt_voxel64: np.ndarray,
        sample: VoxelRefinerSample,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if source == "qwen" and sample.qwen_voxel_path is not None:
            arr = np.load(sample.qwen_voxel_path).astype(bool)
            if arr.shape != gt_voxel64.shape:
                raise ValueError(
                    f"qwen voxel shape {arr.shape} != gt {gt_voxel64.shape} for {sample.asset_id}"
                )
            return arr
        if source == "degraded":
            return degrade_voxel(gt_voxel64, rng, self.degradation)
        if source == "near_gt":
            return light_degrade(gt_voxel64, rng)
        raise ValueError(f"unknown coarse source {source!r}")

    # ----- target TSDF ----------------------------------------------------

    def _load_target_tsdf(self, sample: VoxelRefinerSample) -> np.ndarray:
        cfg = self.config
        if self.cache_root is not None:
            cache_path = _cache_path(self.cache_root, sample.asset_id, cfg.target_resolution, cfg.truncation)
            if cache_path.exists():
                arr = np.load(cache_path)
                if arr.shape == (cfg.target_resolution,) * 3:
                    return arr.astype(np.float32)
        # Compute the true mesh signed distance at voxel centers; fall back to
        # the binary EDT path only if the mesh query itself errors out (e.g.
        # corrupt OBJ). The binary path is staircased so the surface comes out
        # axis-aligned -- the SDF path is the supervision we actually want.
        try:
            tsdf = mesh_to_signed_distance(
                sample.mesh_path,
                resolution=cfg.target_resolution,
                truncation=cfg.truncation,
            )
        except Exception:
            from simfishlib.experimental.voxel_refiner.mesh_target import mesh_to_occupancy
            try:
                occ = mesh_to_occupancy(sample.mesh_path, resolution=cfg.target_resolution)
            except Exception:
                voxel64 = np.load(sample.voxel_path).astype(bool)
                occ_soft = downsample_voxel(voxel64, cfg.target_resolution)
                occ = occ_soft > 0.5
            tsdf = voxel_to_tsdf(occ, truncation=cfg.truncation)
        if self.cache_root is not None:
            cache_path = _cache_path(self.cache_root, sample.asset_id, cfg.target_resolution, cfg.truncation)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, tsdf)
        return tsdf

    # ----- visual loading (mirrors V2) ------------------------------------

    def _load_visual(self, sample: VoxelRefinerSample, rng: np.random.Generator) -> torch.Tensor:
        cfg = self.config
        h = cfg.image_size
        if cfg.visual_mode == "none":
            return torch.zeros(3, h, h, dtype=torch.float32)
        if cfg.visual_mode == "image":
            if not sample.image_candidates:
                return torch.zeros(3, h, h, dtype=torch.float32)
            p = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
            arr = _normalize_image(_load_pil_image(p, h))
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if cfg.visual_mode == "video":
            if not sample.video_candidates:
                return torch.zeros(cfg.num_video_frames, 3, h, h, dtype=torch.float32)
            p = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
            frames = _sample_video_frames(p, cfg.num_video_frames, h)
            arr = _normalize_image(frames)
            return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if cfg.visual_mode == "both":
            tensors: list[torch.Tensor] = []
            if sample.image_candidates:
                p = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
                arr = _normalize_image(_load_pil_image(p, h))
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1)[None])
            if sample.video_candidates:
                p = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
                frames = _sample_video_frames(p, cfg.num_video_frames, h)
                arr = _normalize_image(frames)
                tensors.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
            if not tensors:
                return torch.zeros(cfg.num_video_frames + 1, 3, h, h, dtype=torch.float32)
            return torch.cat(tensors, dim=0).contiguous()
        raise ValueError(cfg.visual_mode)

    # ----- __getitem__ ----------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rng = np.random.default_rng(self.seed + index * 9311 + 17)
        cfg = self.config

        gt_voxel64 = np.load(sample.voxel_path).astype(bool)
        coarse_source = self._pick_coarse_source(rng, sample)
        coarse64 = self._load_coarse64(coarse_source, gt_voxel64, sample, rng).astype(np.float32)
        coarse_target = downsample_voxel(coarse64, cfg.target_resolution)
        # rescale soft occupancy from [0,1] into [-1,1] so it lives in the same
        # range as the target TSDF -- helps the input projection see comparable
        # magnitudes from both halves of its concat input.
        coarse_signed = coarse_target * 2.0 - 1.0

        target_tsdf = self._load_target_tsdf(sample)

        visual = self._load_visual(sample, rng)

        return {
            "coarse": torch.from_numpy(coarse_signed.astype(np.float32))[None],
            "target": torch.from_numpy(target_tsdf.astype(np.float32))[None],
            "visual": visual.float(),
            "asset_id": sample.asset_id,
            "coarse_source": coarse_source,
            "voxel_path": str(sample.voxel_path),
            "mesh_path": str(sample.mesh_path),
            "qwen_voxel_path": str(sample.qwen_voxel_path) if sample.qwen_voxel_path else "",
        }


# ---------------------------------------------------------------------------
# Implicit-decoder mode: point-wise SDF supervision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplicitFlowRefinerConfig:
    """Config for the point-wise implicit-decoder dataset.

    `coarse_resolution` is the input voxel grid the transformer sees (defaults
    to 64 to match Qwen's native voxel64). It is also the reference scale used
    to convert the unit-cube SDF into voxel units before truncation -- i.e. an
    SDF magnitude of `truncation` voxels at this resolution saturates to +/-1.

    `pool_size` is the per-asset pre-computed pool size; `num_query_points` is
    the per-step random subset.

    Thickening flags route through to ``compute_implicit_cache_v2`` to solidify
    paper-thin sub-components (fins) before SDF sampling, so they survive the
    grid resolution and become part of the supervision signal.
    """

    coarse_resolution: int = 64
    truncation: float = 4.0
    pool_size: int = 16384
    num_near_surface_pool: int = 8192
    num_uniform_pool: int = 8192
    near_surface_sigma: float = 0.03
    num_query_points: int = 4096
    visual_mode: str = "video"
    image_size: int = 224
    num_video_frames: int = 4
    thicken_fins: bool = False
    fin_thickness: float = 0.015
    thin_threshold: float = 0.01


def _implicit_cache_path(
    cache_root: Path,
    asset_id: str,
    pool_size: int,
    num_near: int,
    num_unif: int,
    sigma: float,
) -> Path:
    tag = hashlib.md5(
        f"pool{pool_size}_near{num_near}_unif{num_unif}_sigma{sigma:.4f}".encode()
    ).hexdigest()[:10]
    return cache_root / f"implicit_v1_{tag}" / f"{asset_id}.npz"


def _implicit_cache_path_v2(
    cache_root: Path,
    asset_id: str,
    *,
    pool_size: int,
    num_near: int,
    num_unif: int,
    sigma: float,
    coarse_resolution: int,
    truncation: float,
    thicken: bool,
    fin_thickness: float,
    thin_threshold: float,
) -> Path:
    """v2 cache: same per-point pool as v1 *plus* tsdf_grid_R; bakes thickening
    params + coarse_resolution + truncation into the hash so caches with
    different SDF supervision don't collide."""
    # "awrap_s4" = CGAL alpha-wrap solidification + 4-way near-surface-heavy point
    # sampling. Bumping this token invalidates caches built by the old thickening /
    # sampling so the new GT geometry + point mixture are recomputed.
    parts = (
        f"pool{pool_size}_near{num_near}_unif{num_unif}_sigma{sigma:.4f}"
        f"_R{coarse_resolution}_T{truncation:.3f}"
        f"_thick{int(bool(thicken))}_ft{fin_thickness:.4f}_thr{thin_threshold:.4f}"
        f"_awrap_s4m"
    )
    tag = hashlib.md5(parts.encode()).hexdigest()[:10]
    return cache_root / f"implicit_v4_{tag}" / f"{asset_id}.npz"


class ImplicitFlowRefinerDataset(Dataset[dict[str, Any]]):
    """Per-asset yield:
       coarse     [1, R, R, R]    signed input voxel grid
       visual     [3, H, W] or [K, 3, H, W]
       query_xyz  [num_query_points, 3] in [0, 1]^3
       target_sdf [num_query_points, 1] normalized SDF in [-1, 1]
       asset_id, coarse_source, paths

    Caches a larger per-asset (points, sdf) pool on disk; samples
    `num_query_points` random indices per __getitem__.
    """

    def __init__(
        self,
        samples: list[VoxelRefinerSample],
        config: ImplicitFlowRefinerConfig | None = None,
        coarse_mix: CoarseSourceMix | None = None,
        degradation: DegradationConfig | None = None,
        cache_root: Path | None = None,
        seed: int = 42,
        carve_augment: bool = False,
    ) -> None:
        self.samples = samples
        self.config = config or ImplicitFlowRefinerConfig()
        # When the "qwen" coarse root is our carve-coarse dataset, each asset dir
        # also ships tilt-augmented carves (coarse64_ag*.npy). carve_augment=True
        # (train only) randomly samples among {base, tilts}; val/eval keep base.
        self.carve_augment = bool(carve_augment)
        # Guard against the old "eval illusion": a ~all-qwen mix with assets that
        # lack a qwen/carve voxel would silently fall back to degraded-GT.
        _mix = coarse_mix or CoarseSourceMix()
        _missing = [s.asset_id for s in samples if s.qwen_voxel_path is None]
        if _mix.as_tuple()[0] >= 0.99 and _missing:
            import warnings
            warnings.warn(
                f"coarse_mix is ~all-qwen but {len(_missing)}/{len(samples)} assets lack a "
                f"qwen/carve voxel -> they fall back to degraded-GT (eval-illusion risk): "
                f"{_missing[:5]}", RuntimeWarning)
        if self.config.visual_mode not in ("image", "video", "both", "none"):
            raise ValueError(f"unknown visual_mode {self.config.visual_mode!r}")
        self.coarse_mix = coarse_mix or CoarseSourceMix()
        self.degradation = degradation or DegradationConfig(resolution=64)
        self.cache_root = cache_root
        self.seed = seed
        if cache_root is not None:
            cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    # -- coarse / visual (same machinery as the dense dataset) --------------

    def _pick_coarse_source(self, rng: np.random.Generator, sample: VoxelRefinerSample) -> str:
        weights = self.coarse_mix.as_tuple()
        choice = str(rng.choice(("qwen", "degraded", "near_gt"), p=weights))
        if choice == "qwen" and sample.qwen_voxel_path is None:
            sub = np.array([weights[1], weights[2]], dtype=np.float64)
            if sub.sum() <= 0:
                return "degraded"
            return str(rng.choice(("degraded", "near_gt"), p=sub / sub.sum()))
        return choice

    def _load_coarse64(
        self,
        source: str,
        gt_voxel64: np.ndarray,
        sample: VoxelRefinerSample,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if source == "qwen" and sample.qwen_voxel_path is not None:
            path = sample.qwen_voxel_path
            if self.carve_augment:
                # sibling tilt-augmented carves live next to the base file
                path = self._pick_carve_variant(path, rng)
            arr = np.load(path).astype(bool)
            if arr.shape != gt_voxel64.shape:
                raise ValueError(
                    f"qwen voxel shape {arr.shape} != gt {gt_voxel64.shape} for {sample.asset_id}"
                )
            return arr
        if source == "degraded":
            return degrade_voxel(gt_voxel64, rng, self.degradation)
        if source == "near_gt":
            return light_degrade(gt_voxel64, rng)
        raise ValueError(f"unknown coarse source {source!r}")

    @staticmethod
    def _pick_carve_variant(base_path: Path, rng: np.random.Generator) -> Path:
        variants = [base_path] + sorted(base_path.parent.glob("coarse64_ag*.npy"))
        return variants[int(rng.integers(len(variants)))]

    def _load_visual(self, sample: VoxelRefinerSample, rng: np.random.Generator) -> torch.Tensor:
        cfg = self.config
        h = cfg.image_size
        if cfg.visual_mode == "none":
            return torch.zeros(3, h, h, dtype=torch.float32)
        if cfg.visual_mode == "image":
            if not sample.image_candidates:
                return torch.zeros(3, h, h, dtype=torch.float32)
            p = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
            arr = _normalize_image(_load_pil_image(p, h))
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if cfg.visual_mode == "video":
            if not sample.video_candidates:
                return torch.zeros(cfg.num_video_frames, 3, h, h, dtype=torch.float32)
            p = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
            frames = _sample_video_frames(p, cfg.num_video_frames, h)
            arr = _normalize_image(frames)
            return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if cfg.visual_mode == "both":
            tensors: list[torch.Tensor] = []
            if sample.image_candidates:
                p = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
                arr = _normalize_image(_load_pil_image(p, h))
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1)[None])
            if sample.video_candidates:
                p = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
                frames = _sample_video_frames(p, cfg.num_video_frames, h)
                arr = _normalize_image(frames)
                tensors.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
            if not tensors:
                return torch.zeros(cfg.num_video_frames + 1, 3, h, h, dtype=torch.float32)
            return torch.cat(tensors, dim=0).contiguous()
        raise ValueError(cfg.visual_mode)

    # -- query-pool cache ---------------------------------------------------

    def _v2_cache_path(self, asset_id: str) -> Path:
        cfg = self.config
        assert self.cache_root is not None
        return _implicit_cache_path_v2(
            self.cache_root, asset_id,
            pool_size=cfg.pool_size,
            num_near=cfg.num_near_surface_pool,
            num_unif=cfg.num_uniform_pool,
            sigma=cfg.near_surface_sigma,
            coarse_resolution=cfg.coarse_resolution,
            truncation=cfg.truncation,
            thicken=cfg.thicken_fins,
            fin_thickness=cfg.fin_thickness,
            thin_threshold=cfg.thin_threshold,
        )

    def _load_v2_bundle(
        self, sample: VoxelRefinerSample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (points, sdf_unit_cube, tsdf_grid_R). Reads the v2 cache if
        present, otherwise computes everything from the (possibly thickened)
        mesh in one libigl pass and writes the cache."""
        cfg = self.config
        if self.cache_root is not None:
            path = self._v2_cache_path(sample.asset_id)
            if path.exists():
                data = np.load(path)
                return data["points"], data["sdf"], data["tsdf_grid"]
        points, sdf, tsdf_grid = compute_implicit_cache_v2(
            sample.mesh_path,
            num_near_surface=cfg.num_near_surface_pool,
            num_uniform=cfg.num_uniform_pool,
            near_surface_sigma=cfg.near_surface_sigma,
            grid_resolution=cfg.coarse_resolution,
            truncation=cfg.truncation,
            seed=abs(hash(sample.asset_id)) & 0xFFFFFFFF,
            thicken_fins=cfg.thicken_fins,
            fin_thickness=cfg.fin_thickness,
            thin_threshold=cfg.thin_threshold,
        )
        if self.cache_root is not None:
            path = self._v2_cache_path(sample.asset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, points=points, sdf=sdf, tsdf_grid=tsdf_grid)
        return points, sdf, tsdf_grid

    def _load_query_pool(self, sample: VoxelRefinerSample) -> tuple[np.ndarray, np.ndarray]:
        """Legacy v1 entry point. Kept for the older non-thickened code path.

        Note: the v2 dataset emit always uses ``_load_v2_bundle``; this stays
        here in case someone still uses an old ``ImplicitFlowRefinerConfig``
        with default flags via an external caller.
        """
        cfg = self.config
        if self.cache_root is not None:
            path = _implicit_cache_path(
                self.cache_root,
                sample.asset_id,
                cfg.pool_size,
                cfg.num_near_surface_pool,
                cfg.num_uniform_pool,
                cfg.near_surface_sigma,
            )
            if path.exists():
                data = np.load(path)
                return data["points"], data["sdf"]
        points, sdf = sample_query_points_and_sdf(
            sample.mesh_path,
            num_near_surface=cfg.num_near_surface_pool,
            num_uniform=cfg.num_uniform_pool,
            near_surface_sigma=cfg.near_surface_sigma,
            seed=abs(hash(sample.asset_id)) & 0xFFFFFFFF,
            thicken_fins=cfg.thicken_fins,
            fin_thickness=cfg.fin_thickness,
            thin_threshold=cfg.thin_threshold,
        )
        if self.cache_root is not None:
            path = _implicit_cache_path(
                self.cache_root,
                sample.asset_id,
                cfg.pool_size,
                cfg.num_near_surface_pool,
                cfg.num_uniform_pool,
                cfg.near_surface_sigma,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, points=points, sdf=sdf)
        return points, sdf

    # -- __getitem__ --------------------------------------------------------

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rng = np.random.default_rng(self.seed + index * 9311 + 17)
        cfg = self.config

        gt_voxel64 = np.load(sample.voxel_path).astype(bool)
        coarse_source = self._pick_coarse_source(rng, sample)
        coarse64 = self._load_coarse64(coarse_source, gt_voxel64, sample, rng).astype(np.float32)
        coarse_signed = coarse64 * 2.0 - 1.0

        # v2 bundle: per-point pool + dense GT TSDF on the coarse grid
        points, sdf_unit_cube, tsdf_grid = self._load_v2_bundle(sample)
        pool_n = len(points)
        K = cfg.num_query_points
        idx = rng.choice(pool_n, size=K, replace=(K > pool_n))
        q_xyz = points[idx]
        q_sdf_uc = sdf_unit_cube[idx]
        # SDF is in unit-cube coordinates (one unit = one cube edge). Convert to
        # voxel-of-coarse_resolution units, then normalize by truncation and clip
        # to [-1, 1] -- same convention as the dense TSDF target.
        R_ref = cfg.coarse_resolution
        q_sdf = np.clip(q_sdf_uc * R_ref / max(cfg.truncation, 1e-6), -1.0, 1.0)

        visual = self._load_visual(sample, rng)

        return {
            "coarse": torch.from_numpy(coarse_signed.astype(np.float32))[None],
            "target_tsdf_grid": torch.from_numpy(tsdf_grid.astype(np.float32))[None],
            "query_xyz": torch.from_numpy(q_xyz.astype(np.float32)),
            "target_sdf": torch.from_numpy(q_sdf.astype(np.float32))[:, None],
            "visual": visual.float(),
            "asset_id": sample.asset_id,
            "coarse_source": coarse_source,
            "voxel_path": str(sample.voxel_path),
            "mesh_path": str(sample.mesh_path),
            "qwen_voxel_path": str(sample.qwen_voxel_path) if sample.qwen_voxel_path else "",
        }
