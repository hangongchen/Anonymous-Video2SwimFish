"""V2 implicit-occupancy refiner dataset.

Each __getitem__ returns:
  {
    "coarse":   FloatTensor [1, R, R, R]   # qwen / degraded / near-GT coarse
    "visual":   FloatTensor [3, H, W] or [K, 3, H, W]
    "queries":  FloatTensor [N, 3]         # mesh-space points in [0,1]^3
    "labels":   FloatTensor [N]            # inside-mesh in {0,1}
    "gt_voxel": FloatTensor [1, R, R, R]   # kept for cross-comparison with Qwen
    "asset_id": str
    "coarse_source": "qwen" | "degraded" | "near_gt"
    ...
  }

Discovery walks --data-root (Fish2VoxelMergedV2Processed) and keeps assets
whose voxel.npy is the right shape AND whose asset.obj loads.
Splits are computed at the asset level, never image-level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from simfishlib.experimental.voxel_refiner.degradation import DegradationConfig, degrade_voxel
from simfishlib.experimental.voxel_refiner.mesh_target import sample_mesh_occupancy_points


COARSE_SOURCES = ("qwen", "degraded", "near_gt")


@dataclass(frozen=True)
class CoarseSourceMix:
    qwen: float = 0.5
    degraded: float = 0.4
    near_gt: float = 0.1

    def as_tuple(self) -> tuple[float, float, float]:
        total = self.qwen + self.degraded + self.near_gt
        if total <= 0:
            raise ValueError("coarse-source-mix probabilities sum to 0")
        return (self.qwen / total, self.degraded / total, self.near_gt / total)

    @classmethod
    def parse(cls, spec: str | None) -> "CoarseSourceMix":
        if spec is None:
            return cls()
        parts = [float(x) for x in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--coarse-source-mix expected 'q,d,n', got {spec!r}")
        return cls(qwen=parts[0], degraded=parts[1], near_gt=parts[2])


@dataclass(frozen=True)
class VoxelRefinerSample:
    asset_id: str
    voxel_path: Path
    mesh_path: Path
    qwen_voxel_path: Path | None
    image_candidates: tuple[Path, ...] = field(default_factory=tuple)
    video_candidates: tuple[Path, ...] = field(default_factory=tuple)


def _list_canonical(canonical_root: Path | None, asset_id: str) -> list[Path]:
    if canonical_root is None:
        return []
    asset_dir = canonical_root / asset_id
    if not asset_dir.exists():
        return []
    found: list[Path] = []
    for name in ("left.png", "right.png", "top.png", "bottom.png", "head.png", "tail.png"):
        p = asset_dir / name
        if p.exists():
            found.append(p)
    return found


def _list_angled(angled_root: Path | None, asset_id: str) -> list[Path]:
    if angled_root is None:
        return []
    asset_dir = angled_root / asset_id
    if not asset_dir.exists():
        return []
    return sorted(asset_dir.glob("angle_*.png"))


def _list_processed_images(processed_root: Path, asset_id: str) -> list[Path]:
    asset_dir = processed_root / asset_id
    if not asset_dir.exists():
        return []
    return sorted(asset_dir.glob("image_l*_*.png"))


def _list_videos(video_root: Path | None, asset_id: str) -> list[Path]:
    if video_root is None:
        return []
    asset_dir = video_root / asset_id
    if not asset_dir.exists():
        return []
    return sorted(asset_dir.glob("swim*.mp4"))


def discover_samples(
    data_root: Path,
    resolution: int = 64,
    max_assets: int | None = None,
    qwen_coarse_root: Path | None = None,
    canonical_root: Path | None = None,
    angled_root: Path | None = None,
    video_root: Path | None = None,
    require_mesh: bool = True,
) -> list[VoxelRefinerSample]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    samples: list[VoxelRefinerSample] = []
    skipped: list[str] = []
    asset_dirs = sorted(p for p in data_root.glob("asset_*") if p.is_dir())
    if max_assets is not None:
        asset_dirs = asset_dirs[:max_assets]
    for asset_dir in asset_dirs:
        voxel_path = asset_dir / "voxel.npy"
        mesh_path = asset_dir / "asset.obj"
        if not voxel_path.exists():
            skipped.append(f"{asset_dir.name}: missing voxel.npy")
            continue
        if require_mesh and not mesh_path.exists():
            skipped.append(f"{asset_dir.name}: missing asset.obj")
            continue
        try:
            shape = tuple(np.load(voxel_path, mmap_mode="r").shape)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{asset_dir.name}: voxel.npy unreadable: {exc}")
            continue
        if shape != (resolution, resolution, resolution):
            skipped.append(f"{asset_dir.name}: voxel shape {shape} != {(resolution,) * 3}")
            continue

        qwen_path: Path | None = None
        if qwen_coarse_root is not None:
            cand = qwen_coarse_root / asset_dir.name / "qwen_voxel64.npy"
            if cand.exists():
                qwen_path = cand

        images = tuple(
            _list_canonical(canonical_root, asset_dir.name)
            + _list_angled(angled_root, asset_dir.name)
            + _list_processed_images(data_root, asset_dir.name)
        )
        videos = tuple(_list_videos(video_root, asset_dir.name))

        samples.append(
            VoxelRefinerSample(
                asset_id=asset_dir.name,
                voxel_path=voxel_path,
                mesh_path=mesh_path,
                qwen_voxel_path=qwen_path,
                image_candidates=images,
                video_candidates=videos,
            )
        )

    if not samples:
        message = f"No valid voxel-refiner samples found under {data_root}."
        if skipped:
            message += " First skipped entries: " + "; ".join(skipped[:8])
        raise RuntimeError(message)
    return samples


def split_by_asset(
    samples: Sequence[VoxelRefinerSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[VoxelRefinerSample], list[VoxelRefinerSample]]:
    assets = sorted({sample.asset_id for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(assets)
    val_count = max(1, int(round(len(assets) * val_ratio))) if len(assets) > 1 else 0
    val_assets = set(assets[:val_count])
    train = [s for s in samples if s.asset_id not in val_assets]
    val = [s for s in samples if s.asset_id in val_assets]
    if not train and val:
        train, val = val, []
    return train, val


def _load_pil_image(path: Path, image_size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _normalize_image(rgb_uint8: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5


def _sample_video_frames(path: Path, num_frames: int, image_size: int) -> np.ndarray:
    """Uniformly sample `num_frames` frames from a video. Returns [K, H, W, 3] uint8."""
    try:
        import cv2  # type: ignore
    except Exception:
        frame = _load_pil_image(path, image_size)
        return np.repeat(frame[None, ...], num_frames, axis=0)

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    if total <= 0:
        cap.release()
        frame = _load_pil_image(path, image_size)
        return np.repeat(frame[None, ...], num_frames, axis=0)
    idxs = np.linspace(0, total - 1, num_frames, dtype=np.int64)
    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        frames = [np.zeros((image_size, image_size, 3), dtype=np.uint8)] * num_frames
    return np.stack(frames, axis=0)


def light_degrade(voxel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small perturbations of the GT - for stabilization mixed in at low weight."""
    voxel = voxel.astype(bool)
    voxel = voxel & (rng.random(voxel.shape) >= rng.uniform(0.005, 0.03))
    if rng.random() < 0.3:
        try:
            from scipy import ndimage

            struct = ndimage.generate_binary_structure(3, 1)
            op = rng.choice(["dilate", "erode"])
            if op == "dilate":
                voxel = ndimage.binary_dilation(voxel, structure=struct, iterations=1)
            else:
                voxel = ndimage.binary_erosion(voxel, structure=struct, iterations=1)
        except Exception:
            pass
    return voxel.astype(bool)


class VoxelRefinerV2Dataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[VoxelRefinerSample],
        resolution: int = 64,
        visual_mode: str = "video",
        image_size: int = 224,
        num_video_frames: int = 4,
        num_queries: int = 4096,
        surface_fraction: float = 0.5,
        surface_jitter: float = 0.015,
        coarse_mix: CoarseSourceMix | None = None,
        degradation: DegradationConfig | None = None,
        seed: int = 42,
        debug_save_dir: Path | None = None,
    ) -> None:
        if visual_mode not in ("image", "video", "both", "none"):
            raise ValueError(f"unknown visual_mode {visual_mode!r}")
        self.samples = samples
        self.resolution = resolution
        self.visual_mode = visual_mode
        self.image_size = image_size
        self.num_video_frames = num_video_frames
        self.num_queries = num_queries
        self.surface_fraction = surface_fraction
        self.surface_jitter = surface_jitter
        self.coarse_mix = coarse_mix or CoarseSourceMix()
        self.degradation = degradation or DegradationConfig(resolution=resolution)
        self.seed = seed
        self.debug_save_dir = debug_save_dir
        if debug_save_dir is not None:
            debug_save_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_coarse_source(self, rng: np.random.Generator, sample: VoxelRefinerSample) -> str:
        weights = self.coarse_mix.as_tuple()
        choice = rng.choice(COARSE_SOURCES, p=weights)
        if choice == "qwen" and sample.qwen_voxel_path is None:
            sub = np.array([weights[1], weights[2]], dtype=np.float64)
            if sub.sum() <= 0:
                return "degraded"
            return str(rng.choice(["degraded", "near_gt"], p=sub / sub.sum()))
        return str(choice)

    def _load_coarse(
        self,
        source: str,
        gt_voxel: np.ndarray,
        sample: VoxelRefinerSample,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if source == "qwen" and sample.qwen_voxel_path is not None:
            arr = np.load(sample.qwen_voxel_path).astype(bool)
            if arr.shape != gt_voxel.shape:
                raise ValueError(
                    f"qwen voxel shape {arr.shape} != gt {gt_voxel.shape} for {sample.asset_id}"
                )
            return arr
        if source == "degraded":
            return degrade_voxel(gt_voxel, rng, self.degradation)
        if source == "near_gt":
            return light_degrade(gt_voxel, rng)
        raise ValueError(f"unknown coarse source {source!r}")

    def _load_visual(self, sample: VoxelRefinerSample, rng: np.random.Generator) -> torch.Tensor:
        h = self.image_size
        if self.visual_mode == "none":
            return torch.zeros(3, h, h, dtype=torch.float32)
        if self.visual_mode == "image":
            if not sample.image_candidates:
                return torch.zeros(3, h, h, dtype=torch.float32)
            img_path = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
            arr = _normalize_image(_load_pil_image(img_path, h))
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.visual_mode == "video":
            if not sample.video_candidates:
                return torch.zeros(self.num_video_frames, 3, h, h, dtype=torch.float32)
            vid_path = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
            frames = _sample_video_frames(vid_path, self.num_video_frames, h)
            arr = _normalize_image(frames)
            return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if self.visual_mode == "both":
            tensors: list[torch.Tensor] = []
            if sample.image_candidates:
                img_path = Path(sample.image_candidates[int(rng.integers(0, len(sample.image_candidates)))])
                arr = _normalize_image(_load_pil_image(img_path, h))
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1)[None])
            if sample.video_candidates:
                vid_path = Path(sample.video_candidates[int(rng.integers(0, len(sample.video_candidates)))])
                frames = _sample_video_frames(vid_path, self.num_video_frames, h)
                arr = _normalize_image(frames)
                tensors.append(torch.from_numpy(arr).permute(0, 3, 1, 2))
            if not tensors:
                return torch.zeros(self.num_video_frames + 1, 3, h, h, dtype=torch.float32)
            return torch.cat(tensors, dim=0).contiguous()
        raise ValueError(self.visual_mode)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rng = np.random.default_rng(self.seed + index * 9301 + 49297)
        gt_voxel = np.load(sample.voxel_path).astype(bool)
        coarse_source = self._pick_coarse_source(rng, sample)
        coarse = self._load_coarse(coarse_source, gt_voxel, sample, rng).astype(bool)
        visual = self._load_visual(sample, rng)

        queries, labels = sample_mesh_occupancy_points(
            sample.mesh_path,
            count=self.num_queries,
            rng=rng,
            surface_fraction=self.surface_fraction,
            surface_jitter=self.surface_jitter,
        )

        if self.debug_save_dir is not None and index < 16:
            np.save(self.debug_save_dir / f"{sample.asset_id}_coarse_{coarse_source}.npy", coarse.astype(bool))
            np.save(self.debug_save_dir / f"{sample.asset_id}_gt_voxel.npy", gt_voxel.astype(bool))
            np.save(self.debug_save_dir / f"{sample.asset_id}_queries.npy", queries)
            np.save(self.debug_save_dir / f"{sample.asset_id}_labels.npy", labels)

        return {
            "coarse": torch.from_numpy(coarse.astype(np.float32))[None],
            "gt_voxel": torch.from_numpy(gt_voxel.astype(np.float32))[None],
            "visual": visual.float(),
            "queries": torch.from_numpy(queries),
            "labels": torch.from_numpy(labels),
            "coarse_source": coarse_source,
            "asset_id": sample.asset_id,
            "voxel_path": str(sample.voxel_path),
            "mesh_path": str(sample.mesh_path),
            "qwen_voxel_path": str(sample.qwen_voxel_path) if sample.qwen_voxel_path else "",
        }


def write_split_manifest(path: Path, train: list[VoxelRefinerSample], val: list[VoxelRefinerSample]) -> None:
    payload = {
        "train_assets": sorted({s.asset_id for s in train}),
        "val_assets": sorted({s.asset_id for s in val}),
        "train_samples": len(train),
        "val_samples": len(val),
        "qwen_coverage_train": sum(1 for s in train if s.qwen_voxel_path is not None),
        "qwen_coverage_val": sum(1 for s in val if s.qwen_voxel_path is not None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
