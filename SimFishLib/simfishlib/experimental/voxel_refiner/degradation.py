from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationConfig:
    resolution: int = 64
    downsample_choices: tuple[int, ...] = (16, 32)
    dropout_prob_range: tuple[float, float] = (0.02, 0.12)
    false_positive_cuboids: tuple[int, int] = (0, 3)
    false_positive_size: tuple[int, int] = (2, 8)
    slab_noise_prob: float = 0.35
    morphology_prob: float = 0.7
    cleanup_components: bool = False


def degrade_voxel(voxel: np.ndarray, rng: np.random.Generator, config: DegradationConfig | None = None) -> np.ndarray:
    config = config or DegradationConfig(resolution=int(voxel.shape[0]))
    coarse_res = int(rng.choice(config.downsample_choices))
    degraded = block_down_up(voxel.astype(bool), coarse_res)

    if rng.random() < config.morphology_prob:
        degraded = _morphology(degraded, op=str(rng.choice(["dilate", "erode"])), iterations=int(rng.integers(1, 3)))

    occupied = degraded & (rng.random(degraded.shape) >= rng.uniform(*config.dropout_prob_range))
    degraded = occupied

    for _ in range(int(rng.integers(config.false_positive_cuboids[0], config.false_positive_cuboids[1] + 1))):
        degraded |= _random_cuboid(degraded.shape, rng, config.false_positive_size)

    if rng.random() < config.slab_noise_prob:
        degraded |= _random_slab(degraded.shape, rng)

    if config.cleanup_components:
        degraded = largest_component(degraded)
    return degraded.astype(bool)


def block_down_up(voxel: np.ndarray, coarse_resolution: int) -> np.ndarray:
    voxel = voxel.astype(bool)
    resolution = int(voxel.shape[0])
    if voxel.shape != (resolution, resolution, resolution):
        raise ValueError(f"Expected cubic voxel, got {voxel.shape}.")
    if resolution % coarse_resolution != 0:
        raise ValueError(f"{resolution=} must be divisible by {coarse_resolution=}.")
    factor = resolution // coarse_resolution
    coarse = voxel.reshape(coarse_resolution, factor, coarse_resolution, factor, coarse_resolution, factor).any(axis=(1, 3, 5))
    return np.repeat(np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1), factor, axis=2)


def largest_component(voxel: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return voxel.astype(bool)

    labels, count = ndimage.label(voxel.astype(bool))
    if count <= 1:
        return voxel.astype(bool)
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _morphology(voxel: np.ndarray, op: str, iterations: int) -> np.ndarray:
    try:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(3, 1)
        if op == "dilate":
            return ndimage.binary_dilation(voxel, structure=structure, iterations=iterations)
        if op == "erode":
            return ndimage.binary_erosion(voxel, structure=structure, iterations=iterations)
    except Exception:
        return voxel
    return voxel


def _random_cuboid(shape: tuple[int, int, int], rng: np.random.Generator, size_range: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    size = [int(rng.integers(size_range[0], size_range[1] + 1)) for _ in range(3)]
    start = [int(rng.integers(0, max(1, shape[i] - size[i]))) for i in range(3)]
    out[start[0] : start[0] + size[0], start[1] : start[1] + size[1], start[2] : start[2] + size[2]] = True
    return out


def _random_slab(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    axis = int(rng.integers(0, 3))
    width = int(rng.integers(1, 4))
    start = int(rng.integers(0, max(1, shape[axis] - width)))
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(start, start + width)
    out[tuple(slices)] = rng.random(out[tuple(slices)].shape) < 0.015
    return out
