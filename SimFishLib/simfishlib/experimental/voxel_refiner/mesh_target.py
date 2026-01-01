"""GT-mesh helpers for the V2 implicit-occupancy refiner.

The asset's `asset.obj` is loaded, normalized to the unit cube [0,1]^3 (same
convention as the existing voxel.npy grid: index 0 along each axis is the
minimum corner), and then used to:

  - sample query points + inside/outside labels for training (BCE);
  - sample surface points for Chamfer evaluation;
  - rasterize to an occupancy grid for voxel-IoU evaluation (cross-comparable
    against the existing Qwen voxel64 metric).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_normalized_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}.")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError(f"Invalid or empty mesh: {path}")
    return _normalize_mesh(mesh)


def sample_mesh_occupancy_points(
    path: Path,
    count: int,
    rng: np.random.Generator,
    surface_fraction: float = 0.5,
    surface_jitter: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample query points around the normalized mesh + inside/outside labels.

    Returns:
        queries: float32 [count, 3] in [0, 1]^3
        labels:  float32 [count]    in {0, 1}
    """
    mesh = load_normalized_mesh(path)
    return _sample_from_loaded(mesh, count, rng, surface_fraction, surface_jitter)


def _sample_from_loaded(
    mesh,
    count: int,
    rng: np.random.Generator,
    surface_fraction: float,
    surface_jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    surface_count = int(count * surface_fraction)
    uniform_count = count - surface_count
    points: list[np.ndarray] = []
    if surface_count > 0 and len(mesh.faces) > 0:
        try:
            surface, _ = mesh.sample(surface_count, return_index=True)
        except Exception:
            import trimesh

            surface, _ = trimesh.sample.sample_surface(mesh, surface_count)
        surface = surface + rng.normal(0.0, surface_jitter, size=surface.shape)
        points.append(surface)
    if uniform_count > 0:
        points.append(rng.random((uniform_count, 3)))
    query = np.clip(np.concatenate(points, axis=0).astype(np.float32), 0.0, 1.0)
    labels = _contains_points(mesh, query).astype(np.float32)
    return query, labels


def sample_surface_points(path: Path, count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform surface samples in [0, 1]^3 for Chamfer evaluation."""
    mesh = load_normalized_mesh(path)
    if len(mesh.faces) == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        samples, _ = mesh.sample(count, return_index=True)
    except Exception:
        import trimesh

        samples, _ = trimesh.sample.sample_surface(mesh, count)
    return np.asarray(samples, dtype=np.float32)


def mesh_to_occupancy(path: Path, resolution: int = 64, samples: int = 200_000) -> np.ndarray:
    """Rasterize the normalized mesh into a [R,R,R] boolean grid.

    Grid index 0 along each axis = mesh coordinate 0 along that axis; index R-1
    = mesh coordinate 1. Used for voxel-IoU comparisons.
    """
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("mesh GT target generation requires trimesh.") from exc

    normalized = load_normalized_mesh(path)
    try:
        voxels = normalized.voxelized(pitch=1.0 / resolution).fill()
        points = np.asarray(voxels.points, dtype=np.float32)
    except Exception:
        if len(normalized.faces) == 0 or samples <= 0:
            points = np.asarray(normalized.vertices, dtype=np.float32)
        else:
            sampled, _ = trimesh.sample.sample_surface(normalized, samples)
            points = np.concatenate(
                [
                    np.asarray(normalized.vertices, dtype=np.float32),
                    sampled.astype(np.float32),
                ],
                axis=0,
            )

    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if points.size == 0:
        return grid
    coords = np.floor(points * (resolution - 1)).astype(np.int64)
    coords = np.clip(coords, 0, resolution - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid


# ---------------------------------------------------------------------------
# point-in-mesh + normalization (private helpers)
# ---------------------------------------------------------------------------


def _contains_points(mesh, points: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(points)
    except Exception:
        return _ray_contains_points(mesh, points)


def _ray_contains_points(
    mesh, points: np.ndarray, point_chunk: int = 256, triangle_chunk: int = 2048
) -> np.ndarray:
    """Dependency-free odd/even ray test fallback for mesh occupancy labels."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0 or points.size == 0:
        return np.zeros((len(points),), dtype=bool)

    triangles = vertices[faces]
    direction = np.asarray([1.0, 0.371390676, 0.17320508], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eps = 1e-9
    out = np.zeros((len(points),), dtype=bool)

    for p_start in range(0, len(points), point_chunk):
        query = points[p_start : p_start + point_chunk].astype(np.float64, copy=False)
        counts = np.zeros((len(query),), dtype=np.int64)
        for t_start in range(0, len(triangles), triangle_chunk):
            tri = triangles[t_start : t_start + triangle_chunk]
            edge1 = tri[:, 1] - tri[:, 0]
            edge2 = tri[:, 2] - tri[:, 0]
            h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
            a = np.einsum("tc,tc->t", edge1, h)
            valid = np.abs(a) > eps
            if not np.any(valid):
                continue
            inv_a = np.zeros_like(a)
            inv_a[valid] = 1.0 / a[valid]

            s = query[:, None, :] - tri[None, :, 0, :]
            u = np.einsum("ptc,tc->pt", s, h) * inv_a[None, :]
            q = np.cross(s, edge1[None, :, :])
            v = np.einsum("ptc,c->pt", q, direction) * inv_a[None, :]
            ray_t = np.einsum("ptc,tc->pt", q, edge2) * inv_a[None, :]
            hits = (
                valid[None, :]
                & (u >= -eps)
                & (v >= -eps)
                & ((u + v) <= 1.0 + eps)
                & (ray_t > eps)
            )
            counts += hits.sum(axis=1)
        out[p_start : p_start + len(query)] = (counts % 2) == 1
    return out


def _normalize_mesh(mesh):
    """Uniform-scale + center the mesh in the unit cube [0,1]^3.

    This matches the dataset's `voxel.npy` convention: the longest axis spans
    [0,1] and the two shorter axes are centered (so the occupied bbox of the
    rasterized mesh agrees with the occupied bbox of `voxel.npy`). Pinning to
    the [0,0,0] corner instead would put the supervision in the wrong corner
    of the cube relative to the input voxel.
    """
    normalized = mesh.copy()
    mins = normalized.vertices.min(axis=0)
    maxs = normalized.vertices.max(axis=0)
    scale = float(np.maximum((maxs - mins).max(), 1e-6))
    centered = (normalized.vertices - 0.5 * (mins + maxs)) / scale + 0.5
    normalized.vertices = centered
    return normalized


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean Chamfer distance (L2) between two point sets in [0,1]^3.

    Uses scipy.spatial.cKDTree for efficiency. Returns the mean of (mean a->b
    + mean b->a). Returns NaN if either set is empty.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("chamfer_distance requires scipy") from exc
    ta = cKDTree(a)
    tb = cKDTree(b)
    d_ab, _ = tb.query(a)
    d_ba, _ = ta.query(b)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))
