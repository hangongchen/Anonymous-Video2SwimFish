from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def empty_voxel_grid(resolution: int) -> np.ndarray:
    return np.zeros((resolution, resolution, resolution), dtype=bool)


def parse_voxel_response(text: str, resolution: int) -> np.ndarray:
    """Parse a compact JSON voxel response from the VLM.

    Accepted schema:
    {
      "resolution": 64,
      "occupied_indices": [[x, y, z], ...]
    }

    A flat occupied index list is also accepted through "occupied_flat".
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")

    payload = json.loads(text[start : end + 1])
    response_resolution = int(payload.get("resolution", resolution))
    if response_resolution != resolution:
        raise ValueError(
            f"Voxel response resolution {response_resolution} does not match requested {resolution}."
        )

    grid = empty_voxel_grid(resolution)
    for item in payload.get("occupied_indices", []):
        if len(item) != 3:
            continue
        x, y, z = (int(v) for v in item)
        if 0 <= x < resolution and 0 <= y < resolution and 0 <= z < resolution:
            grid[x, y, z] = True

    for flat in payload.get("occupied_flat", []):
        idx = int(flat)
        if 0 <= idx < resolution**3:
            x = idx // (resolution * resolution)
            y = (idx // resolution) % resolution
            z = idx % resolution
            grid[x, y, z] = True

    return grid


def save_voxel(path: str | Path, voxel: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, voxel.astype(bool))


def load_voxel(path: str | Path) -> np.ndarray:
    return np.load(path).astype(bool)


def voxel_to_obj(voxel: np.ndarray, output_path: str | Path, scale: float = 1.0) -> None:
    """Write an OBJ mesh made from exposed voxel faces.

    This is intentionally simple but valid: adjacent occupied voxels share no
    internal faces, so the output is usable for inspection and downstream tests.
    """
    voxel = voxel.astype(bool)
    resolution = int(voxel.shape[0])
    if voxel.shape != (resolution, resolution, resolution):
        raise ValueError(f"Expected cubic voxel grid, got {voxel.shape}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    step = scale / resolution
    origin = -scale / 2.0

    directions = [
        ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
        ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
    ]

    def add_vertex(coord: tuple[float, float, float]) -> int:
        if coord not in vertex_index:
            vertex_index[coord] = len(vertices) + 1
            vertices.append(coord)
        return vertex_index[coord]

    occupied = np.argwhere(voxel)
    for x, y, z in occupied:
        for (dx, dy, dz), corners in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < resolution and 0 <= ny < resolution and 0 <= nz < resolution and voxel[nx, ny, nz]:
                continue
            face = []
            for cx, cy, cz in corners:
                coord = (
                    origin + (x + cx) * step,
                    origin + (y + cy) * step,
                    origin + (z + cz) * step,
                )
                face.append(add_vertex(coord))
            faces.append(tuple(face))

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# SimFishLib voxel mesh\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.8f} {vy:.8f} {vz:.8f}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")


def voxelize_vertices(vertices: Iterable[Iterable[float]], resolution: int) -> np.ndarray:
    arr = np.asarray(list(vertices), dtype=np.float32)
    grid = empty_voxel_grid(resolution)
    if arr.size == 0:
        return grid
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    coords = ((arr - mins) / span * (resolution - 1)).round().astype(int)
    coords = np.clip(coords, 0, resolution - 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return grid
