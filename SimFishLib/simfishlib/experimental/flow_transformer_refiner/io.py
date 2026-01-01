"""Checkpoint + minimal OBJ writer for the flow-transformer refiner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path: Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a minimal OBJ file. Faces are 0-indexed numpy ints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for v in verts:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
    for f in faces:
        lines.append(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
