"""Repository root resolution.

Absolute paths used to be hard-coded throughout this project. They are now resolved from
(in order):
  1. the ``FISH_ROOT`` environment variable, if set;
  2. the directory containing this file.

External inputs that do not live inside the repository are resolved the same way from their
own environment variables, with a documented default under ``FISH_ROOT``:

  ============== ======================================================================
  ISAACLAB_PATH  a source checkout of Isaac Lab (tested with 2.3.0 / Isaac Sim 5.1)
  ZEF_ROOT       the 3D-ZeF ZebraFish-05 dataset directory
  ============== ======================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("FISH_ROOT", Path(__file__).resolve().parent))


def repo_path(*parts: str) -> Path:
    """Path to something inside this repository."""
    return REPO_ROOT.joinpath(*parts)


ISAACLAB_PATH = Path(os.environ.get("ISAACLAB_PATH", REPO_ROOT.parent / "isaaclab"))
ZEF_ROOT = Path(os.environ.get("ZEF_ROOT", REPO_ROOT / "data" / "ZebraFish-05"))
