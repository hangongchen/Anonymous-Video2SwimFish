"""Small CNN visual encoder for the flow-transformer refiner.

Copied (rather than imported) from the V2 refiner so this module is fully
isolated and can be deleted without touching the V2 codebase. Behavior matches
`simfishlib/experimental/voxel_refiner/visual_encoder.py`: accepts a single
image `[B, 3, H, W]` or a stack of video frames `[B, K, 3, H, W]`, returns a
flat `[B, F]` feature.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int, stride: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(inplace=True),
    )


class SmallImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.stem = _block(in_channels, 32, stride=2)
        self.stage2 = _block(32, 64)
        self.stage3 = _block(64, 128)
        self.stage4 = _block(128, feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.pool(x).flatten(1)


class VisualEncoder(nn.Module):
    """[B,3,H,W] or [B,K,3,H,W] -> [B, feature_dim]."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.image = SmallImageEncoder(feature_dim=feature_dim)
        self.feature_dim = feature_dim

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        if visual.ndim == 5:
            b, k, c, h, w = visual.shape
            flat = visual.reshape(b * k, c, h, w)
            feats = self.image(flat).reshape(b, k, self.feature_dim)
            return feats.mean(dim=1)
        if visual.ndim == 4:
            return self.image(visual)
        raise ValueError(
            f"visual must be [B,3,H,W] or [B,K,3,H,W], got shape {tuple(visual.shape)}"
        )
