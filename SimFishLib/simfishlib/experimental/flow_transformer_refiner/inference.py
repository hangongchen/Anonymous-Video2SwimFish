"""Inference utilities: ODE sampling + marching cubes for the flow refiner."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from simfishlib.experimental.flow_transformer_refiner.flow import euler_sample
from simfishlib.experimental.flow_transformer_refiner.implicit_head import (
    FlowTransformerRefinerImplicit,
)
from simfishlib.experimental.flow_transformer_refiner.model import FlowTransformerRefiner


@torch.no_grad()
def sample_tsdf_volume(
    model: FlowTransformerRefiner,
    coarse: torch.Tensor,                # [B, 1, R, R, R] in [-1, 1] (signed)
    visual: torch.Tensor | None,
    *,
    num_steps: int = 50,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run the rectified-flow ODE from t=1 (noise) to t=0 (TSDF).

    Returns the sampled TSDF volume `[B, 1, R, R, R]`. Negative inside.
    """
    model_was_training = model.training
    model.eval()
    coarse = coarse.to(device=device, dtype=dtype)
    if visual is not None:
        visual = visual.to(device=device, dtype=dtype)

    R = model.resolution
    shape = (coarse.shape[0], 1, R, R, R)

    def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, coarse, t, visual)

    out = euler_sample(
        velocity_fn,
        shape,
        device=device,
        dtype=dtype,
        num_steps=num_steps,
        generator=generator,
    )
    if model_was_training:
        model.train()
    return out


@torch.no_grad()
def sample_implicit_volume(
    model: FlowTransformerRefinerImplicit,
    coarse: torch.Tensor,                # [B, 1, R_coarse, R_coarse, R_coarse]
    visual: torch.Tensor | None,
    *,
    query_resolution: int = 128,
    num_steps: int = 200,
    batch_query: int = 65536,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run rectified-flow Euler integration with the implicit decoder, sampling
    the SDF on a dense ``query_resolution^3`` grid.

    The transformer encoder is run once per Euler step (its conditioning depends
    on t through AdaLN). The decoder MLP is called over all query points in
    batches of ``batch_query`` to stay within GPU memory.

    Returns
    -------
    sdf_volume : torch.Tensor of shape ``[B, 1, R_q, R_q, R_q]`` with
                 R_q = query_resolution. Negative inside, positive outside.
    """
    model_was_training = model.training
    model.eval()
    coarse = coarse.to(device=device, dtype=dtype)
    if visual is not None:
        visual = visual.to(device=device, dtype=dtype)

    B = coarse.shape[0]
    R = int(query_resolution)
    K_total = R * R * R
    R_grid = int(coarse.shape[-1])  # encoder grid resolution

    coords = (torch.arange(R, device=device, dtype=dtype) + 0.5) / R
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1)
    query_xyz = grid.view(1, K_total, 3).expand(B, -1, -1).contiguous()

    # Initial state x_1 ~ N(0, I) of shape [B, K_total, 1]
    if generator is not None:
        x = torch.randn(B, K_total, 1, device=device, dtype=dtype, generator=generator)
    else:
        x = torch.randn(B, K_total, 1, device=device, dtype=dtype)

    # If the encoder consumes a grid-level x_t alongside coarse, we maintain
    # x_t_grid by adaptive-average-pooling the current high-res x onto the coarse
    # grid each step. The encoder grid is read off the coarse tensor directly so
    # this works even if it differs from 64.
    feed_grid_xt = getattr(model, "dit_xt_channels", 0) > 0

    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    for i in range(num_steps):
        t_cur = ts[i].expand(B)
        t_next = ts[i + 1]
        dt = t_next - ts[i]  # negative

        x_t_grid = None
        if feed_grid_xt:
            x_vol = x.view(B, R, R, R, 1).permute(0, 4, 1, 2, 3).contiguous()  # [B,1,R,R,R]
            if R == R_grid:
                x_t_grid = x_vol
            else:
                x_t_grid = torch.nn.functional.adaptive_avg_pool3d(
                    x_vol, output_size=(R_grid, R_grid, R_grid)
                )

        # One transformer pass per step.
        feature_volume = model.encode(coarse, t_cur, visual, x_t_grid=x_t_grid)
        # Batched decoder MLP over query points.
        v = torch.empty_like(x)
        for start in range(0, K_total, int(batch_query)):
            end = min(start + int(batch_query), K_total)
            v[:, start:end] = model.decoder(
                feature_volume,
                query_xyz[:, start:end],
                x[:, start:end],
                t_cur,
            )
        x = x + v * dt

    if model_was_training:
        model.train()

    sdf_volume = x.view(B, R, R, R, 1).permute(0, 4, 1, 2, 3).contiguous()
    return sdf_volume


def tsdf_to_mesh(tsdf: np.ndarray, iso: float = 0.0, pad: int = 1):
    """Marching cubes on a TSDF (negative inside).

    Returns (verts, faces) with verts in [0, 1]^3, or (empty, empty) if the
    surface isn't bracketed by the iso value.

    ``pad`` adds a border of "outside" (positive) cells around the volume before
    marching. Meshes normalized to span the full unit cube put head/tail tips
    right on the boundary, where marching cubes would otherwise leave the surface
    open and clip the tips. Padding closes the surface at its true extent. The
    padding offset is removed from the returned coordinates so the [0,1] mapping
    is unchanged.
    """
    try:
        from skimage.measure import marching_cubes
    except Exception as exc:
        raise RuntimeError("tsdf_to_mesh requires scikit-image") from exc
    if tsdf.ndim != 3:
        raise ValueError(f"expected 3D tsdf, got shape {tsdf.shape}")
    R = tsdf.shape[0]
    lo, hi = float(tsdf.min()), float(tsdf.max())
    if not (lo <= iso <= hi):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    if pad > 0:
        fill = max(hi, iso + 1.0)  # a clearly-outside value
        tsdf = np.pad(tsdf, pad, mode="constant", constant_values=fill)
    verts, faces, _, _ = marching_cubes(tsdf, level=iso)
    # marching_cubes returns verts in voxel-index coordinates. Undo the padding
    # offset, then map to [0,1]^3 (R-1 spans the unit cube, same as the dataset).
    verts = (verts - pad) / max(R - 1, 1)
    return verts.astype(np.float32), faces.astype(np.int64)
