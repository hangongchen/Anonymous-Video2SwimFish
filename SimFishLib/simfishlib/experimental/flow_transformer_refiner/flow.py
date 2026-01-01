"""Rectified-flow / flow-matching utilities.

We adopt the simplest "rectified flow" convention:

    x_t = (1 - t) * x0 + t * x1            with x1 ~ N(0, I)
    v*  = x1 - x0                           (constant in t, "rectified")

The network predicts v_theta(x_t, t, cond). Training loss is MSE(v_theta, v*).
At inference, integrate the ODE  dx/dt = v_theta(x_t, t, cond)  from t=1
(noise) back to t=0 (data) with Euler steps (default 50).

This avoids the noise-schedule machinery of full diffusion while keeping the
PhysX-Anything-style "transformer predicts a velocity" setup.
"""

from __future__ import annotations

from typing import Callable

import torch


def sample_x_t(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * x0 + t * noise.

    t has shape [B] (one timestep per sample). x0, noise have shape
    [B, ...spatial]. t is broadcast across all spatial dims.
    """
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """v* = noise - x0 (constant in t under rectified-flow parameterization)."""
    return noise - x0


def flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE between predicted and target velocities, optional reduction."""
    diff = (pred_velocity - target_velocity) ** 2
    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    raise ValueError(f"unknown reduction {reduction!r}")


def euler_sample(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    num_steps: int = 50,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    initial: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Integrate dx/dt = velocity_fn(x, t) from t=1 down to t=0 with Euler.

    velocity_fn must accept (x_t [B,...], t [B]) and return [B,...].
    Returns the final x_0 estimate of shape `shape`.

    If `initial` is given it overrides the random Gaussian start (useful for
    deterministic tests / repeatable inference).
    """
    if initial is None:
        if generator is not None:
            x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        else:
            x = torch.randn(shape, device=device, dtype=dtype)
    else:
        x = initial.to(device=device, dtype=dtype).clone()

    B = shape[0]
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    traj = [x.detach().cpu()] if return_trajectory else None
    for i in range(num_steps):
        t_cur = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_cur  # negative
        t_batched = t_cur.expand(B)
        v = velocity_fn(x, t_batched)
        x = x + v * dt
        if traj is not None:
            traj.append(x.detach().cpu())
    if return_trajectory:
        return x, traj
    return x
