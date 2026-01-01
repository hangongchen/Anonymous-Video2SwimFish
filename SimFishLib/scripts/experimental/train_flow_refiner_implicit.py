#!/usr/bin/env python3
"""Train the implicit-decoder variant of the flow-transformer refiner.

The encoder is a DiT over patched coarse voxels (same as the dense refiner).
The head is a tiny MLP that, given a continuous query point, trilinearly samples
the encoder's feature volume and predicts the rectified-flow velocity at that
point. Supervision is per-point (random pool of ~16 k mesh-SDF samples per
asset), so the supervision resolution decouples from any voxel grid.

Single GPU:
    python scripts/experimental/train_flow_refiner_implicit.py --output-dir <dir>

Two-GPU DDP:
    python -m torch.distributed.run --nproc_per_node=2 \
        scripts/experimental/train_flow_refiner_implicit.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from simfishlib.experimental.voxel_refiner.dataset import CoarseSourceMix
from simfishlib.experimental.flow_transformer_refiner import (
    ImplicitFlowRefinerConfig,
    ImplicitFlowRefinerDataset,
    build_implicit_model,
    discover_samples,
    save_checkpoint,
    split_by_asset,
)


# ----- distributed helpers (mirror the dense script) ------------------------


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_distributed() -> tuple[torch.device, int, int]:
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


# ----- training step --------------------------------------------------------


def flow_training_step_implicit(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    use_visual: bool,
    surface_loss_weight: float = 0.0,
    surface_loss_mode: str = "linear",
    surface_loss_tau: float = 0.03,
    eikonal_weight: float = 0.0,
    eikonal_band: float = 0.3,
    num_eikonal_points: int = 512,
    sdf_recon_weight: float = 0.0,
    sdf_recon_mode: str = "l1",
    tv_weight: float = 0.0,
    num_tv_points: int = 256,
    truncation: float = 4.0,
    coarse_resolution: int = 64,
    dit_xt_channels: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    coarse = batch["coarse"].to(device, non_blocking=True)
    target = batch["target_sdf"].to(device, non_blocking=True)        # [B, K, 1]
    query_xyz = batch["query_xyz"].to(device, non_blocking=True)      # [B, K, 3]
    visual = batch["visual"].to(device, non_blocking=True) if use_visual else None

    B = target.shape[0]
    t = torch.rand(B, device=device, dtype=target.dtype)
    noise = torch.randn_like(target)
    t_b = t.view(B, 1, 1)
    x_t_at_q = (1.0 - t_b) * target + t_b * noise
    v_target = noise - target

    # Grid-level x_t for the DiT input, if the encoder is configured to take it.
    # We sample the same t across the batch but draw fresh grid noise -- the per-
    # query (target,noise) and the per-grid (target,noise) live in independent
    # noise spaces, which is fine because the model only ever needs them to be
    # consistent in distribution, not in realization.
    x_t_grid = None
    if dit_xt_channels > 0:
        target_grid = batch["target_tsdf_grid"].to(device, non_blocking=True)  # [B,1,R,R,R]
        if target_grid.shape[1] != dit_xt_channels:
            raise ValueError(
                f"target_tsdf_grid has {target_grid.shape[1]} ch but dit_xt_channels={dit_xt_channels}"
            )
        noise_grid = torch.randn_like(target_grid)
        t_g = t.view(B, 1, 1, 1, 1)
        x_t_grid = (1.0 - t_g) * target_grid + t_g * noise_grid

    v_pred = model(
        x_t_at_q, coarse, t,
        query_xyz=query_xyz, visual=visual, x_t_grid=x_t_grid,
    )

    # Per-point surface weight `w`: emphasizes points near the surface (target~0).
    # Computed once and shared by BOTH the flow MSE and the SDF-recon loss, so the
    # model is pushed to be especially accurate on the surface in velocity space
    # *and* in recovered-clean-SDF space.
    if surface_loss_weight > 0.0:
        if surface_loss_mode == "exp":
            # multiplier `surface_loss_weight` at the surface, decaying to 1.0
            # with characteristic distance tau (in TSDF units).
            w = 1.0 + (surface_loss_weight - 1.0) * torch.exp(
                -target.abs() / max(float(surface_loss_tau), 1e-6)
            )
        else:
            w = 1.0 + surface_loss_weight * (1.0 - target.abs()).clamp(min=0.0)
    else:
        w = torch.ones_like(target)

    sq = (v_pred - v_target) ** 2  # [B, K, 1]
    flow_loss = (sq * w).mean()

    parts = {"loss_flow": float(flow_loss.detach()), "t_mean": float(t.mean())}
    loss = flow_loss

    # SDF reconstruction loss: x_hat_0 = x_t - t * v_pred, vs the GT clean target.
    # Free (reuses v_pred). Now surface-weighted by the SAME w as the flow loss, so
    # edge/thin-part reconstruction is supervised directly, not just on average.
    if sdf_recon_weight > 0.0:
        x_hat_0 = x_t_at_q - t_b * v_pred
        if sdf_recon_mode == "l2":
            recon = (w * (x_hat_0 - target).pow(2)).mean()
        else:
            recon = (w * (x_hat_0 - target).abs()).mean()
        loss = loss + float(sdf_recon_weight) * recon
        parts["loss_sdf_recon"] = float(recon.detach())

    if eikonal_weight > 0.0:
        # Finite-difference Eikonal on the implied clean prediction
        # x_hat_0 = x_t - t*v_pred. We can't autograd through F.grid_sample twice
        # (no double-backward), so we estimate the spatial gradient by querying
        # the decoder at p and at p+eps*e_d for d in {x,y,z}. x_t is held to a
        # single noise sample shared across the 4 offsets so that finite-diff
        # captures d/dp v_pred, matching d/dp x_hat_0 = -t * d/dp v_pred.
        # Target |grad x_hat_0| in [0,1]^3 coords is R/truncation (the TSDF is
        # sdf*R/truncation with |grad sdf|=1). Weighted by t so the regularizer
        # is meaningful (at t=0 d/dp x_hat_0 would vanish by construction here).
        K_eik = int(num_eikonal_points)
        eps = 1.0 / (2.0 * float(coarse_resolution))  # half a voxel edge in [0,1]^3 coords
        q_eik = torch.rand(B, K_eik, 3, device=device, dtype=target.dtype)
        offsets = torch.eye(3, device=device, dtype=target.dtype) * eps  # [3, 3]
        # Build the 4*K_eik query batch: [center, +x, +y, +z] stacked along the K axis.
        q_all = torch.cat([
            q_eik,
            (q_eik + offsets[0]).clamp_(0.0, 1.0),
            (q_eik + offsets[1]).clamp_(0.0, 1.0),
            (q_eik + offsets[2]).clamp_(0.0, 1.0),
        ], dim=1)  # [B, 4K, 3]
        x_eik = torch.randn(B, K_eik, 1, device=device, dtype=target.dtype)
        x_all = x_eik.repeat(1, 4, 1)  # x_t value held constant across the 4 offsets
        v_all = model(x_all, coarse, t, query_xyz=q_all, visual=visual, x_t_grid=x_t_grid)
        v_c, v_x, v_y, v_z = v_all.split(K_eik, dim=1)
        t_view = t.view(B, 1, 1)
        x_hat_0_c = x_eik - t_view * v_c
        gx = (x_eik - t_view * v_x - x_hat_0_c).squeeze(-1) / eps  # [B, K_eik]
        gy = (x_eik - t_view * v_y - x_hat_0_c).squeeze(-1) / eps
        gz = (x_eik - t_view * v_z - x_hat_0_c).squeeze(-1) / eps
        grad_norm = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
        target_norm = float(coarse_resolution) / max(float(truncation), 1e-6)
        # Soft near-surface weighting (was a hard |x_hat_0|<band cutoff): the
        # Eikonal constraint is enforced MUCH more strongly near the surface and
        # smoothly fades far away, where |grad|=1 matters less for edge quality.
        # `eikonal_band` is reused as the decay length (in TSDF units).
        band_w = torch.exp(-x_hat_0_c.abs().squeeze(-1) / max(float(eikonal_band), 1e-6))
        t_weight = t.view(B, 1).to(grad_norm.dtype)
        weight = band_w * t_weight
        denom = weight.sum().clamp_min(1.0)
        eik_loss = (((grad_norm - target_norm) ** 2) * weight).sum() / denom
        loss = loss + float(eikonal_weight) * eik_loss
        parts["loss_eikonal"] = float(eik_loss.detach())

    # TV / smoothness loss on x_hat_0. Same finite-difference trick as Eikonal:
    # query at p and p+eps along each axis (sharing one noise sample so the
    # finite difference captures dv/dp -> d(x_hat_0)/dp), then L1 of those diffs.
    if tv_weight > 0.0:
        K_tv = int(num_tv_points)
        eps = 1.0 / (2.0 * float(coarse_resolution))
        q_tv = torch.rand(B, K_tv, 3, device=device, dtype=target.dtype)
        off_tv = torch.eye(3, device=device, dtype=target.dtype) * eps
        q_tv_all = torch.cat([
            q_tv,
            (q_tv + off_tv[0]).clamp_(0.0, 1.0),
            (q_tv + off_tv[1]).clamp_(0.0, 1.0),
            (q_tv + off_tv[2]).clamp_(0.0, 1.0),
        ], dim=1)
        x_tv = torch.randn(B, K_tv, 1, device=device, dtype=target.dtype)
        x_tv_all = x_tv.repeat(1, 4, 1)
        v_tv_all = model(x_tv_all, coarse, t, query_xyz=q_tv_all, visual=visual, x_t_grid=x_t_grid)
        v_c, v_x, v_y, v_z = v_tv_all.split(K_tv, dim=1)
        t_view2 = t.view(B, 1, 1)
        x_hat_c = x_tv - t_view2 * v_c
        x_hat_x = x_tv - t_view2 * v_x
        x_hat_y = x_tv - t_view2 * v_y
        x_hat_z = x_tv - t_view2 * v_z
        tv_loss = (
            (x_hat_x - x_hat_c).abs().mean()
            + (x_hat_y - x_hat_c).abs().mean()
            + (x_hat_z - x_hat_c).abs().mean()
        )
        loss = loss + float(tv_weight) * tv_loss
        parts["loss_tv"] = float(tv_loss.detach())

    parts["loss"] = float(loss.detach())
    return loss, parts


# ----- args -----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Dataset/Fish2VoxelMergedV2Processed")
    p.add_argument("--qwen-coarse-root", default="Dataset/Fish2VoxelMergedV2QwenCoarse")
    p.add_argument("--canonical-root", default="Dataset/Fish2VoxelMergedV2CanonicalViews")
    p.add_argument("--angled-root", default="Dataset/Fish2VoxelMergedV2AngledViews")
    p.add_argument("--video-root", default="Dataset/Fish2VoxelMergedV2Video")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-root", default=None,
                   help="dir holding per-asset (points,sdf) pools; defaults to <output-dir>/cache_implicit")
    p.add_argument("--max-assets", type=int, default=None)
    p.add_argument("--coarse-resolution", type=int, default=64)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--truncation", type=float, default=4.0)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=5)
    p.add_argument("--decoder-num-freqs", type=int, default=10)
    p.add_argument("--visual-mode", choices=["video", "image", "both", "none"], default="video")
    p.add_argument("--no-visual", action="store_true")
    p.add_argument("--num-video-frames", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--coarse-source-mix", default=None)
    p.add_argument("--carve-augment", type=int, default=0,
                   help="1=train samples tilt-augmented carve variants (coarse64_ag*.npy). "
                        "NOTE: current variant sampling is frozen per-asset across epochs "
                        "(index-only RNG) and the tilts are misregistered vs the upright SDF "
                        "target -- keep 0 until per-epoch reseed + tilt registration are fixed.")
    p.add_argument("--num-query-points", type=int, default=4096)
    p.add_argument("--pool-size", type=int, default=16384)
    p.add_argument("--num-near-surface-pool", type=int, default=8192)
    p.add_argument("--num-uniform-pool", type=int, default=8192)
    p.add_argument("--near-surface-sigma", type=float, default=0.03)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="linear LR warmup from 0 -> base lr over this many epochs. 0 disables warmup.")
    p.add_argument("--cosine-schedule", action="store_true",
                   help="enable cosine LR decay from base lr -> min_lr over the remaining epochs after warmup")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="end-of-cosine LR floor")
    p.add_argument("--surface-loss-weight", type=float, default=0.0,
                   help="alpha in weight = 1 + alpha*(1 - |target|); upweights near-surface SDF samples")
    p.add_argument("--dit-xt-channels", type=int, default=0,
                   help="channels of x_t fed into the DiT alongside coarse (0=disabled, 1=TSDF). "
                        "When >0, the dataset must emit target_tsdf_grid (v2 cache).")
    p.add_argument("--thicken-fins", action="store_true",
                   help="solidify paper-thin mesh components before SDF sampling so fins survive supervision")
    p.add_argument("--fin-thickness", type=float, default=0.015,
                   help="thickness applied to thin components (unit-cube coords). ~1/64 = 0.0156 just barely survives 64^3")
    p.add_argument("--thin-threshold", type=float, default=0.01,
                   help="components whose smallest extent exceeds this are NOT thickened")
    p.add_argument("--eikonal-weight", type=float, default=0.0,
                   help="weight on Eikonal regularizer; 0 disables. ~0.01-0.05 is a reasonable starting point.")
    p.add_argument("--eikonal-band", type=float, default=0.3,
                   help="apply Eikonal only where |predicted clean SDF| < this (in TSDF units)")
    p.add_argument("--num-eikonal-points", type=int, default=512,
                   help="number of random query points per step for the Eikonal regularizer")
    p.add_argument("--surface-loss-mode", choices=["linear", "exp"], default="linear",
                   help="linear: w=1+alpha*(1-|target|); exp: w=1+(mult-1)*exp(-|target|/tau)")
    p.add_argument("--surface-loss-tau", type=float, default=0.03,
                   help="decay scale for exp surface weighting (TSDF units)")
    p.add_argument("--sdf-recon-weight", type=float, default=0.0,
                   help="weight on |x_hat_0 - target| (x_hat_0 = x_t - t*v_pred). ~0.1-0.25 recommended.")
    p.add_argument("--sdf-recon-mode", choices=["l1", "l2"], default="l1")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="weight on total-variation of x_hat_0; ~0.001-0.01 to damp high-freq noise")
    p.add_argument("--num-tv-points", type=int, default=256)
    return p.parse_args()


# ----- main loop ------------------------------------------------------------


def main() -> int:
    args = parse_args()
    device, rank, world_size = setup_distributed()

    output_dir = Path(args.output_dir)
    cache_root = Path(args.cache_root) if args.cache_root else output_dir / "cache_implicit"
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        with (output_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=2)

    samples = discover_samples(
        Path(args.data_root),
        resolution=64,
        max_assets=args.max_assets,
        qwen_coarse_root=Path(args.qwen_coarse_root) if args.qwen_coarse_root else None,
        canonical_root=Path(args.canonical_root) if args.canonical_root else None,
        angled_root=Path(args.angled_root) if args.angled_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
    )
    train_samples, val_samples = split_by_asset(samples, args.val_ratio, args.seed)
    if is_main():
        print(f"[flow-implicit] discovered {len(samples)} samples (train={len(train_samples)} val={len(val_samples)})")

    config = ImplicitFlowRefinerConfig(
        coarse_resolution=args.coarse_resolution,
        truncation=args.truncation,
        pool_size=args.pool_size,
        num_near_surface_pool=args.num_near_surface_pool,
        num_uniform_pool=args.num_uniform_pool,
        near_surface_sigma=args.near_surface_sigma,
        num_query_points=args.num_query_points,
        visual_mode=args.visual_mode,
        image_size=args.image_size,
        num_video_frames=args.num_video_frames,
        thicken_fins=bool(args.thicken_fins),
        fin_thickness=float(args.fin_thickness),
        thin_threshold=float(args.thin_threshold),
    )
    mix = CoarseSourceMix.parse(args.coarse_source_mix) if args.coarse_source_mix else CoarseSourceMix()

    train_ds = ImplicitFlowRefinerDataset(train_samples, config=config, coarse_mix=mix,
                                          cache_root=cache_root, seed=args.seed,
                                          carve_augment=bool(args.carve_augment))
    val_ds = (
        # val keeps the deterministic base carve (no tilt aug) for a clean metric
        ImplicitFlowRefinerDataset(val_samples, config=config, coarse_mix=mix,
                                   cache_root=cache_root, seed=args.seed + 1,
                                   carve_augment=False)
        if val_samples else None
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    use_visual = (not args.no_visual) and config.visual_mode != "none"
    model = build_implicit_model(
        coarse_resolution=args.coarse_resolution,
        patch=args.patch,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        feature_dim=args.feature_dim,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        decoder_num_freqs=args.decoder_num_freqs,
        use_visual=use_visual,
        dit_xt_channels=int(args.dit_xt_channels),
    ).to(device)
    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[flow-implicit] model params: {n_params/1e6:.2f}M (use_visual={use_visual})")

    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(args.epochs)
    warmup_steps = steps_per_epoch * int(args.warmup_epochs)
    use_schedule = (args.warmup_epochs > 0) or bool(args.cosine_schedule)
    if use_schedule and is_main():
        print(f"[flow-implicit] LR schedule: warmup_steps={warmup_steps} total_steps={total_steps} "
              f"base_lr={args.lr} min_lr={args.min_lr} cosine={bool(args.cosine_schedule)}")

    def _lr_at(global_step: int) -> float:
        if global_step < warmup_steps:
            return args.lr * (global_step + 1) / max(1, warmup_steps)
        if not args.cosine_schedule:
            return args.lr
        decay_steps = max(1, total_steps - warmup_steps)
        p = (global_step - warmup_steps) / decay_steps
        p = min(max(p, 0.0), 1.0)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * p))

    global_step = 0
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader):
            if use_schedule:
                lr_now = _lr_at(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            loss, parts = flow_training_step_implicit(
                model, batch, device, use_visual,
                surface_loss_weight=args.surface_loss_weight,
                surface_loss_mode=args.surface_loss_mode,
                surface_loss_tau=args.surface_loss_tau,
                eikonal_weight=args.eikonal_weight,
                eikonal_band=args.eikonal_band,
                num_eikonal_points=args.num_eikonal_points,
                sdf_recon_weight=args.sdf_recon_weight,
                sdf_recon_mode=args.sdf_recon_mode,
                tv_weight=args.tv_weight,
                num_tv_points=args.num_tv_points,
                truncation=args.truncation,
                coarse_resolution=args.coarse_resolution,
                dit_xt_channels=int(args.dit_xt_channels),
            )
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            epoch_losses.append(parts["loss"])
            if is_main() and step % args.log_every == 0:
                extra = ""
                if "loss_sdf_recon" in parts: extra += f" rec {parts['loss_sdf_recon']:.3f}"
                if "loss_eikonal" in parts:   extra += f" eik {parts['loss_eikonal']:.3f}"
                if "loss_tv" in parts:        extra += f" tv {parts['loss_tv']:.4f}"
                print(f"[flow-implicit] ep {epoch+1}/{args.epochs} step {step} loss {parts['loss']:.4f} flow {parts['loss_flow']:.4f}{extra} t̄ {parts['t_mean']:.2f}")

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_loss = float("nan")
        if val_ds is not None and is_main():
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    # val tracks the flow loss only (Eikonal needs grad) -- so it's
                    # always comparable across runs whether or not Eikonal is on.
                    loss, _ = flow_training_step_implicit(
                        unwrap(model), batch, device, use_visual,
                        surface_loss_weight=args.surface_loss_weight,
                        surface_loss_mode=args.surface_loss_mode,
                        surface_loss_tau=args.surface_loss_tau,
                        eikonal_weight=0.0,
                        sdf_recon_weight=0.0,
                        tv_weight=0.0,
                        dit_xt_channels=int(args.dit_xt_channels),
                    )
                    losses.append(float(loss))
            val_loss = float(np.mean(losses)) if losses else float("nan")

        if is_main():
            entry = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            history.append(entry)
            with (output_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)
            print(f"[flow-implicit] epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_epoch_{epoch+1:04d}.pt",
                    unwrap(model),
                    meta={"epoch": epoch + 1, "args": vars(args), "train_loss": train_loss, "val_loss": val_loss},
                )
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    unwrap(model),
                    meta={"epoch": epoch + 1, "args": vars(args), "train_loss": train_loss, "val_loss": val_loss},
                )

        if is_distributed():
            dist.barrier()

    if is_main():
        save_checkpoint(output_dir / "checkpoint_last.pt", unwrap(model), meta={"epochs": args.epochs, "args": vars(args)})

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
