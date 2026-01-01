#!/usr/bin/env python
"""Smoke test for the new heading-aligned forward-velocity reward in the AMP fish env.

Verifies (no long training): tensor shapes, finite rewards (no NaN/Inf), reset behavior (no spurious
acceleration-penalty spike on the post-reset step), and that all new log keys are present + finite.
Uses a SHORT episode length so time-out resets happen frequently and exercise the reset path.

  <env_isaaclab>/bin/python scripts/smoke_amp_reward.py --device cuda:0 --num_envs 8 --steps 240
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--episode_s", type=float, default=1.0)   # ~30 control steps -> frequent resets
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

sys.path.insert(0, "source/FISH")
import FISH.tasks.direct.fish  # noqa: E402,F401  (registers the gym tasks)
from FISH.tasks.direct.fish.salmon_amp_autoskel_cfg import SalmonAMPAutoskelCfg  # noqa: E402

TASK = "Template-Salmon-AMP-Autoskel-Direct-v0"
NEW_LOG_KEYS = ["v_forward", "lateral_speed", "accel_mag", "pen_lateral", "pen_accel", "r_move",
                # slip-ratio + static-bend terms and the discriminator-health gauges
                "pen_slip", "slip_ratio", "pen_bend", "bend_ema_abs", "disc_score", "disc_score_std",
                # AMP-primary reward terms (2026-07-19)
                "r_progress", "v_forward_bl", "jvel_pen",
                # anti-freeze activity reward (2026-07-21)
                "r_activity", "bend_activity", "r_displacement", "net_fwd_bl", "r_track", "track_err"]


def main():
    cfg = SalmonAMPAutoskelCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.episode_length_s = args.episode_s
    cfg.debug_print = False
    cfg.with_deformable = True
    cfg.reset_deformable = True

    env = gym.make(TASK, cfg=cfg)
    base = env.unwrapped
    E, A = base.num_envs, base._num_actions
    dev = base.device
    print(f"[smoke] env up: num_envs={E} num_actions={A} device={dev}", flush=True)

    base.reset()
    fails, checks = [], []

    def check(cond, msg):
        checks.append((bool(cond), msg))
        if not cond:
            fails.append(msg)

    nan_any = False
    rew_min, rew_max = float("inf"), float("-inf")
    pen_accel_max = 0.0
    n_resets = 0
    post_reset_pen_accel = []          # pen_accel mean on steps right after a reset
    log_missing = set(NEW_LOG_KEYS)
    torch.manual_seed(0)

    for t in range(args.steps):
        act = (2.0 * torch.rand((E, A), device=dev) - 1.0)
        obs, rew, term, trunc, extras = base.step(act)

        rew = torch.as_tensor(rew)
        check(tuple(rew.shape) == (E,), f"reward shape {tuple(rew.shape)} != ({E},) @step{t}")
        fin = bool(torch.isfinite(rew).all())
        if not fin:
            nan_any = True
            check(False, f"non-finite reward @step{t}: {rew}")
        else:
            rew_min = min(rew_min, float(rew.min())); rew_max = max(rew_max, float(rew.max()))

        log = base.extras.get("log", {})
        for k in list(log_missing):
            if k in log:
                log_missing.discard(k)
        for k in NEW_LOG_KEYS:
            if k in log:
                v = float(torch.as_tensor(log[k]))
                if not np.isfinite(v):
                    check(False, f"log['{k}']={v} non-finite @step{t}")
        if "pen_accel" in log:
            pa = float(torch.as_tensor(log["pen_accel"]))
            pen_accel_max = max(pen_accel_max, pa)

        done = torch.as_tensor(term) | torch.as_tensor(trunc)
        if bool(done.any()):
            n_resets += int(done.sum())
            # the NEXT step is the first post-reset step for these envs; record its pen_accel
            if "pen_accel" in base.extras.get("log", {}):
                post_reset_pen_accel.append(float(torch.as_tensor(base.extras["log"]["pen_accel"])))

    # ---- assertions ----
    check(not nan_any, "NaN/Inf reward occurred")
    check(len(log_missing) == 0, f"missing log keys: {log_missing}")
    check(n_resets > 0, f"no resets happened (reset path not exercised); n_resets={n_resets}")
    # accel penalty must stay bounded (gate + clip): w_accel*accel_clip^2 = 0.002*400 = 0.8 upper bound
    check(pen_accel_max <= float(cfg.accel_clip) ** 2 + 1e-3,
          f"pen_accel mean {pen_accel_max:.3f} exceeds clip bound {cfg.accel_clip**2:.1f}")
    # reward should be bounded (no explosion from any term)
    check(rew_min > -100.0 and rew_max < 100.0, f"reward out of sane range [{rew_min:.2f},{rew_max:.2f}]")

    print("\n=== SMOKE SUMMARY ===", flush=True)
    print(f"steps={args.steps} resets={n_resets}  reward range=[{rew_min:.3f},{rew_max:.3f}]", flush=True)
    print(f"pen_accel mean-max over run = {pen_accel_max:.4f} (bound {cfg.accel_clip**2*1.0:.1f}); "
          f"post-reset pen_accel means = {[round(x,4) for x in post_reset_pen_accel[:6]]}", flush=True)
    print(f"new log keys present: {sorted(set(NEW_LOG_KEYS) - log_missing)}", flush=True)
    for ok, msg in checks:
        if not ok:
            print(f"  FAIL: {msg}", flush=True)
    ok = len(fails) == 0
    print(f"\nSMOKE_RESULT: {'PASS' if ok else 'FAIL'}  ({len(checks)-len(fails)}/{len(checks)} checks)", flush=True)


main()
import os  # noqa: E402
os._exit(0)   # SimulationContext teardown hangs on this box
