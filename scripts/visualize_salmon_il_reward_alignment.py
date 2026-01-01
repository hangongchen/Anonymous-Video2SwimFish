# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_repo_root = Path(__file__).resolve().parents[1]
_fish_source = _repo_root / "source"
if str(_fish_source) not in sys.path:
    sys.path.insert(0, str(_fish_source))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize Salmon IL reward-frame point cloud overlap.")
parser.add_argument("--task", type=str, default="Template-Salmon-IL-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--frames", type=int, default=30)
parser.add_argument("--output-dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import FISH.tasks  # noqa: F401
from FISH.tasks.direct.fish.salmon_IL_cfg import SalmonILEnvCfg


def _build_cfg() -> SalmonILEnvCfg:
    cfg = SalmonILEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    cfg.pointcloud_capture_enabled = False
    cfg.real_pointcloud_random_start = False
    cfg.real_pointcloud_start_frame = 0
    return cfg


def _frame_used_for_last_reward(env) -> int:
    unwrapped = env.unwrapped
    used = unwrapped._il_episode_frame[0] - 1
    used = torch.clamp(used, min=0)
    frame_id = unwrapped._il_real_frame_start[0] + used
    if unwrapped.cfg.real_pointcloud_loop:
        frame_id = torch.remainder(frame_id, unwrapped._real_pointcloud_frame_count)
    else:
        frame_id = torch.clamp(frame_id, max=unwrapped._real_pointcloud_frame_count - 1)
    return int(frame_id.item())


def _collect_reward_frame(env, frame_idx: int) -> dict:
    unwrapped = env.unwrapped
    sim_points = unwrapped._get_live_sim_pointclouds_for_reward()[0]
    real_frame_id = _frame_used_for_last_reward(env)
    real_points = unwrapped._real_pointcloud_sequence[real_frame_id]
    real_aligned, sim_used = unwrapped._get_reward_aligned_pointclouds(0, real_points, sim_points)
    chamfer = float(unwrapped._symmetric_chamfer_distance(real_aligned, sim_used).item())
    flip_major = bool(unwrapped._sim_alignment_flip_major[0].item())

    return {
        "frame": frame_idx,
        "real_frame": real_frame_id,
        "flip_major": flip_major,
        "chamfer": chamfer,
        "real_aligned": real_aligned.detach().cpu().numpy(),
        "sim_aligned": sim_used.detach().cpu().numpy(),
    }


def _projection_limits(records: list[dict], axes: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    all_real = np.concatenate([record["real_aligned"][:, axes] for record in records], axis=0)
    all_sim = np.concatenate([record["sim_aligned"][:, axes] for record in records], axis=0)
    all_points = np.concatenate((all_real, all_sim), axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.05, 1.0e-3)
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def _plot_projection_grid(
    records: list[dict],
    axes: tuple[int, int],
    axis_labels: tuple[str, str],
    output_path: Path,
) -> None:
    xlim, ylim = _projection_limits(records, axes)
    fig, grid = plt.subplots(5, 6, figsize=(18, 14), constrained_layout=True)
    for ax, record in zip(grid.flat, records, strict=False):
        real_pts = record["real_aligned"][:, axes]
        sim_pts = record["sim_aligned"][:, axes]
        ax.scatter(real_pts[:, 0], real_pts[:, 1], s=1.0, alpha=0.35, c="#1f77b4", label="real")
        ax.scatter(sim_pts[:, 0], sim_pts[:, 1], s=1.0, alpha=0.35, c="#ff7f0e", label="sim")
        ax.set_title(
            f"frame {record['frame']:02d} | real {record['real_frame']:03d}\n"
            f"Chamfer {record['chamfer']:.4f} | flip={record['flip_major']}",
            fontsize=8,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in grid.flat[len(records) :]:
        ax.axis("off")

    handles, labels = grid.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"Salmon IL reward-frame overlap ({axis_labels[0]}-{axis_labels[1]} projection)\n"
        "Real and simulated point clouds are shown in the fixed frame-0 reward coordinate used by the policy.",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_metrics(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "real_frame", "chamfer", "flip_major"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame": record["frame"],
                    "real_frame": record["real_frame"],
                    "chamfer": record["chamfer"],
                    "flip_major": record["flip_major"],
                }
            )


def main():
    output_dir = (
        Path(args_cli.output_dir)
        if args_cli.output_dir is not None
        else _repo_root / "run_pointclouds" / f"salmon_il_reward_alignment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=_build_cfg(), render_mode=None)
    env.reset()
    unwrapped = env.unwrapped
    actions = torch.zeros((unwrapped.num_envs, unwrapped._num_actions), device=unwrapped.device)

    records: list[dict] = []
    for frame_idx in range(args_cli.frames):
        _, _, terminated, truncated, _ = env.step(actions)
        records.append(_collect_reward_frame(env, frame_idx))
        if bool(terminated[0]) or bool(truncated[0]):
            env.reset()

    _plot_projection_grid(records, (0, 1), ("x", "y"), output_dir / "overlay_grid_xy_first30.png")
    _plot_projection_grid(records, (0, 2), ("x", "z"), output_dir / "overlay_grid_xz_first30.png")
    _plot_projection_grid(records, (1, 2), ("y", "z"), output_dir / "overlay_grid_yz_first30.png")
    _write_metrics(records, output_dir / "alignment_metrics.csv")

    print(f"Saved overlap plots to {output_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
