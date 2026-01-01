# PCA Gait — swimming a FEM fish through a 2-D real-fish locomotion manifold

This branch contains the **PCA-latent locomotion pipeline**: PPO controls the articulated +
FEM Misty Minnow fish through only **two numbers per step** — the coefficients of the first
two PCA modes of a real zebrafish's body curvature — instead of its 21/27 joint DOFs.

```
PPO action [Δa1, Δa2]
  → bounded-rate coefficient state a(t)          (limits = p99 of the real-fish data)
  → curvature field κ(s,t) = mean(s) + a·V       (ZeF-05 PCA basis)
  → calibrated curvature → joint map q = q0+a·W  (in-sim calibrated Φ, ridge LS)
  → PD position targets on the 7 lateral D6 joints (±45°, k=120/c=6)
  → articulated skeleton + FEM soft body, analytic panel-strip water (zero-g)
```

Results (deterministic, from rest unless noted):

| policy | task | result |
|---|---|---|
| straight-swim (350 ep) | swim forward | **0.41 BL/s**, net yaw −1.4°/20 s — beats the hand-tuned scripted wave from rest (0.38, unstable) |
| reach v1 | random targets | 2.3 reaches/ep, but approaches **broadside** (73° heading error) |
| reach v2.1 "head-first" | random targets | 3.0+ reaches/ep at **36° heading error** (heading-gated progress + alignment-gated success bonus) |

## Layout

| Path | What |
|---|---|
| `source/FISH/FISH/tasks/direct/fish/salmon_swim_pca_{env,cfg}.py` | 2-PC action space + straight-swim task (`Template-Salmon-Swim-PCA-Misty-Direct-v0`) |
| `source/.../salmon_swim_pca_reach_{env,cfg}.py` | random-target multi-reach task + head-first variant (`…-PCA-Reach-…` / `…-PCA-Reach-Head-…`) |
| `source/.../agents/rl_games_ppo_pca*.yaml` | PPO configs (one per task, isolated from the AMP experiments) |
| `scripts/zef_manifold/` | PCA basis extraction from the ZebraFish-05 top-view video |
| `scripts/zef_playback/` | offline playback validation + the in-sim Φ calibration (see `demo_out/zef_playback/REPORT.md`) |
| `scripts/zef_pca_rl/` | train launchers + deterministic eval + video renderers |
| `demo_out/zef_manifold/pca_basis.npz` | **frozen PCA basis** (mean, components, coeffs) — runtime input |
| `demo_out/zef_playback/recording.npz` | **frozen calibration** (Φ, κ_rest, joint order) — runtime input |
| `demo_out/pca_rl_eval_final/` | straight-swim final eval: metrics + 4-condition videos |
| `demo_out/pca_reach_eval/` | reach evals old-vs-new: metrics + nose-arrow video |

## Reproduce

Environment: Isaac Sim 5.1 (pip) + source Isaac Lab, conda env `env_isaaclab_51`
(python 3.11). Always run with the env python directly:
`PY=/path/to/envs/env_isaaclab_51/bin/python`. Two artifacts are checked in
(`pca_basis.npz`, `recording.npz`) so steps 1–2 are optional.

```bash
# 0) smoke-test the env wiring (few minutes, 4 envs)
$PY -c "import sys; sys.path.insert(0,'source')"   # repo root on sys.path is done by the scripts

# 1) (optional) regenerate the PCA basis from the ZeF-05 dataset
$PY scripts/zef_manifold/extract_curvature.py && $PY scripts/zef_manifold/fit_pca.py

# 2) (optional) regenerate the curvature->joint calibration (runs Isaac, ~6 min)
$PY scripts/zef_playback/run_playback.py --device cuda:0     # writes demo_out/zef_playback/recording.npz

# 3) train straight swimming (256 envs, ~90 s/epoch, converges ~350 epochs)
DEV=cuda:0 bash scripts/zef_pca_rl/train_pca_swim.sh --num_envs 256 --horizon_length 128

# 4) train target reaching, head-first variant (recommended; the plain variant learns broadside gliding)
DEV=cuda:0 bash -c '$PY scripts/rl_games/train_ppo.py \
  --task Template-Salmon-Swim-PCA-Reach-Head-Misty-Direct-v0 \
  --device cuda:0 --num_envs 256 --headless --horizon_length 128'
# add --track for W&B (offline fallback + `wandb sync` if no API key)

# 5) deterministic evaluation + videos
$PY scripts/zef_pca_rl/eval_pca_swim.py  --checkpoint logs/rl_games/misty_pca_swim/<run>/nn/misty_pca_swim.pth --device cuda:0
$PY scripts/zef_pca_rl/eval_analyze.py   demo_out/pca_rl_eval/misty_pca_swim
$PY scripts/zef_pca_rl/eval_pca_reach.py --task Template-Salmon-Swim-PCA-Reach-Head-Misty-Direct-v0 \
    --checkpoint logs/rl_games/misty_pca_reach_head/<run>/nn/misty_pca_reach_head.pth --tag my_eval --device cuda:0
```

Training checkpoints/logs land in `logs/rl_games/<name>/<timestamp>/` (git-ignored).

## Key design facts (why it works)

- **Action bounds are data-driven**: |a| ≤ p99 of the real-fish coefficient distribution
  ([11.6, 6.7]); per-step rate ≤ p99 of the real Δa at 30 Hz ([8.9, 7.2]) — this allows a
  full-amplitude 3 Hz tail-beat (the PD corner is 3.2 Hz) but forbids frame-to-frame pose jumps.
- **Φ is calibrated in-sim** (bend each lateral joint alone, measure the midline curvature
  response with the same estimator used for scoring) — estimator bias cancels in the loop.
- **Reach reward, head-first variant**: positive distance-progress × clamp(heading_cos,0,1),
  negative progress ungated, + alignment × target-directed-speed, success bonus ×
  clamp(heading_cos,0,1) at the reach moment. Both the progress gate AND the success gate are
  required — with only the former, the flat bonus still finances broadside bump-reaches
  (measured: heading error stuck at ~105° while reaches climbed).
- FEM stability: dt=1/120, decimation 4 (30 Hz control), FEM nodal-velocity guard on; zero
  blow-ups across all training runs.
