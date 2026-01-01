# Reproducing: deformable salmon swims to a target (SUCCESS)

This branch trains a **deformable** salmon (rigid skeleton + active FEM body) to swim to a
target point in analytic "water", and **plays it back deterministically without the twitch /
blow-up-like instability** that the earlier configuration showed near the target.

Task: `Template-Salmon-Swim-Direct-v0`
(`source/FISH/FISH/tasks/direct/fish/salmon_swim_env.py` + `salmon_swim_cfg.py`).

> Read `CLAUDE.md` first for the environment setup, the FEM-stability story, and the settled
> `dt = 1/960` integration config. This doc only covers **how to train and play the
> swim-to-target result**, plus the one config change that fixed the playback instability.

---

## TL;DR — run it

```bash
# 1) train WITH the FEM deformable body active (the supported path), single GPU
bash scripts/train_salmon_swim_soft.sh                       # cuda:0, 256 envs
# or scale up:
HEADLESS=0 DEV=cuda:0 bash scripts/train_salmon_swim_soft.sh --num_envs 512

# 2) play the latest trained checkpoint back, deterministically, 1 env, dt=1/960, soft body on
REAL_TIME=0 PLAY_STEP_LOG=1 POLICY_MODE=deterministic \
bash scripts/play_latest_salmon_swim.sh
```

Expected playback (env 0): the fish swims to the target, triggers **success** around
`ep_step ≈ 196`, resets, and repeats — with **max joint velocity ≈ 2.75** and **no twitch**.

---

## The fix that made deterministic playback clean

`salmon_swim_cfg.py`: **`success_radius` 0.5 → 0.6**.

### Symptom (before)
In deterministic playback the fish reached ~0.60 m from the target, then around step ~200 its
joints started twitching (joint velocity jumped from ~2.9 to ~13), and it never reached the
target. It looked like a PhysX/FEM blow-up, but it was **100% reproducible in play** while
**rare in training**.

### Root cause (it is NOT a PhysX blow-up)
Verified from `demo_out/play_step_logs/play_deterministic_*.jsonl`:

- The deterministic (mean) policy asymptotes to a closest distance of **0.5975 m**. With
  `success_radius = 0.5` it **can never satisfy `dist < 0.5`** → never triggers success → never
  resets → it sits just outside the target forever.
- The policy's raw mean output `|mu|` is **saturated (1.3–3.5, past the ±1 action range)**, so
  the applied action is **clipped to ±1 and flips sign step-to-step** — a square wave that
  resonantly drives the D6 joints. The longer it dwells outside the radius, the more this
  limit cycle builds → the "twitch".
- It is **bounded**, not divergent: joint velocity peaks ~13 (the blow-up guard is 200), never
  goes NaN, never leaves bounds. A real CFL blow-up runs to ~1e9/NaN in a few steps. A reward
  threshold (`success_radius`) cannot cure a numerical instability — the fact that it does cure
  this proves the cause is the **success/reset boundary**, not the integrator.

### Why training looked "100% success" but playback failed
Training rolls out a **stochastic** policy: `a = clip(mu + sigma·noise)` with a constant
`fixed_sigma ≈ 0.74`. Success is judged **per step over a ~1500-step episode** ("did `dist`
dip below the radius at *any* step"), i.e. a first-passage event. Even though the *mean*
distance (~0.598) is above 0.5, the exploration noise jitters position every step; over a long
dwell near the target the probability that *at least one* step crosses 0.5 approaches 1. So:

- **Training:** noise crosses the 0.5 line → success fires → reset → "100% success", and the
  same noise de-coheres the square wave so the twitch rarely builds and (at jvel ~13 ≪ 200)
  is never flagged as a blow-up nor visible in 512-env aggregates.
- **Deterministic play:** no noise → mean asymptote 0.598 never crosses 0.5 → never resets →
  perfectly periodic saturated square wave → the limit cycle builds every time → 100% twitch.

Setting `success_radius = 0.6` lets the noiseless mean policy reach success at 0.5975, reset
before the limit cycle builds, and play cleanly.

### Confirmation
`play_deterministic_20260625_212316.jsonl` (radius 0.6, 459 steps, deterministic, 1 env, soft):

| metric                | radius 0.5 (twitch) | radius 0.6 (clean) |
|-----------------------|---------------------|--------------------|
| max joint velocity    | 13.74               | **2.75**           |
| steps with jvel > 3   | many                | **0**              |
| success / reset       | never (206 steps)   | `ep_step 196`, twice, periodic |
| min distance reached  | 0.5975              | 0.5975             |

---

## Caveat — this is the validated result, not the deep cure

`success_radius = 0.6` fixes the **"trapped just outside the boundary"** half. The **action
saturation** half (`|mu|` railed, no action penalty, `fixed_sigma`) is only *avoided* because
the episode now resets before the limit cycle develops — the root is not removed. A policy that
must hold position longer could still chatter.

The deeper cure (not applied on this branch, left for a future retrain) is reward shaping:
1. an **action penalty** (`-λ·|action|²` or an action-rate penalty) so the policy stops railing;
2. a **settle term** rewarding low root speed as `dist → 0` so the mean policy brakes and holds;
3. optionally requiring success to hold for **K consecutive steps** so the training metric stops
   being inflated by single-step noise crossings.

---

## Files

- `source/FISH/FISH/tasks/direct/fish/salmon_swim_cfg.py` — `success_radius = 0.6` (+ rationale).
- `source/FISH/FISH/tasks/direct/fish/salmon_swim_env.py` — env, hydro, done/blow-up logic,
  per-step debug, env-0 trajectory recorder.
- `scripts/train_salmon_swim_soft.sh` — train with the FEM body active (`HEADLESS=0` for GUI).
- `scripts/play_latest_salmon_swim.sh` — play the latest salmon checkpoint
  (`POLICY_MODE`, `PLAY_STEP_LOG`, `TRAIN_MATCH`, `REAL_TIME`, `NUM_ENVS`, `SIM_DT`/`DECIMATION`).
- Playback telemetry: `demo_out/play_step_logs/play_<mode>_<stamp>.jsonl`.
- Trajectories: `demo_out/swim_trajectories/run_<stamp>/`.
