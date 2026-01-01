# FISH — Salmon swim-to-target RL (analytic water)

Branch `fish_analytic_water`: a registered Isaac Lab RL task that trains a salmon (rigid
skeleton + FEM deformable body) to **swim toward a target point in water**, using an
analytic MuJoCo-style hydrodynamic force model. This branch is trimmed to the modules
needed to run that training.

## Environment

- Isaac Sim 5.1 (pip) + source Isaac Lab at `${ISAACLAB_PATH}`.
- Conda env `env_isaaclab_51` (Python 3.11). **Run with the env python directly**
  (`${FISH_PYTHON} …`) — the repo's `isaaclab.sh`
  is bound to base python here, so `isaaclab.sh -p` gives `ModuleNotFoundError: isaaclab`.
- 2× RTX PRO 6000 Blackwell (96 GB each), display `:1`.
- Salmon asset: `source/FISH/FISH/tasks/direct/fish/agents/salmon/uniformed_scale_small_salmon.usd`
  — 10 rigid `Cylinder` bones + 9 `D6` joints (3 rotational DOF each, ±15°, 27 DOF total),
  ≈ 0.48 kg / 0.26 m, plus a 175-node FEM deformable mesh attached to the bones via
  `PhysxPhysicsAttachment`.

## RL task: `Template-Salmon-Swim-Direct-v0`

`source/FISH/FISH/tasks/direct/fish/salmon_swim_env.py` + `salmon_swim_cfg.py`, registered
in `__init__.py`. Mirrors `Template-Fish-Direct-v0` (red cuboid target marker;
distance + heading + success reward; obs = `13 + 2·njoints`) but:

- **zero-gravity "water"** — neutrally buoyant; an analytic **MuJoCo ellipsoid fluid
  wrench is applied to every bone each physics step** (`mujoco_fluid_wrench`,
  vectorized over `num_envs × num_bodies`; blunt+slender quadratic drag, angular drag,
  Stokes viscous, Magnus + Kutta lift). The model is passive (drag dissipates, lift does
  no net work). All hydro math is self-contained in `salmon_swim_env.py`.
- **PD position-target control** on the D6 joints. The USD ships the joints with only
  `PhysicsLimitAPI` (no drive) → `set_joint_position_target` is a no-op. `_prepare_env_assets()`
  authors a force-mode angular `DriveAPI` on rotX/Y/Z of each joint on env_0 **before
  `clone_environments`** so position control actuates (verified: target 0.2 → max|q|≈0.19
  at stiffness 60). `cfg.control_mode` can switch to `"effort"`.
- **PER-SLICE hydro** — the pencil-thin skeleton bones (~2–6 mm) give negligible drag, so
  `scripts/precompute_per_bone_hydro.py` fits one ellipsoid per bone to the FEM **body
  slice** around it (body girth ≈ 0.1 m, ~1000× the bone frontal area) →
  `agents/salmon/per_bone_hydro.npz`, which the env loads so the undulation makes real
  thrust. Falls back to thin inertia ellipsoids if the npz is missing.
- **`success_rate` is EPISODIC** (logged in `_get_dones`: fraction of *ended* episodes that
  reached the target, rolling window). The instantaneous `success.mean()` is ~0 because
  reaching the target terminates the episode — that is kept only as `in_radius_frac`.
- **`with_deformable`** (`cfg` / hydra `env.with_deformable=true`) keeps the FEM body
  **active in training**. This is required for a transferable policy: the soft body adds
  attachment / inertia / viscoelastic-damping dynamics the policy is tuned to, so a
  skeleton-only policy does NOT transfer (per-slice hydro fixes the drag *force*, not the
  inertial/structural dynamics). The cfg `PhysxCfg` sizes the GPU soft-body buffers.
- optional **top-down rolling video of env 0** (`cfg.record_video`, needs `--enable_cameras`):
  one mp4 per epoch (`video_clip_steps=256`), keeps the last `video_keep=5` on disk
  (`_record_step`/`_save_clip`). NOTE: recording + `with_deformable` STALLS the Replicator
  shader compile — record on a skeleton-only run.

## FEM deformable body — the blow-up, the cure, and the SETTLED config (read before touching `with_deformable`)

Hard-won facts from the FEM-stability work (the soft body attached to the articulation is fragile in PhysX 5.1):

- **The blow-up is a CFL numerical instability, intrinsic to PhysX — NOT caused by the analytic hydro.**
  A stiff FEM mode goes unstable when `dt > ~2/ω`; a COHERENT swim gait pumps it resonantly → joint vel
  → ~1e9 → NaN (so early/random gaits look fine; it blows up only AFTER the gait coheres, ~epoch 4–6).
  Proof it's PhysX-intrinsic, not the water: the CFL ceiling is predicted from FEM material params alone
  (no hydro term), every fix lever is FEM-internal, and the hydro is passive (net energy SINK). Isaac Lab
  maintainer (2026-02): "deformable-as-articulation-link is UNSUPPORTED in 5.1.0 / Lab 2.3.0." MuJoCo
  doesn't rescue it on GPU either — MJX has NO flex/deformables; mujoco_warp flex is experimental (only
  MuJoCo's CPU C engine does flex+articulation+pin cleanly).

- **The asset ships NO deformable-body material → youngsModulus/damping overrides were SILENT NO-OPS.**
  The deformable mesh has `PhysxDeformableBodyAPI` but no `PhysxDeformableBodyMaterialAPI` prim, so PhysX
  used its default material (`youngsModulus 5e7`, stiff) in EVERY prior run — the "softening to 1e5" never
  happened. `_prepare_env_assets` now CREATEs + BINDs a material (`deformableUtils.add_deformable_body_material`
  + `physicsUtils.add_physics_material_to_prim`, "physics" purpose) under `env_0/skeleton/.../deformableBodyMaterial`
  before clone. **ALWAYS verify it applied via the startup print** `FEM material CREATED+BOUND … (add_material
  ok=True)` — material attrs written to the wrong prim fail silently with no error.

- **The cure = stiffness-proportional material damping + small dt (complementary, not substitutes).**
  `elasticityDamping=0.05` (Rayleigh-β; modal damping grows with ω → bites the unstable high-ω mode → flips
  its growth-rate SIGN) + `dampingScale=1.0`, `youngsModulus=1e5`, `poissonsRatio=0.45` (material prim);
  `vertexVelocityDamping=4.0`, `sleepDamping=10`, `settlingThreshold=0.1`, `solverPositionIterationCount=96`
  (body prim). Damping flips the mode's sign; dt is a HARD CFL ceiling damping cannot substitute for.

- **SETTLED dt = 1/960 (decimation 32). Do NOT re-search.** Results (each blows up only after the gait
  coheres, so "survives 2 epochs" proves nothing): **1/240, 1/480, 1/720 all BLEW UP; 1/960 (1.04 ms)
  STABLE** (64/64 envs ran full 1500-step episodes, 0 blow-ups). Control stays 30 Hz via `decimation =
  (1/dt)/30`, so dt is decoupled from the policy interface. `SIM_DT` (cfg, defined ONCE) feeds BOTH
  `SimulationCfg.dt` AND `episode_length_s = 1500 * decimation * SIM_DT`, which keeps max_episode_length =
  **1500 control steps** for any dt — never revert to a hard-coded `/480`.

## Running

```bash
# train WITH the FEM body active (the supported path), single GPU, 256 envs ~ 8 GB
bash scripts/train_salmon_swim_soft.sh
# skeleton-only (faster); add a top-down rolling video of env 0
bash scripts/train_salmon_swim.sh --enable_cameras env.record_video=true
# regenerate the per-slice ellipsoids
python scripts/precompute_per_bone_hydro.py
```

`scripts/train_salmon_swim*.sh` clean up leftover runs, then launch
`scripts/rl_games/train_ppo.py` (env python). `--num_envs N`, `DEV=cuda:1`, and extra args
pass through. See `how_to_run.md`.

The current cured-soft run (GUI, 512 envs, per-step blow-up logging):
```bash
HEADLESS=0 DEV=cuda:0 bash scripts/train_salmon_swim_soft.sh --num_envs 512 env.debug_print=true
```
- **`HEADLESS=0`** on `train_salmon_swim_soft.sh` opens the Isaac Sim GUI viewport (default `HEADLESS=1`
  = no GUI, faster). GUI is heavy at large `--num_envs`; use few envs to actually watch a fish.
- **`env.debug_print=true`** prints a per-step line merging throughput + blow-up status:
  `[ep N step M] … env0_step=… jvel_max=…/200 | blowups now/cum/distinct_envs/resets …` with a
  `<<< BLOWUP THIS STEP` tag, plus a per-env `ep…ENV…BLOWUP (reason, jvel=…, step=…)` detail line.
  The stability signal is `blowups cum=0` holding PAST epoch ~4–6 (when the gait coheres).
- **env0 trajectory recorder** (no camera needed): writes one PNG+OBJ every 5 episodes (keep ALL,
  rolling deletion removed) into a PER-RUN folder `demo_out/swim_trajectories/run_<timestamp>/`, plus a
  live `traj_live_env0.png`/`.obj` refreshed every 150 control steps so the path can be watched building
  mid-episode. Knobs: `cfg.traj_save_every_n_episodes`, `cfg.traj_live_every`.
- **Kill leftover runs PID-safe**: `simulation_app.close()` hangs (Blackwell+FEM), and the script's
  cleanup `pkill` only matches `--headless` runs, so a GUI (non-headless) run must be killed by PID
  (`kill -KILL` the `train_ppo.py` process whose `/proc/<pid>/comm` is `python`).

## Gotchas

- **Multi-GPU NCCL is broken on this box** — `torchrun --distributed` throws "CUDA illegal
  memory access" in the first cross-GPU broadcast even with `NCCL_P2P_DISABLE=1`/
  `NCCL_CUMEM_ENABLE=0`. Use a single GPU (or two independent single-GPU runs).
- **`simulation_app.close()` HANGS** in Kit teardown here (Blackwell + FEM); kill leftover
  headless runs by PID.
- Large assets (`fluid_env/Water_env.usd` 182 MB, wandb runs, `demo_out/`, `*.deb`,
  `pointclouds/`) are git-ignored — they exceed GitHub's 100 MB limit or are generated.
  `salmon_IL` needs `fluid_env/Water_env.usd` (obtain via git-LFS / out-of-band); the swim
  task does not.
