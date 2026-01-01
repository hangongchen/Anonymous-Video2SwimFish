# How to Run the FISH Isaac Lab Training

This repo registers the tasks `Template-Fish-Direct-v0` and `Template-Fish-Marl-Direct-v0` as Isaac Lab extensions. Use the steps below for the initial setup and for subsequent training runs.

## One-Time Setup

1. **Create/activate the Isaac Lab conda env**
   ```bash
   cd /home/hangong/Reproduction/IsaacLab4.5.0
   ./isaaclab.sh -c env_isaaclab      # creates env if missing
   conda activate env_isaaclab
   ```
2. **Install Isaac Lab into that env**
   ```bash
   ./isaaclab.sh -i
   ```
3. **Install this external project in editable mode**
   ```bash
   cd /home/hangong/Reproduction/FISH
   /home/hangong/Reproduction/IsaacLab4.5.0/isaaclab.sh -p -m pip install -e source/FISH
   ```
4. **Verify the tasks register**
   ```bash
   cd /home/hangong/Reproduction/IsaacLab4.5.0
   ./isaaclab.sh -p /home/hangong/Reproduction/FISH/scripts/list_envs.py
   ```

## Subsequent Runs
1. (Optional) Re-run the editable install if you changed files under `source/FISH`.
2. Launch training (example with nine environments and a custom target offset):
   ```bash
   export WANDB_MODE=online
   export HYDRA_FULL_ERROR=1
   LOG_DIR=/home/hangong/Reproduction/FISH/logs && mkdir -p "$LOG_DIR" \
   && LOG_FILE="$LOG_DIR/isaaclab_train_$(date +%Y%m%d_%H%M%S).txt" \
   && source /home/hangong/anaconda3/etc/profile.d/conda.sh \
   && conda activate env_isaaclab \
   && cd /home/hangong/Reproduction/IsaacLab4.5.0 \
   && ./isaaclab.sh -p /home/hangong/Reproduction/FISH/scripts/rl_games/train_ppo.py \
       --task Template-Fish-Direct-v0 \
       --num_envs 9 \
       --device cuda:0 \
       --target_offset_xy 2 2 \
       --debug_logs \
       --track \
       --wandb-project-name fish_articulation_analytic_water \
       --wandb-entity ANON-ENTITY \
       --wandb-name fish_run_1 \
       --weave 2>&1 | python -u /home/hangong/Reproduction/FISH/scripts/log_tail.py "$LOG_FILE" 10000
   ```
   - Add `--video`/`--enable_cameras` if you need renders.
   - Hydra overrides (e.g., `env.sim.physics_dt=0.005`) go after the script path.
   - `--target_offset_xy X Y` shifts each env's target relative to its own origin (default `5 5` meters).
   - `--horizon_length N` overrides the number of simulator steps collected per PPO update (default comes from the YAML agent config).
   - `--decimation N` keeps each action applied for `N` physics steps (useful when joints need more time to respond).
   - `--action_scale S` changes the torque multiplier sent to the implicit actuators.
   - `--debug_logs` prints per-step action/torque/joint values from env `0` for troubleshooting.
   - `--track` enables Weights & Biases logging (requires `wandb` installed). Use `--wandb-project-name`, `--wandb-entity`, and `--wandb-name` to set project/team/run name.
   - `--weave` enables weave logging (uses project `ANON-ENTITY/fish_articulation`).
2. Launch SAC training (same arguments, different script):
   ```bash
   ./isaaclab.sh -p /home/hangong/Reproduction/FISH/scripts/rl_games/train_sac.py --task Template-Fish-Direct-v0 ...
   ```

### Adjusting the Target Location

Use the `--target_offset_xy` flag whenever you want the fish to swim toward a different goal. The values are meters in the simulator plane and apply uniformly to every replicated env:
```bash
./isaaclab.sh -p .../train_ppo.py --task Template-Fish-Direct-v0 --num_envs 4 --target_offset_xy 2.0 -3.5
```
In this example each environment spawns a red marker 2 m forward and 3.5 m to the left of the fish's origin.

## Tips

- Always go through `isaaclab.sh -p …` so Kit is bootstrapped before Python runs.
- Warnings about `rendering_modes`, `omni.materialx.libs`, or `omni.isaac.dynamic_control` are informational.
- Set `HYDRA_FULL_ERROR=1` before launching if you need full stack traces for bad overrides.

## Fish slide demo
  DISPLAY=:1 python scripts/demo_hydro_swim_deform.py --model mujoco --deform_glide --live --free_cam \
    --shove 0.3 --shove_at 2 --seconds 10
## skeleton demo
DISPLAY=:1 ${FISH_PYTHON} scripts/demo_hydro_swim.py \
    --model mujoco --swim --live --free_cam \
    --shove 0.3 --shove_at 2 --gait_at 4 --seconds 8 --amp 0.24 --freq 2.0
## deformable run
  PY=${FISH_PYTHON}
  cd ${FISH_ROOT}
  DISPLAY=:1 $PY scripts/demo_hydro_swim_deform_swim.py --model mujoco --deform_swim --live --free_cam \
    --shove 0.3 --shove_at 2 --gait_at 4 --seconds 6 --amp 0.2 --freq 1.5
## RL

  cd ${FISH_ROOT}
  export WANDB_MODE=online
  export HYDRA_FULL_ERROR=1
  DISPLAY=:1 ${FISH_PYTHON} -m torch.distributed.run \
      --nnodes=1 --nproc_per_node=2 \
      scripts/rl_games/train_ppo.py \
      --task Template-Salmon-Swim-Direct-v0 \
      --num_envs 12 --distributed \
      --target_offset_xy 1 1 \
      --track \
      --wandb-project-name fish_articulation_analytic_water \
      --wandb-entity ANON-ENTITY \
      --wandb-name salmon_swim_2gpu

  bash scripts/train_salmon_swim_2gpu.sh --num_envs 2048   