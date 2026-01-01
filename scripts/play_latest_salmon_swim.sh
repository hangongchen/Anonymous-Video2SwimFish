#!/usr/bin/env bash
# Launch the latest salmon-swim RL-Games checkpoint in Isaac Sim.
#
# Defaults match the requested single-fish soft-body playback setup:
# - GUI on `cuda:1`
# - `NUM_ENVS=1`
# - soft body ON
# - latest checkpoint from the latest salmon-swim run
# - `dt=1/960`, `decimation=32`, `episode_length_s=50.0` to match training
#
# You can also pass a checkpoint explicitly:
#   bash scripts/play_latest_salmon_swim.sh logs/rl_games/.../nn/cartpole_direct.pth
set -euo pipefail

REPO=${FISH_ROOT}
PY=${PY:-${FISH_PYTHON}}
DEV=${DEV-}
NUM_ENVS=${NUM_ENVS-}
HEADLESS=${HEADLESS:-0}
REAL_TIME=${REAL_TIME:-1}
CHECKPOINT_MODE=${CHECKPOINT_MODE:-best}
POLICY_MODE=${POLICY_MODE:-deterministic}
TRAIN_MATCH=${TRAIN_MATCH:-0}
PLAY_TRAJ_DIR=${PLAY_TRAJ_DIR-}
PLAY_TRAJ_LIVE_EVERY=${PLAY_TRAJ_LIVE_EVERY-}
PLAY_STEP_LOG=${PLAY_STEP_LOG:-0}
PLAY_STEP_LOG_DIR=${PLAY_STEP_LOG_DIR:-${FISH_ROOT}/demo_out/play_step_logs}
RESET_DEFORMABLE=${RESET_DEFORMABLE-}
DEBUG_PRINT=${DEBUG_PRINT-}
RECORD_VIDEO=${RECORD_VIDEO-}
RECORD_TRAJECTORY=${RECORD_TRAJECTORY-}
TRAJ_SAVE_EVERY=${TRAJ_SAVE_EVERY-}
MARKER_SUCCESS_SPHERE=${MARKER_SUCCESS_SPHERE-}

latest_file_by_mtime() {
    local dir="$1"
    shift
    find "$dir" -maxdepth 1 -type f "$@" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/play_latest_salmon_swim.sh [checkpoint.pth] [extra hydra overrides...]

Environment variables:
  DEV=cuda:1             Isaac/torch device passed to play.py
  NUM_ENVS=1             Number of envs to play
  HEADLESS=0             Set to 1 for headless play
  REAL_TIME=1            Set to 0 to run as fast as possible
  CHECKPOINT_MODE=best   "best" uses the non-last .pth, "last" uses the newest last_*.pth
  POLICY_MODE=deterministic
                         "deterministic" uses the policy mean, "stochastic" samples actions
  TRAIN_MATCH=0          Set to 1 to inherit device/num_envs/reset flags from training env.yaml
  PLAY_TRAJ_DIR=.../play trajectories
                         Root folder for saved play trajectories (each run gets a subfolder)
  PLAY_TRAJ_LIVE_EVERY=150
                         Also overwrite traj_live_env0.{png,obj} every N control steps
  PLAY_STEP_LOG=0        Set to 1 to write per-step JSONL telemetry
  PLAY_STEP_LOG_DIR=.../play_step_logs
                         Root folder for per-step play logs
  SIM_DT=0.0010416666666666667
                         Physics dt override for play
  DECIMATION=32          Physics steps per action override for play
  EPISODE_LENGTH_S=50.0 Episode length override for play

Notes:
  - The latest salmon-swim run is detected from saved params/env.yaml files, not the
    experiment name, because the current RL-Games config still logs under cartpole_direct.
  - Playback is always soft-body ON. This launcher no longer supports disabling the deformable body.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

CKPT=""
if [ "$#" -gt 0 ] && { [ -f "$1" ] || [[ "$1" == *.pth ]]; }; then
    CKPT="$1"
    shift
fi
EXTRA_ARGS=("$@")

cd "$REPO"
export DISPLAY=${DISPLAY:-:1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONPATH="$REPO/source${PYTHONPATH:+:$PYTHONPATH}"

latest_run_dir=""
latest_env_yaml=""
trained_soft="unknown"
latest_run_stamp=""
trained_device=""
trained_dt=""
trained_decimation=""
trained_episode_length_s=""
trained_num_envs=""
trained_reset_deformable=""
trained_debug_print=""
trained_record_video=""
trained_record_trajectory=""
trained_traj_dir=""
trained_traj_save_every=""
trained_traj_live_every=""
trained_marker_success_sphere=""

if [ -z "$CKPT" ]; then
    while IFS= read -r env_yaml; do
        if ! grep -q '^per_bone_hydro_path:' "$env_yaml"; then
            continue
        fi

        run_dir=$(dirname "$(dirname "$env_yaml")")
        nn_dir="$run_dir/nn"
        if [ ! -d "$nn_dir" ]; then
            continue
        fi

        if ! find "$nn_dir" -maxdepth 1 -type f -name '*.pth' | grep -q .; then
            continue
        fi

        run_stamp=$(basename "$run_dir")
        if [ -z "$latest_run_stamp" ] || [[ "$run_stamp" > "$latest_run_stamp" ]]; then
            latest_run_stamp="$run_stamp"
            latest_run_dir="$run_dir"
            latest_env_yaml="$env_yaml"
        fi
    done < <(find logs/rl_games -path '*/params/env.yaml' | sort)

    if [ -z "$latest_run_dir" ]; then
        echo "[play] ERROR: no salmon-swim training run found under logs/rl_games." >&2
        exit 1
    fi

    case "$CHECKPOINT_MODE" in
        best)
            CKPT=$(latest_file_by_mtime "$latest_run_dir/nn" -name '*.pth' ! -name 'last_*')
            if [ -z "$CKPT" ]; then
                CKPT=$(latest_file_by_mtime "$latest_run_dir/nn" -name 'last_*.pth')
            fi
            ;;
        last)
            CKPT=$(latest_file_by_mtime "$latest_run_dir/nn" -name 'last_*.pth')
            if [ -z "$CKPT" ]; then
                CKPT=$(latest_file_by_mtime "$latest_run_dir/nn" -name '*.pth')
            fi
            ;;
        *)
            echo "[play] ERROR: CHECKPOINT_MODE must be 'best' or 'last'." >&2
            exit 1
            ;;
    esac
else
    latest_run_dir=$(dirname "$(dirname "$CKPT")")
    latest_env_yaml="$latest_run_dir/params/env.yaml"
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[play] ERROR: could not resolve a checkpoint to play." >&2
    exit 1
fi

if [ -f "$latest_env_yaml" ]; then
    trained_soft=$(grep -m1 '^with_deformable:' "$latest_env_yaml" | awk '{print $2}')
    trained_device=$(awk '/^sim:/{in_sim=1; next} in_sim && /^  device:/{print $2; exit} /^[^ ]/{in_sim=0}' "$latest_env_yaml")
    trained_dt=$(awk '/^sim:/{in_sim=1; next} in_sim && /^  dt:/{print $2; exit} /^[^ ]/{in_sim=0}' "$latest_env_yaml")
    trained_decimation=$(awk '/^decimation:/{print $2; exit}' "$latest_env_yaml")
    trained_episode_length_s=$(awk '/^episode_length_s:/{print $2; exit}' "$latest_env_yaml")
    trained_num_envs=$(awk '/^scene:/{in_scene=1; next} in_scene && /^  num_envs:/{print $2; exit} /^[^ ]/{in_scene=0}' "$latest_env_yaml")
    trained_reset_deformable=$(grep -m1 '^reset_deformable:' "$latest_env_yaml" | awk '{print $2}')
    trained_debug_print=$(grep -m1 '^debug_print:' "$latest_env_yaml" | awk '{print $2}')
    trained_record_video=$(grep -m1 '^record_video:' "$latest_env_yaml" | awk '{print $2}')
    trained_record_trajectory=$(grep -m1 '^record_trajectory:' "$latest_env_yaml" | awk '{print $2}')
    trained_traj_dir=$(grep -m1 '^traj_dir:' "$latest_env_yaml" | awk '{print $2}')
    trained_traj_save_every=$(grep -m1 '^traj_save_every_n_episodes:' "$latest_env_yaml" | awk '{print $2}')
    trained_traj_live_every=$(grep -m1 '^traj_live_every:' "$latest_env_yaml" | awk '{print $2}')
    trained_marker_success_sphere=$(grep -m1 '^marker_success_sphere:' "$latest_env_yaml" | awk '{print $2}')
fi

if [ "$TRAIN_MATCH" = "1" ]; then
    DEV=${DEV:-${trained_device:-cuda:0}}
    NUM_ENVS=${NUM_ENVS:-${trained_num_envs:-512}}
    RESET_DEFORMABLE=${RESET_DEFORMABLE:-${trained_reset_deformable:-false}}
    DEBUG_PRINT=${DEBUG_PRINT:-${trained_debug_print:-true}}
    RECORD_VIDEO=${RECORD_VIDEO:-${trained_record_video:-false}}
    RECORD_TRAJECTORY=${RECORD_TRAJECTORY:-${trained_record_trajectory:-true}}
    PLAY_TRAJ_DIR=${PLAY_TRAJ_DIR:-${trained_traj_dir:-${FISH_ROOT}/demo_out/swim_trajectories}}
    TRAJ_SAVE_EVERY=${TRAJ_SAVE_EVERY:-${trained_traj_save_every:-5}}
    PLAY_TRAJ_LIVE_EVERY=${PLAY_TRAJ_LIVE_EVERY:-${trained_traj_live_every:-150}}
    MARKER_SUCCESS_SPHERE=${MARKER_SUCCESS_SPHERE:-${trained_marker_success_sphere:-false}}
else
    DEV=${DEV:-cuda:1}
    NUM_ENVS=${NUM_ENVS:-1}
    RESET_DEFORMABLE=${RESET_DEFORMABLE:-true}
    DEBUG_PRINT=${DEBUG_PRINT:-false}
    RECORD_VIDEO=${RECORD_VIDEO:-false}
    RECORD_TRAJECTORY=${RECORD_TRAJECTORY:-true}
    PLAY_TRAJ_DIR=${PLAY_TRAJ_DIR:-${FISH_ROOT}/demo_out/play trajectories}
    TRAJ_SAVE_EVERY=${TRAJ_SAVE_EVERY:-1}
    PLAY_TRAJ_LIVE_EVERY=${PLAY_TRAJ_LIVE_EVERY:-150}
    MARKER_SUCCESS_SPHERE=${MARKER_SUCCESS_SPHERE:-false}
fi

SIM_DT=${SIM_DT:-${trained_dt:-0.0010416666666666667}}
DECIMATION=${DECIMATION:-${trained_decimation:-32}}
EPISODE_LENGTH_S=${EPISODE_LENGTH_S:-${trained_episode_length_s:-50.0}}

if [ "${SOFT:-1}" != "1" ]; then
    echo "[play] ERROR: this launcher always keeps the soft body ON. Remove SOFT or set SOFT=1." >&2
    exit 1
fi
case "$POLICY_MODE" in
    deterministic|stochastic) ;;
    *)
        echo "[play] ERROR: POLICY_MODE must be 'deterministic' or 'stochastic'." >&2
        exit 1
        ;;
esac

PLAY_FLAGS=(
    --task Template-Salmon-Swim-Direct-v0
    --num_envs "$NUM_ENVS"
    --device "$DEV"
    --checkpoint "$CKPT"
)

if [ "$HEADLESS" = "1" ]; then
    PLAY_FLAGS+=(--headless)
fi
if [ "$REAL_TIME" = "1" ]; then
    PLAY_FLAGS+=(--real-time)
fi
if [ "$POLICY_MODE" = "stochastic" ]; then
    PLAY_FLAGS+=(--stochastic)
fi

STEP_LOG_PATH=""
STEP_DEBUG_EXTRAS=false
if [ "$PLAY_STEP_LOG" = "1" ]; then
    mkdir -p "$PLAY_STEP_LOG_DIR"
    RUN_STAMP=$(date +%Y%m%d_%H%M%S)
    STEP_LOG_PATH="$PLAY_STEP_LOG_DIR/play_${POLICY_MODE}_${RUN_STAMP}.jsonl"
    PLAY_FLAGS+=(--step_log_path "$STEP_LOG_PATH")
    STEP_DEBUG_EXTRAS=true
fi

TRAJ_FLAGS=()
if [ "$RECORD_TRAJECTORY" = "true" ]; then
    TRAJ_FLAGS=(
        --traj_dir "$PLAY_TRAJ_DIR"
        --traj_save_every "$TRAJ_SAVE_EVERY"
        --traj_live_every "$PLAY_TRAJ_LIVE_EVERY"
    )
fi

HYDRA_OVERRIDES=(
    env.record_video="$RECORD_VIDEO"
    env.marker_success_sphere="$MARKER_SUCCESS_SPHERE"
    env.record_trajectory="$RECORD_TRAJECTORY"
    env.traj_save_every_n_episodes="$TRAJ_SAVE_EVERY"
    env.traj_live_every="$PLAY_TRAJ_LIVE_EVERY"
    env.sim.dt="$SIM_DT"
    env.decimation="$DECIMATION"
    env.sim.render_interval="$DECIMATION"
    env.episode_length_s="$EPISODE_LENGTH_S"
    env.with_deformable=true
    env.reset_deformable="$RESET_DEFORMABLE"
    env.debug_print="$DEBUG_PRINT"
    env.step_debug_extras="$STEP_DEBUG_EXTRAS"
)
if [ "$NUM_ENVS" -eq 1 ] && [ "$TRAIN_MATCH" != "1" ]; then
    echo "[play] WARNING: soft-body play with NUM_ENVS=1 can still differ numerically from the many-env training run." >&2
fi

echo "[play] run        : ${latest_run_dir:-<external checkpoint>}"
echo "[play] checkpoint : $CKPT"
echo "[play] mode       : $( [ "$TRAIN_MATCH" = "1" ] && printf 'train-match' || printf 'play-default' )"
echo "[play] device     : $DEV"
echo "[play] num_envs   : $NUM_ENVS"
echo "[play] policy     : $POLICY_MODE"
echo "[play] soft mode  : ON (trained with_deformable=${trained_soft})"
echo "[play] physics    : dt=$SIM_DT decimation=$DECIMATION episode_length_s=$EPISODE_LENGTH_S"
echo "[play] reset/debug: reset_deformable=$RESET_DEFORMABLE debug_print=$DEBUG_PRINT"
echo "[play] trajectories: $PLAY_TRAJ_DIR (enabled=$RECORD_TRAJECTORY save_every=$TRAJ_SAVE_EVERY live_every=$PLAY_TRAJ_LIVE_EVERY)"
if [ -n "$STEP_LOG_PATH" ]; then
    echo "[play] step log   : $STEP_LOG_PATH"
fi

exec "$PY" scripts/rl_games/play.py \
    "${TRAJ_FLAGS[@]}" \
    "${PLAY_FLAGS[@]}" \
    "${HYDRA_OVERRIDES[@]}" \
    "${EXTRA_ARGS[@]}"
