#!/usr/bin/env bash
# Train PPO in the 2-PC PCA locomotion-manifold action space (straight-forward-swim task).
# Usage: bash scripts/zef_pca_rl/train_pca_swim.sh [--num_envs N] [extra train_ppo.py args...]
# DEV=cuda:1 to pick the GPU. Isolated from the AMP experiments (own task id + agent yaml).
set -euo pipefail
REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV="${DEV:-cuda:0}"
HEADLESS="${HEADLESS:-1}"
ARGS=(--task Template-Salmon-Swim-PCA-Misty-Direct-v0 --device "$DEV" --num_envs 256)
if [[ "$HEADLESS" == "1" ]]; then ARGS+=(--headless); fi
cd "$REPO"
exec "$PY" scripts/rl_games/train_ppo.py "${ARGS[@]}" "$@"
