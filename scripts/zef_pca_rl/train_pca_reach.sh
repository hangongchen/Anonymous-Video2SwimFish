#!/usr/bin/env bash
# Train PPO on the PCA-manifold RANDOM-TARGET REACHING task (multi-reach episodes).
# Usage: bash scripts/zef_pca_rl/train_pca_reach.sh [--num_envs N] [extra train_ppo.py args...]
set -euo pipefail
REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV="${DEV:-cuda:1}"
HEADLESS="${HEADLESS:-1}"
ARGS=(--task Template-Salmon-Swim-PCA-Reach-Misty-Direct-v0 --device "$DEV" --num_envs 256)
if [[ "$HEADLESS" == "1" ]]; then ARGS+=(--headless); fi
cd "$REPO"
exec "$PY" scripts/rl_games/train_ppo.py "${ARGS[@]}" "$@"
