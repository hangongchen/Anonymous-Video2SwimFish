#!/usr/bin/env bash
# Watch the BEST trained salmon-swim policy in the GUI, on cuda:1 by default (so it does NOT
# fight a training run on cuda:0). Soft body ON, deterministic actions, real-time.
#
# Usage:
#   bash scripts/play_salmon_swim_gui.sh                      # auto-pick latest run's BEST checkpoint
#   bash scripts/play_salmon_swim_gui.sh path/to/ckpt.pth     # a specific checkpoint
#   NUM_ENVS=1 bash scripts/play_salmon_swim_gui.sh           # single fish
#   DEV=cuda:0 bash scripts/play_salmon_swim_gui.sh           # other GPU (e.g. if nothing is training)
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:1}
NUM_ENVS=${NUM_ENVS:-4}
# SOFT=1 (default): FEM soft body ON. Play runs STOCHASTIC (--stochastic) -- a DETERMINISTIC
#   mean gait is perfectly coherent and pumps energy into the FEM until it diverges mid-swim;
#   sampling actions (like training did) breaks up that coherence -> stable (verified: 512 envs
#   + stochastic = 0 blow-ups, exactly like training).
# SOFT=0: skeleton-only (no FEM), a deterministic-safe fallback.
SOFT=${SOFT:-1}

cd "$REPO"
export DISPLAY=${DISPLAY:-:1}
export HYDRA_FULL_ERROR=1

# Checkpoint: arg 1 if given, else the most recent run's BEST one ("<experiment>.pth", i.e. the
# non-"last_" .pth that rl_games writes for the best mean reward).
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    RUNDIR=$(ls -dt logs/rl_games/*/2026-*/ 2>/dev/null | head -1)
    CKPT=$(ls -t "${RUNDIR}nn/"*.pth 2>/dev/null | grep -v '/last_' | head -1)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "[play] ERROR: no checkpoint found (looked under logs/rl_games/*/2026-*/nn/). Pass one explicitly."
    exit 1
fi

if [ "$SOFT" = "0" ]; then
    SOFT_ARGS="env.with_deformable=false env.reset_deformable=false"
    echo "[play] SKELETON-ONLY (no FEM)"
else
    SOFT_ARGS="env.with_deformable=true env.reset_deformable=false"   # match training (no FEM reset)
    echo "[play] SOFT BODY ON (stochastic). NOTE: a SINGLE soft fish diverges the FEM -- need MANY"
    echo "       envs for stability (verified stable at 512, blows up at 1). For ONE clean fish use SOFT=0."
    if [ "$NUM_ENVS" -lt 64 ] 2>/dev/null; then
        echo "       WARNING: NUM_ENVS=$NUM_ENVS is small -> the soft body will likely blow up. Raise it or use SOFT=0."
    fi
fi
echo "[play] checkpoint : $CKPT  | device=$DEV  num_envs=$NUM_ENVS  GUI (real-time)"
exec "$PY" scripts/rl_games/play.py \
    --task Template-Salmon-Swim-Direct-v0 \
    --num_envs "$NUM_ENVS" --device "$DEV" \
    --checkpoint "$CKPT" \
    --real-time \
    --stochastic \
    $SOFT_ARGS \
    env.record_video=false \
    env.debug_print=true \
    env.marker_success_sphere=true \
    "${@:2}"
