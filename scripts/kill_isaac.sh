#!/usr/bin/env bash
# kill_isaac.sh -- force-kill ALL Isaac Sim / Isaac Lab processes on this box.
#
# Why a script: Isaac's simulation_app.close() HANGS here (Blackwell + FEM), and the training
# scripts' pkill only matches --headless runs -- so GUI plays and orphaned sims must be SIGKILLed
# by PID. This finds Isaac processes TWO ways and kills each full process tree:
#   (1) GPU compute contexts  -- a live sim always holds GPU memory (most reliable signal), and
#   (2) command-line markers  -- train_ppo.py / play.py / isaaclab / kit / the isaac conda-env python
#       (catches launchers, bash wrappers, and sims still booting before they touch the GPU).
#
# Usage:
#   scripts/kill_isaac.sh                 # kill everything Isaac
#   scripts/kill_isaac.sh --dry-run       # just LIST what would be killed (kills nothing)
#   scripts/kill_isaac.sh -f play.py      # only kill procs whose cmdline also matches this filter
#
set -u

DRY=0
FILTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|-n) DRY=1 ;;
    -f|--filter)  FILTER="${2:-}"; shift ;;
    -h|--help)    grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown arg: $1 (see --help)"; exit 2 ;;
  esac
  shift
done

# command-line markers that identify an Isaac Sim / Isaac Lab process on this box.
# NOTE: deliberately do NOT match "isaaclab" -- the conda env is named env_isaaclab_51, so its
# interpreter path (env_isaaclab_51/bin/python) contains "isaaclab" and would match EVERY python from
# the env, including VSCode language servers. Match Isaac-specific markers only; the main sims are
# also caught by GPU usage, and their torch/kit worker children by the descendant tree-kill.
PATTERNS='train_ppo\.py|play\.py|isaac_[a-z_]*\.py|/isaacsim/|omni\.kit|/kit/kit|simulation_app|isaac-sim'

SELF=$$
declare -A SEEN
TARGETS=()

add_pid() {
  local pid="${1//[[:space:]]/}"
  [ -z "$pid" ] && return
  [ "$pid" = "$SELF" ] && return
  [ -d "/proc/$pid" ] || return
  [ -n "${SEEN[$pid]:-}" ] && return
  local cmd; cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
  case "$cmd" in
    *kill_isaac*) return ;;                                    # never target this script itself
    *.vscode-server*|*ms-python*|*lsp_server*|*pylance*|*jedi*) return ;;   # never kill editor/LSP servers
  esac
  [ -n "$FILTER" ] && case "$cmd" in *"$FILTER"*) : ;; *) return ;; esac
  SEEN[$pid]=1
  TARGETS+=("$pid")
}

# (1) GPU compute processes -- a running sim always holds GPU memory (kept in the main shell via
#     process substitution so the array writes in add_pid persist)
while read -r pid; do add_pid "$pid"; done < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

# (2) command-line markers -- launchers, wrappers, sims not yet on the GPU
while read -r pid; do add_pid "$pid"; done < <(pgrep -f "$PATTERNS" 2>/dev/null)

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "No Isaac Sim processes found."
  exit 0
fi

echo "Isaac Sim / Isaac Lab processes found (${#TARGETS[@]}):"
for pid in "${TARGETS[@]}"; do
  comm=$(cat "/proc/$pid/comm" 2>/dev/null)
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)
  echo "  pid=$pid [$comm]  ${cmd:-<gone>}"
done

if [ "$DRY" = 1 ]; then
  echo "(dry-run: nothing killed. Re-run without --dry-run to kill.)"
  exit 0
fi

# full descendant tree of a pid (Isaac may spawn child processes)
descendants() { local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do echo "$c"; descendants "$c"; done; }

echo "Killing (SIGKILL, whole tree; close() hangs so no graceful term)..."
for pid in "${TARGETS[@]}"; do
  for c in $(descendants "$pid"); do kill -KILL "$c" 2>/dev/null; done
  kill -KILL "$pid" 2>/dev/null
done

sleep 2
LEFT=()
for pid in "${TARGETS[@]}"; do [ -d "/proc/$pid" ] && LEFT+=("$pid"); done
if [ ${#LEFT[@]} -eq 0 ]; then
  echo "Done -- all ${#TARGETS[@]} process(es) killed."
else
  echo "WARNING: still alive after SIGKILL: ${LEFT[*]}"
  echo "  (zombie awaiting parent reap, or not owned by you -- check 'ps -o pid,stat,user -p ${LEFT[*]}')"
fi

echo "GPU memory now:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null
