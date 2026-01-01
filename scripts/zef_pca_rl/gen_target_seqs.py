"""Deterministic evaluation target sequences: seed -> per-(env, attempt) RELATIVE draws.

Targets in this task are pose-conditioned (bearing relative to the fish's CURRENT heading),
so absolute positions cannot be pre-fixed without changing the task. What CAN be fixed --
and what makes the two methods face identical draws -- is the RELATIVE parameter pair per
(env, attempt): (bearing offset in the +/-15 deg cone, distance ~ N(1.0, 0.25) clamped).
Both methods consume the SAME file, so attempt i of env e uses the same offset+distance for
CPG and PCA regardless of where each fish is.

Usage: <env python> scripts/zef_pca_rl/gen_target_seqs.py --seed 101 --n_envs 16 --n_att 10
Writes demo_out/reach10_eval/target_seq_seed<k>.npz (+ readable txt dump).
"""

import argparse
from pathlib import Path

import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n_envs", type=int, default=16)
ap.add_argument("--n_att", type=int, default=10)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
off = rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0), size=(a.n_envs, a.n_att))
dist = np.clip(rng.normal(1.0, 0.25, size=(a.n_envs, a.n_att)), 0.4, 1.8)
out = Path(_P("${FISH_ROOT}/demo_out/reach10_eval"))
out.mkdir(parents=True, exist_ok=True)
p = out / f"target_seq_seed{a.seed:03d}.npz"
np.savez(p, bearing_off=off.astype(np.float32), dist=dist.astype(np.float32), seed=a.seed)
with open(out / f"target_seq_seed{a.seed:03d}.txt", "w") as f:
    for e in range(a.n_envs):
        f.write(f"env {e}: " + "  ".join(
            f"t{i+1}=({np.rad2deg(off[e,i]):+.1f}deg,{dist[e,i]:.2f}m)" for i in range(a.n_att)) + "\n")
print(f"wrote {p}")
print("env0:", "  ".join(f"t{i+1}=({np.rad2deg(off[0,i]):+.1f}deg,{dist[0,i]:.2f}m)" for i in range(a.n_att)))
