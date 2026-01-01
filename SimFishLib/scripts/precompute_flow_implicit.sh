#!/usr/bin/env bash
# Pre-populate the per-asset (points, sdf) cache for the implicit refiner.
# Single short paste -- no env vars to lose, no long path to wrap.
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"

OUTPUT="${OUTPUT:-outputs/experimental/flow_transformer_refiner_implicit}"
mkdir -p "$OUTPUT"

python scripts/experimental/precompute_flow_implicit.py --output-dir "$OUTPUT"
