# Path configuration

This release contains no absolute paths. Everything is resolved from environment variables,
with sensible defaults derived from the repository location (see `repo_root.py`).

| variable | meaning | default |
|---|---|---|
| `FISH_ROOT` | this repository | directory containing `repo_root.py` |
| `FISH_PYTHON` | python of the Isaac Sim / Isaac Lab conda env | must be set |
| `CONDA_ENV` | that conda environment's prefix | must be set |
| `ISAACLAB_PATH` | a source checkout of Isaac Lab (2.3.0, Isaac Sim 5.1) | `$FISH_ROOT/../isaaclab` |
| `ZEF_ROOT` | the 3D-ZeF `ZebraFish-05` dataset | `$FISH_ROOT/data/ZebraFish-05` |
| `BLENDER_ROOT` | directory holding a Blender 5.0.x build | must be set |

```bash
export FISH_ROOT=$(pwd)
export FISH_PYTHON=/path/to/conda/envs/isaaclab/bin/python
export CONDA_ENV=/path/to/conda/envs/isaaclab
export ISAACLAB_PATH=/path/to/isaaclab
export ZEF_ROOT=/path/to/ZebraFish-05
export BLENDER_ROOT=/path/to/blender
```

Shell scripts expand these directly. Python modules expand them through the `_P()` helper
injected at the top of each script, which falls back to the repository location when
`FISH_ROOT` is unset.

## API keys

No credentials are stored in this repository. Services are read from files in the user's home
directory, created by the user:

| file | service |
|---|---|
| `~/.meshy_api_key` | Meshy image-to-3D |
| `~/.wandb_key_<name>` | Weights & Biases |
| `~/.cache/huggingface/token` | Hugging Face Hub |

## Data and assets not included

Raw videos, reconstructed USD assets, FEM caches, checkpoints and Weights & Biases runs are
excluded by `.gitignore` because of their size. The W&B entity/project identifiers in the
training scripts are placeholders (`ANON-ENTITY`) and must be replaced with your own.
