# Flow-transformer refiner (PhysX-Anything style)

Isolated **first prototype** of a conditional flow-matching refiner. The
existing V2 implicit-occupancy refiner is untouched — this lives in a separate
module (`simfishlib/experimental/flow_transformer_refiner/`) and ships its
own scripts, checkpoints, and tests.

## Pipeline

```
image-trained Qwen voxel64               (Dataset/Fish2VoxelMergedV2QwenCoarse/<asset>/qwen_voxel64.npy)
  -> kept at native 64^3 -> map to [-1, 1] -> coarse condition
+ video frames (or single image)         -> SmallImageEncoder -> 256-D vector
+ timestep t ~ U(0,1) + noise ~ N(0,I)   -> x_t = (1-t)*target + t*noise

           ↓

  FlowTransformerRefiner (DiT-style):
    PatchTokenize 4^3 -> 16^3 = 4096 tokens of dim 64
    concat target_tokens + coarse_tokens -> Linear(128 -> hidden=384) + posembed
    6 × DiTBlock with AdaLN-Zero (cond = SiLU(time_mlp(sin(t)) + visual_proj(vis)))
    final LayerNorm + AdaLN + zero-init Linear -> per-token velocity
    Unpatchify -> velocity volume [B, 1, 64, 64, 64]

           ↓

  Rectified-flow training: MSE( v_pred, noise - target )
  Sampling: Euler integrate dx/dt = v_theta(x, t) from t=1 (noise) to t=0
  Decode: marching cubes on TSDF at iso=0 -> mesh.obj
```

## Target representation

**Truncated SDF (TSDF) at 64³** (default), negative inside / positive outside / 0 at the
surface, clipped to `[-1, 1]` via division by `truncation` (in voxel units,
default 4). Computed from `asset.obj`:

```
asset.obj -> mesh_to_occupancy(R=64) -> scipy.ndimage.distance_transform_edt
          -> clip / truncation -> tsdf in [-1, 1]
```

Cached on disk under `<output-dir>/cache/tsdf_R64_T4.00_<hash>/<asset>.npy`.

## Coarse condition

Same `qwen_voxel64.npy` files the V2 refiner uses. Kept at native 64³ and
rescaled `[0,1] -> [-1,1]` so it shares the same magnitude as the TSDF target
inside the input projection's concat. If `--target-resolution` is set below
64, the coarse voxel is average-pooled to match.

The dataset reuses the V2 `CoarseSourceMix(qwen=0.5, degraded=0.4, near_gt=0.1)`
for training-time augmentation; pass `--coarse-source-mix 1,0,0` for the
Qwen-only ablation.

## Files

```
simfishlib/experimental/flow_transformer_refiner/
  __init__.py            # public API
  target.py              # voxel_to_tsdf, mesh_to_tsdf, downsample_voxel, PatchTokenizer
  flow.py                # sample_x_t, velocity_target, flow_matching_loss, euler_sample
  model.py               # FlowTransformerRefiner + build_model + DiTBlock + AdaLNZero
  visual_encoder.py      # copy of the SmallImageEncoder (isolated)
  dataset.py             # FlowRefinerDataset (reuses V2 discover_samples + split_by_asset)
  inference.py           # sample_tsdf_volume + tsdf_to_mesh
  io.py                  # save/load checkpoint + write_obj + write_json

scripts/experimental/
  train_flow_refiner.py
  evaluate_flow_refiner.py
  infer_flow_refiner.py
  README_flow_transformer_refiner.md   (this file)

scripts/
  train_flow_refiner_smoke.sh          # 2 epochs / 4 assets / single GPU
  train_flow_refiner.sh                # full single-GPU run
  train_flow_refiner_ddp.sh            # 2× H200 DDP

tests/test_flow_transformer_refiner.py # 18 CPU tests
```

## Acceptance smoke (already run)

```
pytest tests/test_flow_transformer_refiner.py -v
  -> 18 passed in 68s
```

End-to-end model sanity:

```
$ python -c "from simfishlib.experimental.flow_transformer_refiner import build_model; ..."
Full model params: ~19.6M
Forward OK: (1, 1, 64, 64, 64)
Sample OK: shape=(1, 1, 64, 64, 64)
```

## tmux commands (do NOT launch from here)

Grab a node first:

```bash
srun --pty --gres=gpu:h200:2 --cpus-per-task=64 --mem=512G --time=24:00:00 bash
```

### A. Smoke training (2 epochs, 4 assets, ~5 minutes on 1× H200)

```bash
tmux new -s flow_smoke
bash scripts/train_flow_refiner_smoke.sh
```

### B. Full single-GPU training

```bash
tmux new -s flow_train
bash scripts/train_flow_refiner.sh
# overrides:  OUTPUT=outputs/experimental/flow_v1 EPOCHS=300 bash ...
```

### C. Two-GPU DDP training

```bash
tmux new -s flow_ddp
bash scripts/train_flow_refiner_ddp.sh
```

### D. Evaluate

```bash
/work/yzha1/miniforge3/envs/simfishlib/bin/python \
  scripts/experimental/evaluate_flow_refiner.py \
  --checkpoint outputs/experimental/flow_transformer_refiner/checkpoint_best.pt \
  --output-dir outputs/experimental/flow_transformer_refiner_eval \
  --num-steps 50
```

Reports `mean_iou_refined`, `mean_iou_raw_qwen`, and the per-asset delta
between them — the central A/B question is "does the flow refiner beat raw
Qwen voxel64 at the same 32³ resolution."

### E. Single-asset inference

```bash
/work/yzha1/miniforge3/envs/simfishlib/bin/python \
  scripts/experimental/infer_flow_refiner.py \
  --coarse-voxel Dataset/Fish2VoxelMergedV2QwenCoarse/<asset>/qwen_voxel64.npy \
  --video Dataset/Fish2VoxelMergedV2Video/<asset>/swim_v00.mp4 \
  --checkpoint outputs/experimental/flow_transformer_refiner/checkpoint_best.pt \
  --output-dir outputs/experimental/flow_transformer_refiner_infer/<asset> \
  --num-steps 50
```

Writes `tsdf.npy`, `occupancy.npy`, `mesh.obj`, `summary.json`.

## Comparing against current refiners

After training:

```bash
# A. raw Qwen voxel64 (already cached) — used as iou_raw_qwen baseline by eval
# B. V2 U-Net/implicit refiner — outputs/experimental/voxel_refiner_v2*/testset_inference/*/mesh.obj
# C. this flow refiner — outputs/experimental/flow_transformer_refiner_eval/*/mesh.obj
```

Render side-by-side via the existing `scripts/experimental/infer_testset_v2.py`
plumbing (it already accepts arbitrary `mesh.obj` paths).

## Knobs likely to matter early

| flag | default | notes |
|---|---|---|
| `--target-resolution` | 64 | 64 = native Qwen voxel; 32 gives 512 tokens (cheaper ablation); 16 even cheaper |
| `--patch` | 4 | must divide resolution evenly; at R=64 → 16³=4096 tokens; bump to 8 if too slow |
| `--hidden-dim / --depth / --num-heads` | 384 / 6 / 6 | DiT-S sized, ~18M params |
| `--num-steps` | 50 | Euler steps at inference; 25/50/100 worth comparing |
| `--coarse-source-mix` | `0.5,0.4,0.1` | qwen / degraded / near_gt during training |
| `--visual-mode` | `video` | switch to `none` for voxel-only ablation |
| `--truncation` | 4.0 | TSDF clamp distance (voxel units) |
| `--lr` | 2e-4 | AdamW; DiT used 1e-4 with longer warmup |

## Limitations of v0

- **No SDF augmentation** — TSDF is computed from `mesh_to_occupancy` rasterized
  at the target resolution, so it inherits the voxel quantization. For finer
  surfaces, swap in an exact mesh-distance computation (e.g. via `mesh.nearest.signed_distance`).
- **Single fixed clip per asset** — no temporal cross-attention; visual feature
  is mean-pooled across K frames (same as V2).
- **No classifier-free guidance** — we don't randomly drop conditioning during
  training. Easy to add if conditioning ends up too weak.
- **TSDF target only** — no latent shape autoencoder. Adding one would let the
  transformer operate on a much smaller token grid (TRELLIS-style).
- **Two-stage marching cubes** — we output a 64³ TSDF and march at iso=0. For
  even higher-res meshes we'd need to either bump `--target-resolution` (which
  blows up token count: 96³ at P=4 → 13 824 tokens) or add a learned upsampler
  post-flow.

## Cleanup

This experiment is fully self-contained. To remove:

```bash
rm -rf simfishlib/experimental/flow_transformer_refiner \
       scripts/experimental/{train,evaluate,infer}_flow_refiner.py \
       scripts/experimental/README_flow_transformer_refiner.md \
       scripts/train_flow_refiner*.sh \
       tests/test_flow_transformer_refiner.py \
       outputs/experimental/flow_transformer_refiner*
git branch -D flow_transformer_refiner_experiment   # if abandoned
```
