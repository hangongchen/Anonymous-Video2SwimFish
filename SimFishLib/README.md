# SimFishLib — multi-view video → controllable fish USD

End-to-end pipeline that turns multi-view fish video into a rigged, Isaac
Sim-ready articulated USD asset:

```
multi-view video
  → silhouette carve (SC)        # segment fish per view, space-carve a voxel volume
  → coarse mesh                  # marching-cubes the carved volume
  → refined mesh                 # flow-transformer implicit-occupancy refiner
  → VLM-generated skeleton       # Qwen3-VL places template bones inside the mesh
  → fish asset pipeline          # deterministic Blender → USD physics builder
  → fish_articulated.usd
```

## Stages

**1. Multi-view video → silhouette carve → coarse mesh**
- `scripts/segment_zef_fish.py` — segment the fish out of each view (black background frames/video).
- `scripts/carve_voxel_from_two_views.py` — space-carve a voxel volume from the segmented views and export a coarse mesh.

**2. Refine mesh (flow-transformer refiner)**
- `simfishlib/experimental/flow_transformer_refiner/` — implicit-occupancy flow refiner (3D U-Net feature backbone + implicit decoder + flow matching). Output mesh resolution is decoupled from the 64³ coarse input.
- `simfishlib/experimental/voxel_refiner/` (subset: `dataset`, `degradation`, `mesh_target`) — coarse-source mixing, degradation, and mesh occupancy targets used by the refiner dataset.
- `scripts/experimental/refine_carved_voxel.py` — run the trained refiner on a carved coarse voxel + source video.
- Training: `scripts/experimental/train_flow_refiner_implicit.py`, `scripts/experimental/build_carve_coarse_dataset.py`, `scripts/experimental/precompute_flow_implicit.py`, launch scripts `scripts/train_flow_refiner_implicit_ddp.sh`, `scripts/run_flow_implicit_thicken_v3.sh`, `scripts/run_flow_implicit_carve_v1.sh`, `scripts/run_flow_implicit_carve_v2_edge.sh`.
- Eval/inference: `scripts/experimental/evaluate_flow_refiner_implicit.py`, `scripts/experimental/infer_testset_flow_implicit.py`.

**3. VLM-generated skeleton**
- `simfishlib/experimental/vlm_rigger/` — a fine-tuned Qwen3-VL builds a fish skeleton by placing template bones one action per turn (add/adjust/remove/stop), viewing an overlay render each turn. Bones are constrained to stay inside the fish mesh. Output is a GT-style blend (fish mesh + `bone1..boneN` meshes, no armature).
  - `rig_loop.py` — runtime driver (VLM policy + deterministic executor, containment enforcement).
  - `render_overlay.py` — Blender overlay renderer + GT-style blend saver.
  - `extract_bone_templates.py`, `gen_trajectories.py`, `build_sft_jsonl.py` — Stage-0 template extraction and SFT trajectory synthesis.
  - `check_gt_containment.py`, `check_head_detect.py` — calibration utilities.
- `simfishlib/inference/qwen3vl.py` — Qwen3-VL client used by the rig loop.
- `scripts/download_qwen3_vl.py`, `scripts/train_qwen3vl_rigger_ms_swift.sh` — fetch base model and LoRA fine-tune the rigger.

**4. Fish asset pipeline → USD**
- `fish_asset_pipeline/` — deterministic Blender → Isaac Sim USD builder (vendored USD-stage subset).
  - `scripts/generate_single_fish_usd.sh` — Linux end-to-end driver for a VLM-rigged blend: scale to real length → extract skeleton → export USDA → add PhysX physics/joints/attachments/mass → convert to binary USD.
  - `scripts/scale_skeleton_blend_to_real_length.py`, `extract_fish_skeleton.py`, `export_blend_to_usd.py`, `add_isaac_physics_to_usd.py`, `convert_usda_to_usd.py`.
  - `configs/usd_physics_defaults.json` — physics/mass/joint defaults.
  - `generated_usd_dataset/vlm_rigged_single_fish/zef_f0001/` — example output: the 2-view silhouette-carved zebrafish rigged by the VLM and converted to `fish_articulated.usd` (7 bones, 6 D6 joints).

The full `fish_asset_pipeline` (with its GT template dataset and V19 reference USD set) lives in its own repository; only the USD-stage code, config, and the produced asset are vendored here.

## Requirements

- Python deps: `requirements.txt`.
- Blender 5.0.x (its bundled Python provides `pxr` for the USD steps). GT-style blends require Blender 5.0 (4.x cannot read them).
- Qwen3-VL for the rigger stage (`scripts/download_qwen3_vl.py`).
