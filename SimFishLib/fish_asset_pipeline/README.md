# Articulated Fish Asset Pipeline

This project builds a deterministic articulated fish asset generation pipeline for Isaac Sim.

The current pipeline does not ask a VLM to generate USD. Instead, USD generation is handled by Python scripts:

```text
Fish .blend mesh
  -> auto skeleton fitting
  -> skeleton.json extraction
  -> Blender USD export
  -> deterministic Isaac Sim physics USD builder
  -> fish_articulated.usd
```

The VLM part is intentionally out of scope for the current milestone.

## Current Status

The latest working pipeline is V19:

- Fish meshes are scaled to a real-world target length, default `0.5m`.
- Skeletons are generated automatically from the reusable supervised template set.
- Bone count is selected from fish shape instead of forcing one fixed template.
- Fish mesh, bones, and joints are positioned so the fish bottom is near `Z=1.0m`.
- Bone mass is computed from scaled fish mesh volume and water density.
- The mesh signed-volume is only trusted when it is a plausible fraction of the
  bounding-box volume (`mesh_volume_min/max_bbox_fraction`, default 0.15-0.8).
  Open/non-watertight meshes report volumes 100-1000x too small; for those the
  volume is estimated as `bbox_volume * volume_fill_fraction` (default 0.35),
  which gives a 0.5m fish a total bone mass of roughly 1-4kg.
- Deformable fish mesh mass comes from `configs/usd_physics_defaults.json`.
- Joint drivers are removed.
- Joint rotation limits are `-15` to `15`.
- Attachments are created between each bone and the deformable fish mesh.
- `attach rigid surface` metadata is enabled for attachments.
- `GroundPlane` is authored as a mesh collider and is inactive by default.

## One Fish Command

Use this command to generate one Isaac Sim-ready articulated fish USD from one Blender fish mesh:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\Fish_dataset\fish_asset_pipeline\scripts\generate_single_fish_usd.ps1" `
  -InputBlend "E:\path\to\new_fish.blend" `
  -OutputDir "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\single_fish_outputs\new_fish" `
  -TargetLengthM 0.5
```

Outputs:

```text
fish_articulated.usd   binary USD for Isaac Sim
fish_articulated.usda  readable USDA for inspection
```

For debugging, keep intermediate files:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\Fish_dataset\fish_asset_pipeline\scripts\generate_single_fish_usd.ps1" `
  -InputBlend "E:\path\to\new_fish.blend" `
  -OutputDir "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\single_fish_outputs\new_fish" `
  -TargetLengthM 0.5 `
  -KeepIntermediate
```

Intermediate files are written under `_work/`:

```text
_work/auto_skeleton/auto_skeleton.blend
_work/auto_skeleton/skeleton.json
_work/auto_skeleton/fish_mesh.obj
_work/usd/base_export.usda
```

## Batch Commands

Generate auto skeletons and `skeleton.json` for all unlabeled Blender files:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\Fish_dataset\fish_asset_pipeline\scripts\batch_generate_auto_skeletons_v3.ps1" `
  -BlenderExe "E:\blender\blender.exe" `
  -OutputRoot "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\unlabeled_auto_skeleton_v14_real_size_mass"
```

Export an auto-skeleton dataset to articulated USDA:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\Fish_dataset\fish_asset_pipeline\scripts\batch_export_v3_to_usd.ps1" `
  -BlenderExe "E:\blender\blender.exe" `
  -UsdPython "E:\blender\5.1\python\bin\python.exe" `
  -InputRoot "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\unlabeled_auto_skeleton_v14_real_size_mass" `
  -OutputRoot "E:\Fish_dataset\fish_asset_pipeline\generated_dataset\usd_articulated_v19_groundplane_mesh_collider_inactive"
```

Convert final USDA files to binary USD:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\Fish_dataset\fish_asset_pipeline\scripts\batch_convert_final_usda_to_usd.ps1" `
  -UsdPython "E:\blender\5.1\python\bin\python.exe" `
  -InputRoot "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\usd_articulated_final_v19_groundplane_mesh_collider_inactive" `
  -OutputRoot "E:\Fish_dataset\fish_asset_pipeline\generated_usd_dataset\usd_articulated_final_v19_groundplane_mesh_collider_inactive_binary"
```

## Latest Generated Dataset

Readable USDA:

```text
generated_usd_dataset/usd_articulated_final_v19_groundplane_mesh_collider_inactive/
```

Binary USD:

```text
generated_usd_dataset/usd_articulated_final_v19_groundplane_mesh_collider_inactive_binary/
```

There are currently 35 generated binary USD files in the V19 output.

## FEM deformable cook step (required for a real soft body)

`add_isaac_physics_to_usd.py` runs under Blender/usd-python, which has **no `PhysxSchema`**, so it
authors the deformable's PhysX parameters (`simulationHexahedralResolution`, material, etc.) as bare
`custom` attributes. Isaac Sim does **not** honour those, so the FEM body cooks to a degenerate
**~8-node bounding box** (a single hex cell) and the "deformable" fish deforms like a rigid block.

`scripts/cook_deformable_isaac.py` fixes this: run under an **Isaac Lab python** (which has
`omni.physx`), it re-authors the deformable the canonical way via
`deformableUtils.add_physx_deformable_body(..., simulation_hexahedral_resolution=N)` so PhysX cooks a
proper **~150-node** simulation mesh at load (parity with the reference salmon's 175). It also
creates+binds a real `PhysxDeformableBodyMaterialAPI` (so youngsModulus/elasticityDamping are honoured
instead of silent no-ops) and right-hands the reflected (negative-scale) deformable xform that Isaac's
`DeformablePrim` otherwise rejects.

`generate_single_fish_usd.sh` runs it automatically as step **[6/6]** when `ISAAC_PY` points at an
Isaac Lab python (else it warns and skips):

```bash
ISAAC_PY=/path/to/env_isaaclab/bin/python \
  bash scripts/generate_single_fish_usd.sh <in.blend> <out_dir> [target_len_m]
# HEX_RES=10 (default; 12-15 for a finer body -- higher raises the CFL blow-up risk)
```

Or cook an already-generated USD in place:

```bash
/path/to/env_isaaclab/bin/python scripts/cook_deformable_isaac.py \
  --in fish_articulated.usd --out fish_articulated.usd --hex-resolution 10
```

Verified on `zef_f0001`: the FEM simulation mesh went from **8 → 165 nodes** and the asset runs stably
in the analytic-water IL task.

## Important Isaac Sim Notes

`GroundPlane` is authored with collider schemas but is inactive by default:

```text
/root/GroundPlane
  type: Mesh
  active: false
  schemas: PhysicsCollisionAPI, PhysicsMeshCollisionAPI
  physics:collisionEnabled: true
```

Activate `GroundPlane` in Isaac Sim if you want the fish to collide with it during simulation.

The deformable mesh mass is intentionally separate from bone mass. Extremely small deformable mass values such as `0.00001kg` can make the solver unstable when attached to multi-kilogram rigid bones. If the fish jitters, explodes, or disappears, increase deformable mass in:

```text
configs/usd_physics_defaults.json
```

Recommended first test values:

```text
0.01kg to 0.05kg
```

## Main Scripts

```text
scripts/generate_single_fish_usd.ps1       one fish, end-to-end
scripts/generate_auto_skeleton_blend.py    fit/generate skeleton in Blender
scripts/extract_fish_skeleton.py           export fish_mesh.obj and skeleton.json
scripts/export_blend_to_usd.py             Blender USD export
scripts/add_isaac_physics_to_usd.py        add physics, joints, masses, attachments
scripts/convert_usda_to_usd.py             convert USDA to binary USD
```

## Design Rule

The final USD should always be generated by deterministic scripts. A future VLM may predict compact skeleton/template parameters, but it should not write USD directly.
