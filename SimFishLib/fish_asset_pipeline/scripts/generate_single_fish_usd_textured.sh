#!/usr/bin/env bash
# TEXTURE-PRESERVING fish pipeline for RAW Meshy blends (no bones needed), Linux/this box.
#   bash generate_single_fish_usd_textured.sh <raw.blend> <output_dir> [target_length_m]
# Produces <output_dir>/fish_articulated.usd with UsdUVTexture materials (base color /
# roughness / normal from the blend's packed images), skeleton, joints, FEM deformable.
#
# Encodes the 2026-08-28 lessons:
#  - export_blend_to_usd.py must pass export_textures=True (patched; packed images
#    otherwise never reach disk and the fish comes out flat gray like misty did)
#  - generate_auto_skeleton_blend.py and extract_fish_skeleton.py treat --output as a
#    DIRECTORY and write <output>/<basename> inside it
#  - physics/convert steps need Blender's bundled python (has pxr standalone);
#    the FEM cook needs the Isaac Lab python (has omni.physx)
set -euo pipefail
IN="${1:?usage: $0 <raw.blend> <out_dir> [len_m]}"
OUT="${2:?usage: $0 <raw.blend> <out_dir> [len_m]}"
LEN="${3:-0.5}"
ROOT="${FISH_ROOT}/SimFishLib/fish_asset_pipeline"
BLENDER=$(ls ${BLENDER_ROOT}/blender-5.0.1*/blender | head -1)
BPY=$(ls ${BLENDER_ROOT}/blender-5.0.1*/5.0/python/bin/python3* | head -1)
ISAAC_PY="${ISAAC_PY:-${FISH_PYTHON}}"
S="$ROOT/scripts"; CFG="$ROOT/configs/usd_physics_defaults.json"
IDX=$(find "$ROOT" -maxdepth 3 -name "*.json" | grep -i index | head -1)
W="$OUT/_work"; mkdir -p "$W/usd" "$OUT/_auto"

echo "[1/7] auto-skeleton fit (template-matched)..."
"$BLENDER" -b "$IN" --python "$S/generate_auto_skeleton_blend.py" -- \
  --template-index "$IDX" --target-length-m "$LEN" --output "$OUT/_auto/auto_skeleton.blend" \
  > "$W/step1_autoskel.log" 2>&1
SKB="$OUT/_auto/auto_skeleton.blend/auto_skeleton.blend"; test -f "$SKB"

echo "[2/7] scale to ${LEN}m..."
"$BLENDER" -b "$SKB" --python "$S/scale_skeleton_blend_to_real_length.py" -- \
  --target-length-m "$LEN" --output "$W/scaled.blend" > "$W/step2_scale.log" 2>&1
test -f "$W/scaled.blend"

echo "[3/7] extract skeleton.json..."
"$BLENDER" -b "$W/scaled.blend" --python "$S/extract_fish_skeleton.py" -- \
  --output "$W/skeleton.json" > "$W/step3_extract.log" 2>&1
SKJ="$W/skeleton.json/skeleton.json"; test -f "$SKJ"

echo "[4/7] export USD WITH textures..."
"$BLENDER" -b "$W/scaled.blend" --python "$S/export_blend_to_usd.py" -- \
  --output "$W/usd/base_export.usda" > "$W/step4_export.log" 2>&1
test -d "$W/usd/textures"
[ "$(grep -c UsdUVTexture "$W/usd/base_export.usda")" -ge 1 ] || { echo "NO TEXTURES EXPORTED"; exit 1; }

cp -r "$W/usd/textures" "$OUT/textures"   # texture refs are RELATIVE to the final USD

echo "[5/7] author Isaac physics (bones, joints, attachments, mass)..."
"$BPY" "$S/add_isaac_physics_to_usd.py" --input-usd "$W/usd/base_export.usda" \
  --skeleton-json "$SKJ" --config "$CFG" --output-usd "$OUT/fish_articulated.usda" \
  > "$W/step5_phys.log" 2>&1
test -f "$OUT/fish_articulated.usda"

echo "[6/7] convert to binary USD..."
"$BPY" "$S/convert_usda_to_usd.py" --input "$OUT/fish_articulated.usda" \
  --output "$OUT/fish_articulated.usd" > "$W/step6_conv.log" 2>&1

echo "[7/7] FEM cook (Isaac)..."
"$ISAAC_PY" "$S/cook_deformable_isaac.py" --in "$OUT/fish_articulated.usd" \
  --out "$OUT/fish_articulated.usd" --hex-resolution "${HEX_RES:-10}" > "$W/step7_cook.log" 2>&1

N=$(grep -c UsdUVTexture "$OUT/fish_articulated.usda" || true)
echo "DONE: $OUT/fish_articulated.usd  (UsdUVTexture prims in usda: $N; textures in $W/usd/textures)"
