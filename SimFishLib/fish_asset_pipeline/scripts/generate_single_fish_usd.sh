#!/usr/bin/env bash
# Linux port of generate_single_fish_usd.ps1.
# Takes a GT-style skeleton blend (fish mesh + bone meshes, bones already placed)
# and produces an Isaac Sim articulated USD. Skips step 1 (auto-skeleton fitting)
# because the bones are already present; instead it scales the asset to real
# length, then runs extract -> export -> physics -> convert.
set -euo pipefail

INPUT_BLEND="${1:?usage: generate_single_fish_usd.sh <input.blend> <output_dir> [target_length_m]}"
OUTPUT_DIR="${2:?usage: generate_single_fish_usd.sh <input.blend> <output_dir> [target_length_m]}"
TARGET_LEN="${3:-0.5}"

ROOT="/work/yzha1/SimFishLib/fish_asset_pipeline"
BLENDER="/work/yzha1/tools/blender/blender"
PY="/work/yzha1/tools/blender/5.0/python/bin/python3.11"
CFG="$ROOT/configs/usd_physics_defaults.json"
SCRIPTS="$ROOT/scripts"

# Optional FEM cook step: the Blender/usd-python 'PY' has no PhysxSchema, so the deformable is
# authored with bare custom attrs and Isaac Sim cooks it to a degenerate ~8-node box. Point
# ISAAC_PY at an Isaac Lab python (which has omni.physx) to re-author a proper ~150-node sim mesh.
ISAAC_PY="${ISAAC_PY:-}"
HEX_RES="${HEX_RES:-10}"

WORK="$OUTPUT_DIR/_work"
SKEL="$WORK/auto_skeleton"
USDW="$WORK/usd"
mkdir -p "$SKEL" "$USDW"

SCALED="$SKEL/auto_skeleton.blend"
SKEL_JSON="$SKEL/skeleton.json"
BASE_USD="$USDW/base_export.usda"
FINAL_USDA="$OUTPUT_DIR/fish_articulated.usda"
FINAL_USD="$OUTPUT_DIR/fish_articulated.usd"

echo "[1/5] Scale asset to real length ${TARGET_LEN}m (bones already placed; auto-fit skipped)..."
"$BLENDER" -b "$INPUT_BLEND" --python "$SCRIPTS/scale_skeleton_blend_to_real_length.py" -- \
  --target-length-m "$TARGET_LEN" --output "$SCALED" 2>&1 | grep -E "SCALED|Error" || true
test -f "$SCALED"

echo "[2/5] Extract skeleton.json..."
"$BLENDER" -b "$SCALED" --python "$SCRIPTS/extract_fish_skeleton.py" -- --output "$SKEL" >/dev/null 2>&1
test -f "$SKEL_JSON"

echo "[3/5] Export base USDA..."
"$BLENDER" -b "$SCALED" --python "$SCRIPTS/export_blend_to_usd.py" -- --output "$BASE_USD" >/dev/null 2>&1
test -f "$BASE_USD"

echo "[4/5] Add Isaac Sim physics, joints, attachments, mass..."
"$PY" "$SCRIPTS/add_isaac_physics_to_usd.py" \
  --input-usd "$BASE_USD" --skeleton-json "$SKEL_JSON" --config "$CFG" --output-usd "$FINAL_USDA"
test -f "$FINAL_USDA"

echo "[5/6] Convert USDA to binary USD..."
"$PY" "$SCRIPTS/convert_usda_to_usd.py" --input "$FINAL_USDA" --output "$FINAL_USD"
test -f "$FINAL_USD"

echo "[6/6] Cook FEM deformable to a proper simulation mesh (Isaac Sim)..."
if [ -n "$ISAAC_PY" ] && [ -x "$ISAAC_PY" ]; then
  "$ISAAC_PY" "$SCRIPTS/cook_deformable_isaac.py" \
    --in "$FINAL_USD" --out "$FINAL_USD" --hex-resolution "$HEX_RES"
  echo "  cooked in place -> $FINAL_USD"
else
  echo "  SKIPPED: ISAAC_PY not set. The deformable will cook to a DEGENERATE ~8-node box in"
  echo "  Isaac Sim. To make it sim-ready (~150 nodes), re-run with e.g.:"
  echo "    ISAAC_PY=/path/to/env_isaaclab/bin/python bash $0 <in.blend> <out_dir> [len]"
  echo "  or run directly: ISAAC_PY ... cook_deformable_isaac.py --in $FINAL_USD --out $FINAL_USD"
fi

echo "Generated readable USDA: $FINAL_USDA"
echo "Generated binary  USD : $FINAL_USD"
