import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a skeleton template index from extracted supervised skeleton.json files.")
    parser.add_argument("--project-root", default="E:/Fish_dataset/fish_asset_pipeline")
    parser.add_argument("--supervised-blend-dir", default="E:/Fish_dataset/fish_asset_pipeline/dataset/dataset/supervised_GT_dataset/fish_bone")
    parser.add_argument("--extracted-supervised-dir", default="E:/Fish_dataset/fish_asset_pipeline/extracted_dataset/supervised")
    parser.add_argument("--output", default="E:/Fish_dataset/fish_asset_pipeline/configs/skeleton_template_index.json")
    return parser.parse_args()


def normalized_name(name):
    base = Path(name).stem.lower()
    for suffix in ("_obj_1", "_obj", "_fish"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base.endswith(" fish"):
        base = base[:-5]
    return "".join(ch for ch in base if ch.isalnum())


def bbox_features(size):
    length = max(size)
    thickness = min(size)
    middle = sum(size) - length - thickness
    return {
        "size": size,
        "length_axis": size.index(length),
        "height_axis": size.index(middle),
        "thickness_axis": size.index(thickness),
        "length": length,
        "height": middle,
        "thickness": thickness,
        "length_height_ratio": length / middle if middle else 0.0,
        "length_thickness_ratio": length / thickness if thickness else 0.0,
    }


def desired_bone_count(length_height_ratio):
    if length_height_ratio >= 3.2:
        return 8
    if length_height_ratio >= 2.6:
        return 7
    if length_height_ratio >= 2.15:
        return 6
    return 5


def main():
    args = parse_args()
    supervised_blend_dir = Path(args.supervised_blend_dir)
    extracted_dir = Path(args.extracted_supervised_dir)
    templates = []

    for json_path in sorted(extracted_dir.glob("*/skeleton.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fish_mesh = data.get("fish_mesh") or {}
        bbox = fish_mesh.get("bbox_world") or {}
        size = bbox.get("size")
        if not size:
            continue
        sample_name = json_path.parent.name
        blend_path = supervised_blend_dir / f"{sample_name}.blend"
        if not blend_path.exists():
            candidates = list(supervised_blend_dir.glob(f"{sample_name}*.blend"))
            blend_path = candidates[0] if candidates else blend_path
        bone_count = len(data.get("skeleton_objects", [])) + sum(item.get("bone_count", 0) for item in data.get("armatures", []))
        features = bbox_features(size)
        templates.append(
            {
                "sample_name": sample_name,
                "key": normalized_name(sample_name),
                "template_blend": str(blend_path),
                "template_skeleton_json": str(json_path),
                "fish_mesh_name": fish_mesh.get("object_name"),
                "bone_count": bone_count,
                "features": features,
            }
        )

    index = {
        "schema_version": "0.1.0",
        "selection_policy": {
            "name_match_priority": True,
            "shape_fallback": "desired_bone_count_then_length_height_ratio_nearest",
            "desired_bone_count_rules": [
                {"if_length_height_ratio_gte": 3.2, "bone_count": "7-8"},
                {"if_length_height_ratio_gte": 2.6, "bone_count": "6-7"},
                {"if_length_height_ratio_gte": 2.15, "bone_count": "5-6"},
                {"default": "5"},
            ],
        },
        "templates": templates,
    }
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
