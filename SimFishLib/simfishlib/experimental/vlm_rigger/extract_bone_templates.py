"""Stage 0: extract reusable bone template meshes + GT skeleton layouts.

Runs inside Blender:
  blender --background --python extract_bone_templates.py -- \
      --gt-dir fish_asset_pipeline/dataset/dataset/supervised_GT_dataset/fish_bone \
      --out-dir simfishlib_data/vlm_rigger/stage0

Outputs:
  out_dir/templates/tpl_<id>.obj          canonical unit-normalized bone meshes
  out_dir/bone_template_library.json      template metadata (hash, verts, source)
  out_dir/gt_layouts.json                 per-fish bone layout in fish-bbox-normalized coords
"""

import argparse
import hashlib
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def reset_blend():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c[i] for c in corners) for i in range(3)))
    mx = Vector((max(c[i] for c in corners) for i in range(3)))
    return mn, mx


def mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def is_bone(obj):
    return obj.name.lower().startswith("bone")


def canonical_geometry(obj):
    """Vertex coords in local space, centered on bbox center, scaled to unit max-extent.

    Returns (hash, verts_canonical, local_size).
    """
    verts = [v.co.copy() for v in obj.data.vertices]
    mn = Vector((min(v[i] for v in verts) for i in range(3)))
    mx = Vector((max(v[i] for v in verts) for i in range(3)))
    center = (mn + mx) / 2
    size = mx - mn
    scale = max(size) or 1.0
    canon = [tuple(round((v[i] - center[i]) / scale, 4) for i in range(3)) for v in verts]
    # order-independent-ish hash: sort vertex tuples
    h = hashlib.sha1()
    h.update(str(len(obj.data.polygons)).encode())
    for t in sorted(canon):
        h.update(("%.4f,%.4f,%.4f;" % t).encode())
    return h.hexdigest()[:12], canon, [size[0] / scale, size[1] / scale, size[2] / scale]


def export_template_obj(obj, path):
    """Export the object's canonical-normalized mesh as a simple OBJ (no bpy exporter deps)."""
    verts = [v.co.copy() for v in obj.data.vertices]
    mn = Vector((min(v[i] for v in verts) for i in range(3)))
    mx = Vector((max(v[i] for v in verts) for i in range(3)))
    center = (mn + mx) / 2
    scale = max(mx - mn) or 1.0
    with open(path, "w") as f:
        f.write("# bone template extracted from GT blend\n")
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % tuple((v[i] - center[i]) / scale for i in range(3)))
        for poly in obj.data.polygons:
            f.write("f " + " ".join(str(i + 1) for i in poly.vertices) + "\n")


def main():
    args = parse_args()
    tpl_dir = os.path.join(args.out_dir, "templates")
    os.makedirs(tpl_dir, exist_ok=True)

    templates = {}  # hash -> meta
    layouts = {}

    blend_files = sorted(
        f for f in os.listdir(args.gt_dir) if f.endswith(".blend")
    )
    for bf in blend_files:
        sample = os.path.splitext(bf)[0]
        path = os.path.join(args.gt_dir, bf)
        reset_blend()
        bpy.ops.wm.open_mainfile(filepath=path)

        meshes = mesh_objects()
        bones = [o for o in meshes if is_bone(o)]
        non_bones = [o for o in meshes if not is_bone(o)]
        if not bones or not non_bones:
            print("WARN: %s bones=%d non_bones=%d, skipping" % (sample, len(bones), len(non_bones)))
            continue
        # fish mesh = largest world bbox volume among non-bone meshes
        def bbox_vol(o):
            mn, mx = world_bbox(o)
            s = mx - mn
            return s[0] * s[1] * s[2]
        fish = max(non_bones, key=bbox_vol)
        fmn, fmx = world_bbox(fish)
        fsize = fmx - fmn

        bone_entries = []
        for b in bones:
            ghash, _, unit_size = canonical_geometry(b)
            if ghash not in templates:
                tpl_id = "tpl_%02d_%s" % (len(templates), ghash[:6])
                obj_path = os.path.join(tpl_dir, tpl_id + ".obj")
                export_template_obj(b, obj_path)
                templates[ghash] = {
                    "template_id": tpl_id,
                    "geometry_hash": ghash,
                    "vertex_count": len(b.data.vertices),
                    "face_count": len(b.data.polygons),
                    "unit_size": unit_size,
                    "first_seen": {"sample": sample, "object": b.name},
                    "obj_file": os.path.relpath(obj_path, args.out_dir),
                    "usage_count": 0,
                }
            templates[ghash]["usage_count"] += 1

            bmn, bmx = world_bbox(b)
            bcenter = (bmn + bmx) / 2
            bsize = bmx - bmn
            bone_entries.append({
                "object_name": b.name,
                "template_id": templates[ghash]["template_id"],
                "center_world": list(bcenter),
                "size_world": list(bsize),
                "center_norm": [
                    (bcenter[i] - fmn[i]) / (fsize[i] or 1.0) for i in range(3)
                ],
                "size_norm": [bsize[i] / (fsize[i] or 1.0) for i in range(3)],
            })

        # sort head->tail along length axis (longest fish axis)
        length_axis = max(range(3), key=lambda i: fsize[i])
        bone_entries.sort(key=lambda e: e["center_norm"][length_axis])

        layouts[sample] = {
            "blend_file": bf,
            "fish_mesh_object": fish.name,
            "fish_bbox_world": {"min": list(fmn), "max": list(fmx), "size": list(fsize)},
            "length_axis": length_axis,
            "bone_count": len(bone_entries),
            "bones": bone_entries,
        }
        print("OK %s: fish=%s bones=%d" % (sample, fish.name, len(bone_entries)))

    with open(os.path.join(args.out_dir, "bone_template_library.json"), "w") as f:
        json.dump({
            "schema_version": "0.1.0",
            "templates": sorted(templates.values(), key=lambda t: t["template_id"]),
        }, f, indent=2)
    with open(os.path.join(args.out_dir, "gt_layouts.json"), "w") as f:
        json.dump({"schema_version": "0.1.0", "samples": layouts}, f, indent=2)

    print("DONE: %d unique bone templates from %d samples" % (len(templates), len(layouts)))


if __name__ == "__main__":
    main()
