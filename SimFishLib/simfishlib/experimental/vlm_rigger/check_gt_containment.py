"""Measure GT bone containment inside the fish mesh (calibration for the
inside-the-mesh hard constraint).

  blender --background --python check_gt_containment.py -- --gt-dir <fish_bone dir>

For every GT bone vertex: closest point on fish mesh, outside if the surface
normal points along (vertex - surface_point). Reports per-bone fraction of
outside vertices and max penetration depth normalized by fish height.
"""

import argparse
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    return p.parse_args(argv)


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c[i] for c in corners) for i in range(3)))
    mx = Vector((max(c[i] for c in corners) for i in range(3)))
    return mn, mx


RAY_DIRS = [Vector(d) for d in
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]


def point_inside(fish, p_local, max_hits=64):
    """Ray-parity vote over 6 axis directions (robust to bad normals)."""
    votes = 0
    for d in RAY_DIRS:
        hits = 0
        origin = p_local.copy()
        for _ in range(max_hits):
            ok, loc, _, _ = fish.ray_cast(origin, d)
            if not ok:
                break
            hits += 1
            origin = loc + d * 1e-5
        if hits % 2 == 1:
            votes += 1
    return votes >= 3


def containment(fish, bone, depsgraph):
    inv = fish.matrix_world.inverted()
    fmn, fmx = world_bbox(fish)
    norm = (fmx - fmn).y or 1.0
    n_out, max_depth = 0, 0.0
    verts = bone.data.vertices
    for v in verts:
        p_local = inv @ (bone.matrix_world @ v.co)
        if not point_inside(fish, p_local):
            n_out += 1
            ok, loc, _, _ = fish.closest_point_on_mesh(p_local, depsgraph=depsgraph)
            if ok:
                depth = (fish.matrix_world.to_scale()[1] * (p_local - loc).length) / norm
                max_depth = max(max_depth, depth)
    return n_out / max(len(verts), 1), max_depth


def main():
    import os
    args = parse_args()
    for bf in sorted(os.listdir(args.gt_dir)):
        if not bf.endswith(".blend"):
            continue
        bpy.ops.wm.open_mainfile(filepath=os.path.join(args.gt_dir, bf))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        bones = [o for o in meshes if o.name.lower().startswith("bone")]
        others = [o for o in meshes if not o.name.lower().startswith("bone")]
        def vol(o):
            mn, mx = world_bbox(o)
            s = mx - mn
            return s[0] * s[1] * s[2]
        fish = max(others, key=vol)
        worst = (0.0, 0.0, "")
        for b in bones:
            fo, md = containment(fish, b, depsgraph)
            if fo > worst[0]:
                worst = (fo, md, b.name)
        print("GTCHECK %-24s worst_bone=%s frac_out=%.3f max_depth=%.4f" %
              (os.path.splitext(bf)[0], worst[2], worst[0], worst[1]))


if __name__ == "__main__":
    main()
