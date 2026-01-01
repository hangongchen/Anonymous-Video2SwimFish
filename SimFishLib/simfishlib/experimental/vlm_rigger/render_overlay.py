"""Stage 1: canonical overlay renderer for the VLM rigging loop.

Renders a fish mesh (translucent) with the currently placed bones (opaque,
colored, numbered) from canonical side/top orthographic views using the
Workbench engine (CPU, headless-safe).

Runs inside Blender:
  blender --background --python render_overlay.py -- --spec state.json

Spec JSON:
{
  "fish": {"blend": "/path/fish.blend", "object": null},   # object: null = largest mesh
  "templates_dir": ".../stage0/templates",
  "bones": [
    {"template_id": "tpl_00_ab12cd", "center_norm": [x,y,z], "size_norm": [x,y,z]}
  ],
  "out_dir": "/path/out",
  "views": ["side", "top"],
  "resolution": 1024
}

Outputs: out_dir/side.png, out_dir/top.png, out_dir/render_meta.json
(bone screen-space centers per view, fish bbox, normalization applied).

Coordinate convention (GT): length on X, height on Y, thickness on Z.
center_norm/size_norm are relative to the fish world bbox (0..1 per axis).
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector

BONE_COLORS = [
    (0.90, 0.10, 0.10, 1.0),  # red
    (0.10, 0.55, 0.95, 1.0),  # blue
    (0.10, 0.75, 0.20, 1.0),  # green
    (0.95, 0.65, 0.05, 1.0),  # orange
    (0.60, 0.20, 0.80, 1.0),  # purple
    (0.05, 0.75, 0.75, 1.0),  # teal
    (0.90, 0.30, 0.60, 1.0),  # pink
    (0.55, 0.45, 0.10, 1.0),  # olive
    (0.30, 0.30, 0.95, 1.0),  # indigo
    (0.70, 0.70, 0.10, 1.0),  # yellow-green
]
FISH_COLOR = (0.45, 0.45, 0.52, 1.0)


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    return p.parse_args(argv)


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(c[i] for c in corners) for i in range(3)))
    mx = Vector((max(c[i] for c in corners) for i in range(3)))
    return mn, mx


def load_fish(spec):
    """Append the fish mesh object from a source blend (or import an OBJ)."""
    src = spec["fish"]
    path = src.get("blend") or src.get("obj")
    if path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    else:
        with bpy.data.libraries.load(path) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        meshes = []
        for o in data_to.objects:
            if o is None:
                continue
            if o.type != "MESH" or o.name.lower().startswith("bone"):
                bpy.data.objects.remove(o, do_unlink=True)
                continue
            bpy.context.scene.collection.objects.link(o)
            meshes.append(o)
    if not meshes:
        raise RuntimeError("no fish mesh found in %s" % path)
    wanted = src.get("object")
    if wanted:
        fish = next(o for o in meshes if o.name == wanted)
    else:
        def vol(o):
            mn, mx = world_bbox(o)
            s = mx - mn
            return s[0] * s[1] * s[2]
        fish = max(meshes, key=vol)
    # remove extra meshes
    for o in meshes:
        if o is not fish:
            bpy.data.objects.remove(o, do_unlink=True)
    return fish


def normalize_fish(fish):
    """Rotate/position so length is on X, height on Y, thickness on Z, centered at origin."""
    mn, mx = world_bbox(fish)
    size = mx - mn
    order = sorted(range(3), key=lambda i: -size[i])  # axes by extent desc
    # If already length>height>thickness on X,Y,Z nothing to do; else rotate.
    # We only handle the common case: swap axes via rotation matrix.
    from mathutils import Matrix
    axes = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
    rot = Matrix.Identity(4)
    if order != [0, 1, 2]:
        m = Matrix.Identity(3)
        for dst, src_axis in enumerate(order):
            for r in range(3):
                m[dst][r] = 1.0 if r == src_axis else 0.0
        rot = m.to_4x4()
        fish.matrix_world = rot @ fish.matrix_world
        bpy.context.view_layer.update()
        mn, mx = world_bbox(fish)

    # Head/tail orientation: dataset convention is head at +X. The tail fin is
    # thin in Z (thickness), so the bulkier-in-Z end is the head. Validated
    # 14/14 on GT. Flip 180 about Y (keeps up axis) if head is on -X.
    if head_is_minus_x(fish):
        flip = Matrix.Rotation(math.pi, 4, "Y")
        fish.matrix_world = flip @ fish.matrix_world
        bpy.context.view_layer.update()
        mn, mx = world_bbox(fish)

    center = (mn + mx) / 2
    fish.matrix_world.translation -= center
    bpy.context.view_layer.update()
    return world_bbox(fish)


def head_is_minus_x(fish):
    """True if the head is on the -X end.

    Cue: fish mass concentrates at the head/body; the tail is light and tapers
    to a thin fin that carries almost no vertices, so the length-axis center of
    mass sits toward the head. Validated 14/14 on the GT set (head at +X);
    robust to tail fins in a way girth/thickness measures are not.
    """
    xs = [(fish.matrix_world @ v.co).x for v in fish.data.vertices]
    com = sum(xs) / len(xs)
    center = (min(xs) + max(xs)) / 2
    return com < center


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def add_bone(idx, entry, templates_dir, library, fmn, fsize):
    tpl = library[entry["template_id"]]
    obj_path = os.path.join(templates_dir, os.path.basename(tpl["obj_file"]))
    bpy.ops.wm.obj_import(filepath=obj_path, forward_axis="Y", up_axis="Z")
    bone = [o for o in bpy.context.selected_objects if o.type == "MESH"][0]
    bone.name = "vlm_bone_%d" % (idx + 1)
    # template obj is unit-normalized (max extent 1, centered) -> scale to size_norm*fish size
    unit = Vector(tpl["unit_size"])
    target = Vector(
        entry["size_norm"][i] * fsize[i] for i in range(3)
    )
    bone.scale = Vector(target[i] / (unit[i] or 1.0) for i in range(3))
    bone.location = Vector(
        fmn[i] + entry["center_norm"][i] * fsize[i] for i in range(3)
    )
    color = BONE_COLORS[idx % len(BONE_COLORS)]
    bone.color = color
    mat = make_material("bone_mat_%d" % idx, color)
    bone.data.materials.clear()
    bone.data.materials.append(mat)
    return bone


def add_label(idx, bone, height):
    """3D number label above the bone, facing the side camera."""
    bpy.ops.object.text_add()
    txt = bpy.context.object
    txt.data.body = str(idx + 1)
    txt.data.align_x = "CENTER"
    txt.data.size = height
    mn, mx = world_bbox(bone)
    txt.location = ((mn[0] + mx[0]) / 2, mx[1] + height * 0.3, (mn[2] + mx[2]) / 2)
    color = BONE_COLORS[idx % len(BONE_COLORS)]
    txt.color = color
    mat = make_material("label_mat_%d" % idx, color)
    txt.data.materials.append(mat)
    return txt


def setup_camera(view, fmn, fmx, margin=1.25):
    size = fmx - fmn
    center = (fmn + fmx) / 2
    cam_data = bpy.data.cameras.new("cam_%s" % view)
    cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("cam_%s" % view, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    dist = max(size) * 3
    if view == "side":
        # look along -Z (fish X-Y plane fills the frame)
        cam.location = center + Vector((0, 0, dist))
        cam.rotation_euler = (0, 0, 0)
        extent = max(size[0], size[1] * 2)  # leave room for labels
    elif view == "top":
        # look along -Y
        cam.location = center + Vector((0, dist, 0))
        cam.rotation_euler = (-math.pi / 2, math.pi, 0)
        extent = max(size[0], size[2] * 2)
    else:
        raise ValueError(view)
    cam_data.ortho_scale = extent * margin
    return cam


def screen_coords(cam, points):
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    out = []
    for p in points:
        co = world_to_camera_view(scene, cam, Vector(p))
        out.append([round(co.x, 4), round(1.0 - co.y, 4)])  # y down, 0..1
    return out


RAY_DIRS = [Vector(d) for d in
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]


def point_inside(fish, p_local, max_hits=64):
    """Ray-parity vote over 6 axis directions (robust to bad mesh normals)."""
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


def outside_fraction(fish, bone, max_samples=400):
    inv = fish.matrix_world.inverted()
    verts = bone.data.vertices
    stride = max(1, len(verts) // max_samples)
    sampled = [verts[i] for i in range(0, len(verts), stride)]
    n_out = sum(
        0 if point_inside(fish, inv @ (bone.matrix_world @ v.co)) else 1
        for v in sampled
    )
    return n_out / max(len(sampled), 1)


def enforce_inside(fish, bone, max_frac_out=0.02, shrink=0.94, max_iter=15):
    """Hard constraint: bone must lie inside the fish mesh (GT worst case is
    1.7% of verts / 0.012 depth). Uniformly shrink the bone about its center
    until the outside fraction is within tolerance. Returns (n_shrinks, ok)."""
    for it in range(max_iter + 1):
        if outside_fraction(fish, bone) <= max_frac_out:
            return it, True
        if it == max_iter:
            break
        bone.scale *= shrink
        bpy.context.view_layer.update()
    return max_iter, False


def setup_render(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "OBJECT"
    shading.show_xray = True
    shading.xray_alpha = 0.5
    shading.background_type = "VIEWPORT"
    shading.background_color = (1.0, 1.0, 1.0)
    scene.display.render_aa = "5"
    scene.render.film_transparent = False


def render_state(spec, state, library, fmn, fmx, res):
    """Add bones for one state, optionally save blend, render views, clean up."""
    scene = bpy.context.scene
    fsize = fmx - fmn
    created = []
    bones = []
    for i, entry in enumerate(state.get("bones", [])):
        b = add_bone(i, entry, spec["templates_dir"], library, fmn, fsize)
        bones.append(b)
        created.append(b)
    bpy.context.view_layer.update()

    os.makedirs(state["out_dir"], exist_ok=True)

    # hard constraint: bones must lie inside the fish mesh
    bones_final = []
    if state.get("enforce_inside", spec.get("enforce_inside", False)):
        fish = spec["_fish_obj"]
        for i, b in enumerate(bones):
            n_shrinks, ok = enforce_inside(fish, b)
            bmn, bmx = world_bbox(b)
            bones_final.append({
                "template_id": state["bones"][i]["template_id"],
                "center_norm": [((bmn[j] + bmx[j]) / 2 - fmn[j]) / (fsize[j] or 1.0) for j in range(3)],
                "size_norm": [(bmx[j] - bmn[j]) / (fsize[j] or 1.0) for j in range(3)],
                "shrunk": n_shrinks,
                "contained": ok,
            })
            if n_shrinks:
                print("ENFORCE bone %d shrunk x%d contained=%s" % (i + 1, n_shrinks, ok))

    # Save GT-style blend (fish mesh + bone1..boneN, nothing else) before
    # labels/cameras are added. This is the pipeline handoff artifact.
    if state.get("save_blend"):
        for i, b in enumerate(bones):
            b.name = "bone%d" % (i + 1)
        bpy.ops.wm.save_as_mainfile(filepath=state["save_blend"])
        print("BLEND SAVED %s" % state["save_blend"])

    for i, b in enumerate(bones):
        created.append(add_label(i, b, height=fsize[1] * 0.14))
    bpy.context.view_layer.update()

    meta = {"views": {}, "fish_bbox": {"min": list(fmn), "max": list(fmx)},
            "bone_count": len(bones), "bones_final": bones_final}
    for view in state.get("views", ["side", "top"]):
        cam = setup_camera(view, fmn, fmx)
        created.append(cam)
        scene.camera = cam
        aspect = 0.5 if view == "top" else 0.6
        scene.render.resolution_x = res
        scene.render.resolution_y = int(res * aspect)
        out_png = os.path.join(state["out_dir"], "%s.png" % view)
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        centers = [(world_bbox(b)[0] + world_bbox(b)[1]) / 2 for b in bones]
        meta["views"][view] = {
            "png": out_png,
            "bone_screen_centers": screen_coords(cam, centers),
        }
        print("RENDER OK %s -> %s" % (view, out_png))

    with open(os.path.join(state["out_dir"], "render_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for o in created:
        bpy.data.objects.remove(o, do_unlink=True)


def main():
    args = parse_args()
    with open(args.spec) as f:
        spec = json.load(f)

    lib_path = os.path.join(os.path.dirname(spec["templates_dir"].rstrip("/")), "bone_template_library.json")
    with open(lib_path) as f:
        library = {t["template_id"]: t for t in json.load(f)["templates"]}

    bpy.ops.wm.read_factory_settings(use_empty=True)

    fish = load_fish(spec)
    fmn, fmx = normalize_fish(fish)
    spec["_fish_obj"] = fish
    fish.color = FISH_COLOR
    fish.data.materials.clear()
    fish.data.materials.append(make_material("fish_mat", FISH_COLOR))

    setup_render(bpy.context.scene)
    res = int(spec.get("resolution", 1024))

    # batch mode: many bone-states over one fish; single mode: spec is the state
    states = spec.get("states") or [{
        "bones": spec.get("bones", []),
        "out_dir": spec["out_dir"],
        "views": spec.get("views", ["side", "top"]),
        "save_blend": spec.get("save_blend"),
    }]
    for si, state in enumerate(states):
        render_state(spec, state, library, fmn, fmx, res)
        print("STATE DONE %d/%d" % (si + 1, len(states)))
    print("DONE render")


if __name__ == "__main__":
    main()
