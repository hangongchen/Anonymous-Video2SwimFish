"""Validate a geometric head/tail detector against GT (head must be +X end).

Heuristic: slice the fish along its length axis; the tail tapers to a thin
caudal peduncle, so the end whose outer slices have the smaller cross-section
is the tail. Put the bulkier end at +X. We test two cross-section measures:
  - height span (Y extent) per slice
  - vertex count per slice (proxy for surface area / girth)

  blender --background --python check_head_detect.py -- --gt-dir <fish_bone dir>
"""

import argparse
import os
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


def head_scores(fish, nbins=24, outer=0.2, measure="thickness"):
    """Bulk of the outer slices at each X end. Tail fin is thin in Z (thickness)
    and sparse, so thickness/girth separate head from tail even though the fin
    can be tall in Y."""
    verts = [fish.matrix_world @ v.co for v in fish.data.vertices]
    xs = [p.x for p in verts]
    xmin, xmax = min(xs), max(xs)
    span = (xmax - xmin) or 1.0
    ext = [[[], []] for _ in range(nbins)]  # per-bin: [Y list, Z list]
    cnt = [0] * nbins
    for p in verts:
        b = min(nbins - 1, int((p.x - xmin) / span * nbins))
        ext[b][0].append(p.y)
        ext[b][1].append(p.z)
        cnt[b] += 1
    def zspan(i):
        return (max(ext[i][1]) - min(ext[i][1])) if ext[i][1] else 0.0
    def yspan(i):
        return (max(ext[i][0]) - min(ext[i][0])) if ext[i][0] else 0.0
    def area(i):
        return zspan(i) * yspan(i)
    fn = {"thickness": zspan, "area": area, "count": lambda i: cnt[i]}[measure]
    k = max(1, int(nbins * outer))
    low = sum(fn(i) for i in range(k)) / k
    high = sum(fn(nbins - 1 - i) for i in range(k)) / k
    return low, high


def head_is_plus_x(fish, measure="thickness"):
    low, high = head_scores(fish, measure=measure)
    return high >= low, low, high


def measures_all(fish, nbins=32, outer=0.12):
    """Return {name: (+X_is_head_bool, detail)} for several robust cues."""
    verts = [fish.matrix_world @ v.co for v in fish.data.vertices]
    xs = [p.x for p in verts]
    xmin, xmax = min(xs), max(xs)
    span = (xmax - xmin) or 1.0
    ybin = [[] for _ in range(nbins)]
    zbin = [[] for _ in range(nbins)]
    cnt = [0] * nbins
    for p in verts:
        b = min(nbins - 1, int((p.x - xmin) / span * nbins))
        ybin[b].append(p.y)
        zbin[b].append(p.z)
        cnt[b] += 1
    def yspan(i):
        return (max(ybin[i]) - min(ybin[i])) if ybin[i] else 0.0
    def zspan(i):
        return (max(zbin[i]) - min(zbin[i])) if zbin[i] else 0.0
    k = max(1, int(nbins * outer))
    out = {}
    # center of mass along X vs bbox center: mass sits at the head (bulky) end
    com = sum(xs) / len(xs)
    center = (xmin + xmax) / 2
    out["com"] = (com > center, "com=%.3f center=%.3f" % (com, center))
    # girth (Y*Z area) of outer slices: head end bulkier
    lo_area = sum(yspan(i) * zspan(i) for i in range(k)) / k
    hi_area = sum(yspan(nbins - 1 - i) * zspan(nbins - 1 - i) for i in range(k)) / k
    out["area"] = (hi_area >= lo_area, "lo=%.4f hi=%.4f" % (lo_area, hi_area))
    # tip solidity: head tip stays wide, tail tapers to a point (use widest cross axis)
    def wide(i):
        return max(yspan(i), zspan(i))
    body = max(wide(i) for i in range(nbins)) or 1.0
    lo_tip = wide(0) / body
    hi_tip = wide(nbins - 1) / body
    out["tip"] = (hi_tip >= lo_tip, "lo_tip=%.2f hi_tip=%.2f" % (lo_tip, hi_tip))
    return out


def main():
    args = parse_args()
    correct = 0
    total = 0
    for bf in sorted(os.listdir(args.gt_dir)):
        if not bf.endswith(".blend"):
            continue
        bpy.ops.wm.open_mainfile(filepath=os.path.join(args.gt_dir, bf))
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        bones = [o for o in meshes if o.name.lower().startswith("bone")]
        others = [o for o in meshes if not o.name.lower().startswith("bone")]
        def vol(o):
            mn, mx = world_bbox(o)
            s = mx - mn
            return s[0] * s[1] * s[2]
        fish = max(others, key=vol)

        # canonicalize axes: length -> X (match render_overlay.normalize_fish)
        fmn, fmx = world_bbox(fish)
        size = fmx - fmn
        order = sorted(range(3), key=lambda i: -size[i])
        from mathutils import Matrix
        if order != [0, 1, 2]:
            m = Matrix.Identity(3)
            for dst, src_axis in enumerate(order):
                for r in range(3):
                    m[dst][r] = 1.0 if r == src_axis else 0.0
            fish.matrix_world = m.to_4x4() @ fish.matrix_world
            for b in bones:
                b.matrix_world = m.to_4x4() @ b.matrix_world
        bpy.context.view_layer.update()

        # Dataset convention (visually confirmed on carp/rudd/goldfish renders):
        # head always renders at +X after the length->X canonicalization.
        gt_head_plus_x = True

        row = [os.path.splitext(bf)[0]]
        acc = globals().setdefault("_acc", {})
        for name, (plus_head, detail) in measures_all(fish).items():
            ok = plus_head == gt_head_plus_x
            acc[name] = acc.get(name, 0) + ok
            row.append("%s:%s%s" % (name, "+X" if plus_head else "-X", "" if ok else "!"))
        total += 1
        print("HEAD %-24s %s" % (row[0], "  ".join(row[1:])))
    acc = globals().get("_acc", {})
    print("HEAD ACCURACY " + "  ".join("%s=%d/%d" % (k, v, total) for k, v in acc.items()))


if __name__ == "__main__":
    main()
