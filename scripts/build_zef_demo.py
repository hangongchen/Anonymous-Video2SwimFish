#!/usr/bin/env python
"""Build a DEMO npz for the Salmon-IL-Water task from a 3D-ZeF annotation trajectory.

The IL "demo" reward matches the sim fish's FEM point cloud, in the env-LOCAL frame, to a
recorded reference cloud `pc (T,N,3)` following a root path `root (T,13)`. Here the reference
motion is a REAL zebrafish swimming trajectory (3D head position per frame, from 3D-ZeF
gt.txt), and the reference *shape* is the SimFishLib zebrafish mesh (`fish_articulated.usd`'s
deformable `final_mesh`) rigidly carried along that path -- i.e. the same fish we swap in as
the agent, so a perfect imitation could drive chamfer -> 0.

Only pc + root are written (no joint/nodal full state), so RSI auto-disables in the env and
every episode replays the clip from frame 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



def load_zef_track(gt_path: Path, fish_id: int) -> np.ndarray:
    g = np.loadtxt(gt_path, delimiter=",")
    g = g[g[:, 1] == fish_id]
    g = g[np.argsort(g[:, 0])]
    return g[:, 2:5].astype(np.float64)  # (F,3) 3D head position, world, centimetres


def mesh_points_from_usd(usd_path: Path, mesh_token: str) -> np.ndarray:
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    pts = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh" and mesh_token in prim.GetName():
            if "PhysxDeformableBodyAPI" in prim.GetAppliedSchemas():
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                break
    if pts is None:  # fall back to any mesh with the token
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh" and mesh_token in prim.GetName():
                pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                break
    if pts is None:
        raise RuntimeError(f"no mesh matching '{mesh_token}' in {usd_path}")
    return pts


def rot_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation mapping unit vector a onto unit vector b (shortest arc)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-8:  # opposite -> 180 deg about any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis) + 1e-12
        x, y, z = axis
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    s = np.linalg.norm(v) + 1e-12
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / (np.linalg.norm(q) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=_P("${ZEF_ROOT}/data/ZebraFish-01/gt/gt.txt"))
    ap.add_argument("--fish-id", type=int, default=1)
    ap.add_argument("--points-npy", default=_P("${FISH_ROOT}/demo_out/zef_demo/zef_mesh_points.npy"),
                    help="pre-extracted deformable mesh points (normalized asset frame)")
    ap.add_argument("--body-length-m", type=float, default=0.5,
                    help="scale the (normalized) base cloud so its long axis spans this many metres")
    ap.add_argument("--out", default=_P("${FISH_ROOT}/demo_out/zef_demo/zef01_id1_demo.npz"))
    ap.add_argument("--n-points", type=int, default=600)
    ap.add_argument("--frame-stride", type=int, default=2)    # 60fps ZeF -> ~30fps control
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--start-frame", type=int, default=1000)  # skip the settle-in at clip start
    ap.add_argument("--spawn-height", type=float, default=1.0)  # env-local Z of the fish spawn
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    args = ap.parse_args()

    track_cm = load_zef_track(Path(args.gt), args.fish_id)              # (F,3) cm
    track = track_cm * 0.01                                              # -> metres
    track = track[args.start_frame::args.frame_stride][:args.max_frames]
    T = len(track)
    # env-LOCAL frame: recenter so frame 0 sits at the sim fish's spawn (0,0,spawn_height)
    track = track - track[0] + np.array([0.0, 0.0, args.spawn_height])

    # base shape cloud: the zebrafish mesh, centred, scaled to a real body length, subsampled
    pts = np.load(args.points_npy).astype(np.float64)                   # (M,3) normalized asset frame
    pts = pts - pts.mean(0)
    long_extent = float((pts.max(0) - pts.min(0)).max())               # normalized long-axis span
    pts *= args.body_length_m / max(long_extent, 1e-9)                 # -> metres
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(args.n_points, len(pts)), replace=False)
    base = pts[idx]                                                     # (N,3)
    N = len(base)
    # body long axis of the asset = its largest-extent principal axis -> we point THAT along velocity
    u, s, vt = np.linalg.svd(base - base.mean(0), full_matrices=False)
    long_axis = vt[0]

    # per-frame heading from the (smoothed) velocity direction; hold last valid when ~stationary
    vel = np.gradient(track, axis=0) / args.dt                          # (T,3) m/s
    pc = np.empty((T, N, 3), np.float32)
    root = np.empty((T, 13), np.float32)
    R_prev = np.eye(3)
    for t in range(T):
        v = vel[t]
        R = rot_from_a_to_b(long_axis, v) if np.linalg.norm(v) > 1e-4 else R_prev
        R_prev = R
        pc[t] = (base @ R.T + track[t]).astype(np.float32)
        root[t, 0:3] = track[t]
        root[t, 3:7] = mat_to_quat_wxyz(R)
        root[t, 7:10] = vel[t]
        root[t, 10:13] = 0.0

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pc=pc, root=root.astype(np.float32))
    span = track.max(0) - track.min(0)
    print(f"[build_zef_demo] wrote {out}")
    print(f"  frames T={T}  points N={N}  (from {len(pts)} mesh verts)")
    print(f"  path span (m): {span.round(3)}  speed mean/max (m/s): "
          f"{np.linalg.norm(vel,axis=1).mean():.3f}/{np.linalg.norm(vel,axis=1).max():.3f}")
    print(f"  pc bbox (m): min {pc.reshape(-1,3).min(0).round(3)} max {pc.reshape(-1,3).max(0).round(3)}")


if __name__ == "__main__":
    main()
