#!/usr/bin/env python
"""Standalone Isaac Lab experiment: does the cooked deformable (FEM) zebrafish BLOW UP when a
manipulator grips/squeezes it, using the CURRENT training config (dt=1/960, youngsModulus 1e5,
elasticityDamping 0.05)?

Design
------
* Gravity ON + a ground plane -- the fish just RESTS on the ground (no swim policy; the D6 joints
  hold a neutral posture at the authored drive, so the fish keeps its natural straight shape while
  being squeezed -- this is "a captured live fish held still", not a controlled one).
* The FEM body is set up via the EXACT same `_prepare_env_assets(...)` path the RL env uses
  (same deformable material create+bind, same reflected/root fixups, same dissipation attrs), so
  this measures the real training config, not a different one.
* A parallel two-jaw KINEMATIC gripper measures the settled body, opens around its mid-section, then
  PROGRESSIVELY squeezes laterally (y) from just-touching down to ~85% compression, then holds.
* Every physics step (960 Hz) we log max FEM nodal velocity, max joint velocity and NaN flags.
  BLOW-UP := nodal_vel > 50 m/s OR joint_vel > 200 rad/s (the env's own limit) OR any NaN.
  We report the squeeze % at first blow-up (or "survived").

Run (own GPU so it doesn't disturb training on cuda:0):
  <env_isaaclab>/bin/python scripts/gripper_squeeze_fem.py --device cuda:1 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--usd", default="SimFishLib/fish_asset_pipeline/generated_usd_dataset/"
                "vlm_rigged_single_fish/zef_f0001/fish_articulated_cooked.usd")
ap.add_argument("--out-dir", default="demo_out/gripper_squeeze")
ap.add_argument("--squeeze-final", type=float, default=0.15,
                help="final inner half-gap as a fraction of the body half-width (0.15 -> 85% squeeze)")
ap.add_argument("--t-settle", type=float, default=1.5)
ap.add_argument("--t-approach", type=float, default=0.5)
ap.add_argument("--t-squeeze", type=float, default=4.0)
ap.add_argument("--t-hold", type=float, default=1.5)
ap.add_argument("--drop-z", type=float, default=0.15)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True if not hasattr(args, "headless") else args.headless

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# render the viewport when running with a GUI (so you can actually watch the squeeze)
RENDER = not bool(getattr(args, "headless", False))

# ---------------------------------------------------------------- imports (post-boot) ------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

# reuse the EXACT asset-prep the RL env uses (module-level fn -> importable standalone)
sys.path.insert(0, os.path.abspath("source/FISH"))
from FISH.tasks.direct.fish.salmon_swim_env import _prepare_env_assets  # noqa: E402

DEV = args.device
DT = 1.0 / 960
FISH_PATH = "/World/fish"
DEFORM_TOKEN = "final_mesh"

# blow-up thresholds (env's own joint limit + a generous FEM nodal-velocity ceiling)
JVEL_LIMIT = 200.0
NVEL_LIMIT = 50.0


def make_jaw(stage, path, dims, pos, color):
    """A kinematic rigid box collider (unit Cube scaled to `dims`), used as one gripper jaw."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)                       # unit cube spans [-0.5, 0.5]
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in dims]))   # box dims = dims
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return prim


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    sim = SimulationContext(SimulationCfg(
        dt=DT, device=DEV, gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,
            gpu_collision_stack_size=2 ** 30, gpu_heap_capacity=2 ** 30,
            gpu_temp_buffer_capacity=2 ** 28, gpu_max_soft_body_contacts=2 ** 22,
            gpu_max_particle_contacts=2 ** 28,
        ),
    ))
    stage = sim.stage

    # ground plane
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # fish (reference the cooked USD; drop from a small height so it settles on its belly)
    fish = stage.DefinePrim(FISH_PATH, "Xform")
    fish.GetReferences().AddReference(os.path.abspath(args.usd))
    UsdGeom.Xformable(fish).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.drop_z))

    # IDENTICAL FEM setup to training (material create+bind, root/reflect fixups, dissipation attrs)
    n_soft, n_drive, n_coll = _prepare_env_assets(
        stage, FISH_PATH, control_mode="position", keep_deformable=True,
        deformable_token=DEFORM_TOKEN, fix_multi_articulation_root=True,
        fix_reflected_deformable=True, keep_bone_colliders=True,
        fem_youngs_modulus=1.0e5, fem_elasticity_damping=0.05, fem_damping_scale=1.0,
    )
    print(f"[squeeze] _prepare_env_assets: soft={n_soft} drive={n_drive} coll={n_coll}", flush=True)

    # two jaws, created OPEN and far apart (repositioned after we measure the settled fish)
    JAW_DIMS = (0.20, 0.02, 0.30)          # (x span, y thickness, z tall) -- broad clamp plates
    make_jaw(stage, "/World/jaw_a", JAW_DIMS, (0.0, +0.30, args.drop_z), (0.9, 0.2, 0.2))
    make_jaw(stage, "/World/jaw_b", JAW_DIMS, (0.0, -0.30, args.drop_z), (0.2, 0.4, 0.9))

    sim.reset()

    # ---- views ------------------------------------------------------------------------------
    from isaacsim.core.prims import DeformablePrim, RigidPrim
    # deformable prim path (final_mesh with simulationPoints)
    soft_path = None
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if DEFORM_TOKEN in p and prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
            soft_path = p
            break
    assert soft_path is not None, "deformable prim not found"
    soft = DeformablePrim(prim_paths_expr=soft_path, reset_xform_properties=False)
    soft.initialize()
    N = soft.get_simulation_mesh_nodal_positions().shape[1]
    print(f"[squeeze] deformable: {soft_path}  ({N} sim nodes)", flush=True)

    jaws = RigidPrim(prim_paths_expr="/World/jaw_*", reset_xform_properties=False)
    jaws.initialize()
    jaw_paths = list(jaws.prim_paths) if hasattr(jaws, "prim_paths") else ["a", "b"]
    print(f"[squeeze] jaws: {jaw_paths}", flush=True)

    # optional joint-velocity view (best-effort)
    art = None
    try:
        from isaacsim.core.prims import Articulation
        art = Articulation(prim_paths_expr=FISH_PATH, reset_xform_properties=False)
        art.initialize()
        _ = art.get_joint_velocities()
        print(f"[squeeze] articulation joint-vel view OK ({art.num_dof} dof)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[squeeze] joint-vel view unavailable ({e}); monitoring nodal vel only", flush=True)

    def nodal():
        return soft.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy()

    def nodal_vmax():
        v = soft.get_simulation_mesh_nodal_velocities()[0]
        return float(torch.nan_to_num(v).abs().max()), bool(torch.isnan(v).any())

    def jvel_max():
        if art is None:
            return 0.0, False
        v = art.get_joint_velocities()[0]
        return float(torch.nan_to_num(v).abs().max()), bool(torch.isnan(v).any())

    def set_jaws(gap_half, cx, cz):
        """place jaw inner faces at cx +/- gap_half about center (cx=fish center x already baked in pos)."""
        yb = gap_half + JAW_DIMS[1] / 2.0            # jaw CENTER y = inner-face + half-thickness
        pos = torch.tensor([[cx, +yb, cz], [cx, -yb, cz]], device=DEV, dtype=torch.float32)
        quat = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], device=DEV, dtype=torch.float32)
        jaws.set_world_poses(positions=pos, orientations=quat)

    # ---- phases (in physics steps) ----------------------------------------------------------
    n_settle = int(args.t_settle / DT)
    n_appr = int(args.t_approach / DT)
    n_sq = int(args.t_squeeze / DT)
    n_hold = int(args.t_hold / DT)

    log = []          # per-step: t, phase, gap_half, squeeze_frac, nvel, jvel, nan
    frames = []       # for video: (t, nodal_pts(N,3), jaw_centers(2,3))
    rec_every = 16    # 960/16 = 60 fps
    blowup_at = None  # (t, squeeze_frac, reason)

    # keep jaws parked wide-open during settle
    set_jaws(0.30, 0.0, args.drop_z)
    print(f"[squeeze] phases: settle {n_settle}, approach {n_appr}, squeeze {n_sq}, hold {n_hold} "
          f"(total {n_settle+n_appr+n_sq+n_hold} steps @ {1/DT:.0f} Hz)", flush=True)

    # --- settle ---
    for i in range(n_settle):
        sim.step(render=RENDER)

    # measure settled fish
    pts = nodal()
    lo, hi = pts.min(0), pts.max(0)
    cen = (lo + hi) / 2.0
    hw_y = float((hi[1] - lo[1]) / 2.0)               # body half-width (y)
    cx, cz = float(cen[0]), float(cen[2])
    print(f"[squeeze] settled bbox size(LxWxH)={(hi-lo).round(4)} center={cen.round(4)} "
          f"half_width_y={hw_y:.4f} m", flush=True)

    g_open = hw_y + 0.03                              # inner face 3 cm outside the body
    g_final = hw_y * float(args.squeeze_final)
    set_jaws(g_open, cx, cz)                          # snap jaws to fish mid-section (still open)

    def record(t, gap_half, phase):
        nv, nn = nodal_vmax()
        jv, jn = jvel_max()
        sq = max(0.0, (hw_y - gap_half) / hw_y)       # fraction of half-width compressed
        log.append((t, phase, gap_half, sq, nv, jv, int(nn or jn)))
        if len(log) % rec_every == 0:
            yb = gap_half + JAW_DIMS[1] / 2.0
            frames.append((t, nodal().copy(), np.array([[cx, +yb, cz], [cx, -yb, cz]])))
        blow = (nv > NVEL_LIMIT) or (jv > JVEL_LIMIT) or nn or jn
        return blow, nv, jv, sq

    t = 0.0
    # --- approach (move from parked-open to g_open; already at g_open, so this just holds+contacts) ---
    for i in range(n_appr):
        sim.step(render=RENDER)
        t += DT
        blow, nv, jv, sq = record(t, g_open, "approach")
        if blow and blowup_at is None:
            blowup_at = (t, sq, f"nvel={nv:.1f} jvel={jv:.1f}")
            break

    # --- squeeze ramp g_open -> g_final ---
    if blowup_at is None:
        for i in range(n_sq):
            frac = (i + 1) / n_sq
            gap = g_open + (g_final - g_open) * frac
            set_jaws(gap, cx, cz)
            sim.step(render=RENDER)
            t += DT
            blow, nv, jv, sq = record(t, gap, "squeeze")
            if i % 480 == 0:
                print(f"  [squeeze] t={t:5.2f}s gap={gap*1000:5.1f}mm squeeze={sq*100:4.0f}% "
                      f"nvel={nv:7.2f} jvel={jv:7.2f}", flush=True)
            if blow and blowup_at is None:
                blowup_at = (t, sq, f"nvel={nv:.1f} jvel={jv:.1f}")
                break

    # --- hold at max squeeze ---
    if blowup_at is None:
        for i in range(n_hold):
            set_jaws(g_final, cx, cz)
            sim.step(render=RENDER)
            t += DT
            blow, nv, jv, sq = record(t, g_final, "hold")
            if blow and blowup_at is None:
                blowup_at = (t, sq, f"nvel={nv:.1f} jvel={jv:.1f}")
                break

    # ---- report + save ----------------------------------------------------------------------
    L = np.array([(r[0], r[2], r[3], r[4], r[5], r[6]) for r in log], dtype=np.float64)  # t,gap,sq,nv,jv,nan
    out_npz = os.path.join(args.out_dir, "squeeze_log.npz")
    np.savez_compressed(
        out_npz,
        t=L[:, 0], gap_half=L[:, 1], squeeze_frac=L[:, 2], nodal_vmax=L[:, 3],
        joint_vmax=L[:, 4], nan=L[:, 5],
        frames_t=np.array([f[0] for f in frames]),
        frames_pts=np.array([f[1] for f in frames]),
        frames_jaw=np.array([f[2] for f in frames]),
        jaw_dims=np.array(JAW_DIMS), half_width_y=hw_y, center=np.array([cx, cen[1], cz]),
        n_nodes=N, dt=DT,
    )
    print("\n================= RESULT =================", flush=True)
    print(f"body half-width (y)     : {hw_y*1000:.1f} mm", flush=True)
    print(f"peak nodal velocity     : {L[:,3].max():.2f} m/s", flush=True)
    print(f"peak joint velocity     : {L[:,4].max():.2f} rad/s", flush=True)
    print(f"max squeeze reached      : {L[:,2].max()*100:.0f} %  (gap {L[:,1].min()*1000:.1f} mm)", flush=True)
    if blowup_at is not None:
        print(f">>> BLEW UP at t={blowup_at[0]:.2f}s, squeeze={blowup_at[1]*100:.0f}%  ({blowup_at[2]})", flush=True)
    else:
        print(">>> NO BLOW-UP: survived full squeeze + hold, FEM stayed finite.", flush=True)
    print(f"log -> {out_npz}  ({len(frames)} video frames captured)", flush=True)
    print("==========================================", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
    os._exit(0)   # SimulationContext/Kit teardown hangs on this box -> hard-exit
