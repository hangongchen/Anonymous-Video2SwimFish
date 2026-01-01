"""Analytic-hydrodynamics swimming demo for the salmon skeleton (physics + log).

Drives the salmon spine (10 rigid links / 9 D6 joints, 3 rot-DOF each, +/-15 deg)
OPEN-LOOP with a head->tail travelling wave on ONE DOF per vertebra, using EFFORT
control with a manual PD (these D6 joints do not respond to position targets).
Every physics step an analytic per-link hydrodynamic wrench is applied to the
rigid skeleton:

  * anisotropic quadratic drag  (normal coeff >> tangential coeff)
  * rotational drag             (keeps the free body from tumbling)

Gravity is zeroed (neutrally buoyant -> "swim in air"). With ANISOTROPIC drag the
body wave is rectified into NET FORWARD THRUST; with --isotropic the same gait
makes ~no progress (Gray & Taylor). The slow soft mesh is deactivated so reset()
is <1s; rendering is done offline by render_swim.py (matplotlib, no RTX).

--sweep_dof runs short episodes on each per-vertebra DOF and reports forward
distance so we can pick the bending (vs twist) axis. Otherwise a single run is
saved to <out>/run_<mode>.npz.
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


parser = argparse.ArgumentParser(description="Analytic-hydrodynamics salmon swim demo")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--physics_hz", type=float, default=240.0)
parser.add_argument("--isotropic", action="store_true")
parser.add_argument("--with_deformable", action="store_true",
                    help="keep the soft body active (fast now: disable internal rigid collisions "
                         "+ sane collision offsets) and log its surface for rendering")
parser.add_argument("--sweep_dof", action="store_true", help="try DOF 0/1/2 and report forward dist")
# gait
parser.add_argument("--freq", type=float, default=2.0)
parser.add_argument("--amp", type=float, default=0.22, help="joint amplitude (rad, <0.262)")
parser.add_argument("--wavelengths", type=float, default=1.0)
parser.add_argument("--reverse_wave", action="store_true")
parser.add_argument("--dof", type=int, default=0)
parser.add_argument("--torque_amp", type=float, default=0.3, help="open-loop torque-wave amplitude (N.m)")
parser.add_argument("--joint_damping", type=float, default=4.0, help="implicit joint damping (stability)")
parser.add_argument("--recenter_stiffness", type=float, default=1.0, help="implicit stiffness recentering joints")
parser.add_argument("--torque_clip", type=float, default=1.0)
parser.add_argument("--kd_explicit", type=float, default=0.03, help="explicit joint velocity damping")
# hydro coefficients (lumped 0.5*rho*Cd*A per link)
parser.add_argument("--k_normal", type=float, default=8.0)
parser.add_argument("--k_tangent", type=float, default=0.1)
parser.add_argument("--c_lin", type=float, default=0.05)
parser.add_argument("--c_ang", type=float, default=0.01)
parser.add_argument("--c_ang_lin", type=float, default=0.01)
parser.add_argument("--force_clip", type=float, default=5.0)
# ---- MuJoCo ellipsoid fluid model (faithful port of mj ellipsoid-fluid; see
#      mujoco.readthedocs.io/en/stable/computation/fluid.html). Each bone is treated
#      as the equivalent ellipsoid derived from its inertia tensor + mass, and the
#      wrench is computed in that bone's principal frame. ----
parser.add_argument("--model", type=str, default="legacy", choices=["legacy", "mujoco"],
                    help="hydro force model: 'legacy' = hand-rolled resistive-force; "
                         "'mujoco' = MuJoCo ellipsoid fluid model (drag+viscous+Magnus+Kutta)")
parser.add_argument("--passive", action="store_true",
                    help="apply NO joint actuation (no PD/effort/kinematic wave); joints carry "
                         "only implicit damping (stiffness forced to 0). The body moves purely "
                         "under the fluid wrench -- the correct-rest test.")
parser.add_argument("--passive_test", action="store_true",
                    help="headless passivity test: start at rest (or with --shove), apply ONLY "
                         "the fluid wrench, and report whether the body stays at rest / dissipates "
                         "to rest with energy monotonically non-increasing (no self-propulsion).")
parser.add_argument("--passive_joint_damp", type=float, default=0.5,
                    help="implicit joint damping in --passive mode (reasonable, non-locking)")
parser.add_argument("--shove", type=float, default=0.0,
                    help="passivity test: initial root linear speed (m/s) along the body axis; "
                         "0 = pure rest test (must not move at all)")
parser.add_argument("--rho", type=float, default=1000.0, help="fluid density (kg/m^3); water=1000")
parser.add_argument("--visc", type=float, default=0.0,
                    help="fluid (dynamic) viscosity for the Stokes linear-drag terms; 0 disables")
parser.add_argument("--cd_blunt", type=float, default=0.5, help="MuJoCo blunt-drag coef")
parser.add_argument("--cd_slender", type=float, default=0.25, help="MuJoCo slender-drag coef")
parser.add_argument("--cd_angular", type=float, default=1.5, help="MuJoCo angular-drag coef")
parser.add_argument("--ck", type=float, default=1.0, help="MuJoCo Kutta-lift coef")
parser.add_argument("--cm", type=float, default=1.0, help="MuJoCo Magnus-lift coef")
parser.add_argument("--added_mass", action="store_true",
                    help="include the (gyroscopic, velocity-product) added-mass coupling terms "
                         "(off by default; the acceleration-based -m_A*vdot term is omitted as it "
                         "belongs on the LHS, not as an explicit external force)")
parser.add_argument("--max_lin_vel", type=float, default=1.0, help="PhysX per-link linear vel cap (m/s)")
parser.add_argument("--max_ang_vel", type=float, default=8.0)
parser.add_argument("--warmup", type=float, default=0.5)
parser.add_argument("--out", type=str, default=_P("${FISH_ROOT}/demo_out"))
# live viewer
parser.add_argument("--live", action="store_true",
                    help="open the GUI and continuously drive the gait with a follow camera "
                         "(implies not headless); close the viewport window to stop")
parser.add_argument("--render_every", type=int, default=2,
                    help="in --live, render every Nth physics step (1=smoothest/slowest)")
parser.add_argument("--live_seconds", type=float, default=0.0,
                    help="in --live, auto-stop after this many sim-seconds (0 = run until the "
                         "viewport window is closed). Used to bounded-test live stability.")
# --live --passive: actuate the MIDDLE TWO joints at t=--wiggle_at so the otherwise-passive
# fish bends/moves a little. NB these D6 joints ignore set_joint_position_target (verified);
# position control on them really has to go through the effort channel (computed-torque PD).
parser.add_argument("--wiggle", action="store_true",
                    help="in --live --passive, position-control the middle two joints starting "
                         "at --wiggle_at to make the fish move a bit")
parser.add_argument("--wiggle_at", type=float, default=5.0, help="seconds at which the wiggle starts")
parser.add_argument("--wiggle_amp", type=float, default=0.12, help="wiggle joint amplitude (rad)")
parser.add_argument("--wiggle_freq", type=float, default=1.0, help="wiggle frequency (Hz)")
parser.add_argument("--wiggle_mode", type=str, default="kinematic",
                    choices=["kinematic", "position", "effort_pd"],
                    help="how to drive the middle joints to the position setpoint. These D6 joints "
                         "ignore BOTH set_joint_position_target AND actuation-force/effort (verified: "
                         "|q| stays ~0), so the only reliable position control is 'kinematic' "
                         "(write_joint_state_to_sim prescribes the joint angle directly). "
                         "'position'/'effort_pd' kept for the record (they don't move these joints).")
parser.add_argument("--wiggle_stiffness", type=float, default=40.0, help="position-mode P gain on wiggle joints")
parser.add_argument("--wiggle_damping", type=float, default=4.0, help="position-mode D gain on wiggle joints")
parser.add_argument("--wiggle_kp", type=float, default=8.0, help="effort_pd P gain on wiggle joints")
parser.add_argument("--wiggle_kd", type=float, default=0.4, help="effort_pd D gain on wiggle joints")
# ---- SWIM demo: D6-drive fixed -> real PD position-target control. Timeline:
#      rest -> external shove (glide) -> traveling-wave POSITION gait (fish self-swims). ----
parser.add_argument("--swim", action="store_true",
                    help="rest -> shove -> self-actuated swim via PD POSITION targets on the D6 "
                         "joints (authors the missing angular DriveAPI so position control works)")
parser.add_argument("--shove_at", type=float, default=2.0, help="--swim: time of the forward shove (s)")
parser.add_argument("--gait_at", type=float, default=4.0, help="--swim: time the position gait starts (s)")
parser.add_argument("--swim_stiffness", type=float, default=60.0,
                    help="--swim: D6 angular-drive P gain (higher -> crisper gait tracking)")
parser.add_argument("--swim_damping", type=float, default=6.0, help="--swim: D6 angular-drive D gain")
# ---- DEFORMABLE passive glide: water drag computed from the DEFORMABLE body's volume/
#      shape (one ellipsoid fit to the FEM nodal cloud, NOT the thin skeleton bones),
#      applied to the skeleton (the soft body rides along via its attachments). ----
parser.add_argument("--deform_glide", action="store_true",
                    help="passive forward glide WITH the deformable body; the MuJoCo fluid wrench "
                         "is sized by the deformable's volume/extent (one ellipsoid PCA-fit to the "
                         "FEM node cloud each step), not the bones. Implies --with_deformable + --passive")
parser.add_argument("--deform_log", type=str, default=_P("${FISH_ROOT}/demo_out/deform"),
                    help="dir to save the nodal-position log (for offline rendering)")
parser.add_argument("--deform_swim", action="store_true",
                    help="DEFORMABLE fish, ACTUATED: rest -> shove forward @--shove_at (glide) -> "
                         "PD position-target gait @--gait_at (self-actuated swim). Drag sized by the "
                         "deformable volume; D6 drive authored so position control works. "
                         "Implies --with_deformable (+ swim PD gains, NOT passive)")
parser.add_argument("--free_cam", action="store_true",
                    help="in --live, do NOT auto-follow the fish (let you navigate freely)")
parser.add_argument("--no_hydro", action="store_true",
                    help="in --live, drive ONLY the kinematic body wave (no external hydro "
                         "forces) -- diagnostic to isolate render-vs-force instability")
parser.add_argument("--glide", type=float, default=0.0,
                    help="(legacy kinematic mode) forward glide speed (m/s). Unused in the "
                         "default position-control live path; 0 disables.")
# position-control actuation (matches the salmon_IL task's NaN-stable config:
# ImplicitActuatorCfg stiffness=25 damping=20 -> every joint is PD-anchored, which
# is what tames this 0.03 kg/link skeleton instead of exploding).
parser.add_argument("--stiffness", type=float, default=25.0, help="implicit actuator P gain")
parser.add_argument("--damping", type=float, default=20.0, help="implicit actuator D gain")
parser.add_argument("--armature", type=float, default=0.01,
                    help="added joint rotor inertia. These bones are ~2e-5 kg.m^2, so explicit "
                         "effort explodes (Delta-omega=tau/I*dt ~1000 rad/s/step). Armature raises "
                         "the effective inertia so effort/PD is numerically stable.")
parser.add_argument("--kinematic_live", action="store_true",
                    help="use the OLD kinematic-teleport+glide live path instead of "
                         "position control (diagnostic / fallback)")
parser.add_argument("--control", type=str, default="position",
                    choices=["position", "effort_pd", "kinematic"],
                    help="live joint drive: 'position' = set_joint_position_target (+vel ff); "
                         "'effort_pd' = computed-torque PD via set_joint_effort_target "
                         "(D6 joints respond to effort even when they ignore position drive); "
                         "'kinematic' = teleport via write_joint_state_to_sim")
parser.add_argument("--kp", type=float, default=40.0, help="effort_pd position gain")
parser.add_argument("--kd", type=float, default=3.0, help="effort_pd velocity gain")

# off-screen RTX frame capture (bypass the GUI): run headless physics + a Camera sensor,
# write RGB frames, assemble a video. Lets the hydro-force swim be RTX-rendered without the
# live viewport (tests whether the camera render path also hits the external-force+flush bug).
parser.add_argument("--capture", action="store_true", help="off-screen RTX frame capture mode")
parser.add_argument("--capture_dir", type=str, default=_P("${FISH_ROOT}/demo_out/capture"))
parser.add_argument("--capture_fps", type=float, default=30.0, help="captured frames per sim-second")
parser.add_argument("--cam_w", type=int, default=1280)
parser.add_argument("--cam_h", type=int, default=720)
parser.add_argument("--rest_seconds", type=float, default=3.0,
                    help="--capture --passive: seconds the fish sits still BEFORE the shove "
                         "(phase 1); --seconds is the glide duration after the shove (phase 2)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# the passivity test runs headless and never actuates the joints
if args_cli.passive_test:
    args_cli.passive = True
# --wiggle is a --live --passive feature (otherwise-passive fish, middle joints actuated)
if args_cli.wiggle:
    args_cli.passive = True
# --deform_glide: passive glide with the soft body, drag sized by the deformable volume
if args_cli.deform_glide:
    args_cli.passive = True
    args_cli.with_deformable = True
# --deform_swim: ACTUATED soft fish (swim PD gains, NOT passive), drag from deformable volume
if args_cli.deform_swim:
    args_cli.with_deformable = True
# --capture needs the renderer (cameras) but no visible window
if args_cli.capture:
    args_cli.enable_cameras = True
    args_cli.headless = True
# --live forces a visible viewport; otherwise default to headless (offline logging path)
elif args_cli.live:
    args_cli.headless = False
elif not getattr(args_cli, "headless", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import omni.usd
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

logging.getLogger("isaaclab.assets.articulation.articulation").setLevel(logging.ERROR)

sys.path.insert(0, _P("${FISH_ROOT}/source/FISH"))
from FISH.tasks.direct.fish.salmon_IL_cfg import SALMON_STAGE_PATH  # noqa: E402

DEVICE = args_cli.device
DT = 1.0 / args_cli.physics_hz


def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def deactivate_soft_body():
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if (p.endswith("/deformable_salmon") or "/deformable_salmon/" in p) and prim.IsActive():
            prim.SetActive(False); n += 1
    return n


def prepare_deformable_active():
    """Keep the soft body but make it load/run fast.

    Root cause of the >25min 'cook': the deformable mesh envelops the rigid bone
    colliders, and the asset authored contactOffset/restOffset = -inf, so the GPU
    contact pipeline chokes on the permanent deformable-vs-bone overlap.
    FIX1 (decisive): disable CollisionAPI on the internal rigid bones.
    FIX2 (hardening): replace the degenerate -inf/inf collision offsets.
    Returns the deformable mesh prim path (for the DeformablePrim view).
    """
    stage = omni.usd.get_context().get_stage()
    soft_path = None
    n_rigid = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith("/World/Fish"):
            continue
        if "/deformable_salmon" in p:
            if prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
                soft_path = p
                for attr, val in [("physxCollision:contactOffset", 0.01),
                                  ("physxCollision:restOffset", 0.0),
                                  ("physxDeformable:maxDepenetrationVelocity", 1.0),
                                  ("physxDeformable:selfCollisionFilterDistance", 0.01),
                                  ("physxDeformable:enableCCD", False)]:
                    a = prim.GetAttribute(attr)
                    if a and a.IsValid():
                        a.Set(val)
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            n_rigid += 1
    print(f"[demo] soft body ACTIVE: disabled {n_rigid} rigid-bone colliders, "
          f"fixed offsets on {soft_path}", flush=True)
    return soft_path


def enable_d6_drive():
    """Author a force-mode angular DriveAPI on every D6 joint's rotX/rotY/rotZ axes.

    ROOT CAUSE the salmon's D6 joints ignore set_joint_position_target: the USD ships the
    joints with ONLY PhysicsLimitAPI (no PhysicsDriveAPI) -> PhysX has no position drive,
    so the implicit-actuator stiffness is set on a DOF whose drive_type=0 and the target is
    never tracked (verified: max|q|=0 even at stiffness=25; drive_types=0). Authoring a
    DriveAPI (force mode) flips drive_type->1 and position control then actuates (verified:
    target 0.2 -> max|q|=0.13). Must run AFTER spawn and BEFORE sim.reset(); the
    ImplicitActuator overwrites the stiffness/damping placeholders at init.
    """
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in stage.Traverse():
        if "D6Joint" not in prim.GetName() or str(prim.GetTypeName()) != "PhysicsJoint":
            continue
        for axis in ("rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(100.0)   # overwritten by the ImplicitActuator at init
            drive.CreateDampingAttr().Set(10.0)
            drive.CreateMaxForceAttr().Set(1.0e6)
            drive.CreateTargetPositionAttr().Set(0.0)
        n += 1
    print(f"[demo] enabled force-mode angular DriveAPI on {n} D6 joints "
          f"(position control now actuates)", flush=True)
    return n


def _color_skeleton(rgb):
    """Tint the skeleton gprims via primvars:displayColor (the asset has no MDL
    material; the RTX renderer falls back to displayColor)."""
    from pxr import UsdGeom, Vt, Gf
    stage = omni.usd.get_context().get_stage()
    col = Vt.Vec3fArray([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])
    n = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Fish"):
            continue
        if "/deformable_salmon" in str(prim.GetPath()):
            continue
        g = UsdGeom.Gprim(prim)
        if g:
            try:
                g.CreateDisplayColorAttr(col)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    print(f"[demo] colored {n} skeleton gprims", flush=True)


def extract_ellipsoids(robot):
    """Per-body equivalent-ellipsoid semi-axes, derived from the link inertia tensor
    + mass (MuJoCo's inertia-derived-shape philosophy -- robust, asset-agnostic, no
    fragile USD geom parsing). For an ellipsoid of mass m and semi-axes (a,b,c),
    I_principal_x = (1/5) m (b^2 + c^2) (cyclic), so

        a^2 = (5/2m)(I_y + I_z - I_x)   (cyclic),

    where (I_x,I_y,I_z) are the principal moments (eigenvalues of the inertia tensor)
    and the eigenvectors give the principal axes in the link frame.

    Returns:
      semi  (nb,3)   semi-axes along the principal axes (ascending eigenvalue order:
                     semi[:,0] is the LONGEST axis -- smallest moment of inertia)
      R_pl  (nb,3,3) rotation principal->link (columns = principal axes in link frame)
      mass  (nb,)
    """
    mass = robot.data.default_mass[0].to(DEVICE).float()                       # (nb,)
    inertia = robot.data.default_inertia[0].to(DEVICE).float().reshape(-1, 3, 3)  # link frame, about CoM
    inertia = 0.5 * (inertia + inertia.transpose(-1, -2))                      # symmetrize
    evals, evecs = torch.linalg.eigh(inertia)        # evals ascending; evecs columns = principal axes (link frame)
    Ix, Iy, Iz = evals[:, 0], evals[:, 1], evals[:, 2]
    coef = 5.0 / (2.0 * mass.clamp_min(1e-9))
    rx2 = (coef * (Iy + Iz - Ix)).clamp_min(1e-10)   # longest (pairs with smallest moment Ix)
    ry2 = (coef * (Ix + Iz - Iy)).clamp_min(1e-10)
    rz2 = (coef * (Ix + Iy - Iz)).clamp_min(1e-10)
    semi = torch.sqrt(torch.stack([rx2, ry2, rz2], dim=-1))   # (nb,3)
    return semi, evecs, mass


def mujoco_fluid_wrench(V_w, W_w, Q_wl, semi, R_pl, *, rho, visc,
                        cd_blunt, cd_slender, cd_angular, ck, cm, added_mass):
    """Faithful MuJoCo ellipsoid fluid model, vectorized over bodies.

    All forces are functions of velocity only (no acceleration) -> stable to apply as
    an explicit external wrench, and PASSIVE: drag/viscous dissipate energy, lift
    (Kutta/Magnus) and the gyroscopic added-mass terms do no net work, so a body at
    rest in still fluid stays at rest and a moving body coasts to a stop.

    Inputs are world-frame (nb,3)/(nb,4 wxyz); the wrench is computed in each body's
    principal (ellipsoid) frame then rotated back to world.
    Returns world-frame force (nb,3), torque (nb,3).
    """
    eps = 1e-9
    pi = math.pi
    # world -> link -> principal frame
    v_l = quat_rotate_inverse(Q_wl, V_w)
    w_l = quat_rotate_inverse(Q_wl, W_w)
    v = torch.einsum("nji,nj->ni", R_pl, v_l)        # R_pl^T @ v_l
    w = torch.einsum("nji,nj->ni", R_pl, w_l)

    rx, ry, rz = semi[:, 0], semi[:, 1], semi[:, 2]
    rmax = semi.max(-1).values
    rmin = semi.min(-1).values
    rmid = semi.sum(-1) - rmax - rmin
    rD = semi.mean(-1)
    Vol = (4.0 / 3.0) * pi * rx * ry * rz

    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    speed = v.norm(dim=-1)
    # projected area normal to v (MuJoCo lemma); 0 where v ~ 0 (force ~ |v|v -> 0 anyway)
    den = torch.sqrt((ry**2 * rz**2 * vx**2 + rz**2 * rx**2 * vy**2 + rx**2 * ry**2 * vz**2).clamp_min(0.0))
    num = pi * torch.sqrt((ry**4 * rz**4 * vx**2 + rz**4 * rx**4 * vy**2 + rx**4 * ry**4 * vz**2).clamp_min(0.0))
    Aproj = torch.where(den > eps, num / den.clamp_min(eps), torch.zeros_like(den))
    Amax = pi * rmax * rmid

    # (1) quadratic drag: blunt (normal pressure) + slender (skin friction)
    drag_coef = (cd_blunt * Aproj + cd_slender * (Amax - Aproj)).unsqueeze(-1)
    f_D = -rho * drag_coef * (speed.unsqueeze(-1) * v)

    # (2) angular drag
    r2 = semi**2
    rj4 = torch.stack([                       # max(r_j, r_k)^4 for axis i
        torch.maximum(ry, rz) ** 4,
        torch.maximum(rx, rz) ** 4,
        torch.maximum(rx, ry) ** 4], dim=-1)
    ID = (8.0 * pi / 15.0) * semi * rj4       # (nb,3) reference angular-drag inertia
    Imax = ID.max(-1, keepdim=True).values
    wspeed = w.norm(dim=-1, keepdim=True)
    g_D = -rho * wspeed * ((cd_angular * ID + cd_slender * (Imax - ID)) * w)

    # (3) viscous (Stokes) linear drag -- low Reynolds; 0 when visc=0
    f_V = -6.0 * pi * (rD * visc).unsqueeze(-1) * v
    g_V = -8.0 * pi * (rD**3 * visc).unsqueeze(-1) * w

    # (4) Magnus lift
    f_M = cm * rho * Vol.unsqueeze(-1) * torch.linalg.cross(w, v)

    # (5) Kutta lift
    ns = torch.stack([ry * rz * vx / rx, rz * rx * vy / ry, rx * ry * vz / rz], dim=-1)
    n_hat = ns / (ns.norm(dim=-1, keepdim=True) + eps)
    v_hat = v / (speed.unsqueeze(-1) + eps)
    dotp = (v_hat * n_hat).sum(-1, keepdim=True)
    f_K = ck * rho * Aproj.unsqueeze(-1) * dotp * torch.linalg.cross(torch.linalg.cross(n_hat, v), v)

    f_p = f_D + f_V + f_M + f_K
    g_p = g_D + g_V

    # (6) optional added-mass gyroscopic coupling (velocity products only; the
    #     -m_A*vdot / -I_A*wdot acceleration terms are intentionally omitted -- they
    #     belong on the LHS, and are 0 at rest so they don't affect the rest test).
    if added_mass:
        # virtual-mass coefficients via the elliptic integral kappa_i (Tuckerman),
        # evaluated numerically; m_A,i = rho*Vol*kappa_i/(2-kappa_i).
        lam = torch.linspace(0.0, 1.0, 256, device=semi.device)[1:]            # (0,1]
        # substitute lambda = (1-u)/u  (u in (0,1]) -> integrate 0..inf
        u = lam
        L = (1.0 - u) / u
        dL = 1.0 / (u**2)
        a2 = r2[:, 0:1]; b2 = r2[:, 1:2]; c2 = r2[:, 2:3]                      # (nb,1)
        Lr = L.unsqueeze(0)                                                    # (1,K)
        def kappa(p2, q2, s2):
            integ = 1.0 / torch.sqrt((p2 + Lr)**3 * (q2 + Lr) * (s2 + Lr) + eps)
            return (rx * ry * rz).unsqueeze(-1) * integ * dL.unsqueeze(0)      # (nb,K)
        kx = torch.trapz(kappa(a2, b2, c2), u, dim=-1)
        ky = torch.trapz(kappa(b2, a2, c2), u, dim=-1)
        kz = torch.trapz(kappa(c2, a2, b2), u, dim=-1)
        kap = torch.stack([kx, ky, kz], dim=-1).clamp(0.0, 1.9)
        m_A = rho * Vol.unsqueeze(-1) * kap / (2.0 - kap).clamp_min(0.1)       # (nb,3)
        mav = m_A * v
        f_p = f_p + torch.linalg.cross(mav, w)
        g_p = g_p + torch.linalg.cross(mav, v)

    f_p = torch.nan_to_num(f_p)
    g_p = torch.nan_to_num(g_p)
    # principal -> link -> world
    f_l = torch.einsum("nij,nj->ni", R_pl, f_p)
    g_l = torch.einsum("nij,nj->ni", R_pl, g_p)
    F_w = quat_rotate(Q_wl, f_l)
    T_w = quat_rotate(Q_wl, g_l)
    return F_w, T_w


class Swimmer:
    def __init__(self, robot, init_pose, soft_view=None):
        self.robot = robot
        self.nb = robot.num_bodies
        self.nj = robot.num_joints
        self.stride = 3 if self.nj % 3 == 0 else 1
        self.verts = self.nj // self.stride
        self.sv = torch.arange(self.verts, device=DEVICE, dtype=torch.float32) / max(self.verts - 1, 1)
        self.amp_env = args_cli.amp * (0.3 + 0.7 * self.sv)
        self.k_wave = 2 * math.pi * args_cli.wavelengths * (-1.0 if args_cli.reverse_wave else 1.0)
        self.omega = 2 * math.pi * args_cli.freq
        self.init_pose = init_pose  # (7,)
        self.soft_view = soft_view
        self.n_soft = None
        if soft_view is not None:
            try:
                self.n_soft = int(soft_view.get_simulation_mesh_nodal_positions().shape[1])
            except Exception:  # noqa: BLE001
                self.n_soft = None
        # per-bone ellipsoid geometry for the MuJoCo fluid model (inertia-derived)
        self.semi, self.R_pl, self.body_mass = extract_ellipsoids(robot)
        if args_cli.model == "mujoco":
            sm = self.semi
            print(f"[demo] ellipsoid semi-axes (m): mean={sm.mean(0).tolist()} "
                  f"len(2*rmax)={float(2*sm.max(-1).values.mean()):.4f} "
                  f"thick(2*rmin)={float(2*sm.min(-1).values.mean()):.4f}", flush=True)

    def fluid_wrench(self):
        r = self.robot
        return mujoco_fluid_wrench(
            r.data.body_link_lin_vel_w[0], r.data.body_link_ang_vel_w[0],
            r.data.body_link_quat_w[0], self.semi, self.R_pl,
            rho=args_cli.rho, visc=args_cli.visc, cd_blunt=args_cli.cd_blunt,
            cd_slender=args_cli.cd_slender, cd_angular=args_cli.cd_angular,
            ck=args_cli.ck, cm=args_cli.cm, added_mass=args_cli.added_mass)

    def deform_fluid_wrench(self):
        """MuJoCo fluid wrench sized by the DEFORMABLE body's volume/shape: fit ONE
        ellipsoid to the FEM node cloud each step (PCA -> principal axes + half-extents),
        and use the deformable's OWN mean nodal velocity as the through-water velocity.
        So the drag area/volume come from the real fish body, NOT the thin skeleton bones.
        Returns the TOTAL world-frame force/torque + the fitted semi-axes."""
        sv = self.soft_view
        pos = sv.get_simulation_mesh_nodal_positions()[0].to(DEVICE).float()    # (N,3) world
        vel = sv.get_simulation_mesh_nodal_velocities()[0].to(DEVICE).float()   # (N,3) world
        centered = pos - pos.mean(0)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)             # Vh (3,3) rows = principal axes
        proj = centered @ Vh.transpose(0, 1)                                   # (N,3) principal-frame coords
        semi = proj.abs().max(0).values.clamp_min(1e-3)                        # half-extents along principal axes
        v_body = vel.mean(0)                                                   # body translational velocity (world)
        w_body = self.robot.data.body_link_ang_vel_w[0].mean(0)               # body angular velocity (world)
        ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
        R_pl = Vh.transpose(0, 1).unsqueeze(0)                                 # principal->world (cols = principal axes)
        F, T = mujoco_fluid_wrench(
            v_body.unsqueeze(0), w_body.unsqueeze(0), ident, semi.unsqueeze(0), R_pl,
            rho=args_cli.rho, visc=args_cli.visc, cd_blunt=args_cli.cd_blunt,
            cd_slender=args_cli.cd_slender, cd_angular=args_cli.cd_angular,
            ck=args_cli.ck, cm=args_cli.cm, added_mass=args_cli.added_mass)
        return F[0], T[0], semi, v_body

    def run_deform_glide(self, n_steps):
        """Passive forward glide WITH the deformable body. Water drag sized by the
        DEFORMABLE's volume/extent (deform_fluid_wrench), applied to the skeleton bones
        (PhysX has no per-node force API; the soft body rides along via its attachments).
        Shove the whole fish at --shove_at, then it decelerates under the large deformable drag.

        headless -> logs nodal positions to npz (for offline rendering).
        --live   -> renders in the GUI with a follow camera (or --free_cam). The
                    deformable + live viewport + external forces combo historically NaNs
                    the FEM, so a divergence just stops the run (no recovery attempt)."""
        assert self.soft_view is not None, "deform_glide needs the deformable view"
        r = self.robot
        sv = self.soft_view
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        live = not args_cli.headless
        if live:
            sim.set_camera_view((c0 + side * 1.6 + up * 0.7 + axis * 0.3).tolist(),
                                (c0 + axis * 0.3).tolist())
        shove_step = int(args_cli.shove_at * args_cli.physics_hz)
        cap = int(args_cli.live_seconds * args_cli.physics_hz) if (live and args_cli.live_seconds > 0) else 0
        nodal_log = np.zeros((n_steps, self.n_soft, 3), np.float32) if (self.n_soft and not live) else None
        fwd_log = np.zeros(n_steps, np.float32) if not live else None
        spd_log = np.zeros(n_steps, np.float32) if not live else None
        _, _, semi0, _ = self.deform_fluid_wrench()
        print(f"[deform] deformable-fit ellipsoid semi-axes (m): {[round(float(x),4) for x in semi0]} "
              f"-> body 2*len={float(2*semi0.max()):.3f} x 2*thick={float(2*semi0.min()):.3f} "
              f"(vs skeleton-bone thick ~0.0065 -> MUCH larger drag area)", flush=True)
        print(f"[deform] {'LIVE GUI' if live else 'headless'} passive glide: shove {args_cli.shove:.2f} m/s "
              f"@ t={args_cli.shove_at:.1f}s; drag sized by DEFORMABLE volume, applied via the skeleton; "
              f"{'close window / auto-stop' if live else str(n_steps)+' steps'}", flush=True)
        t = 0.0; step = 0; blew = False; spd0 = 0.0; cen_prev = None
        while (simulation_app.is_running() if live else step < n_steps):
            if step == shove_step and args_cli.shove > 0.0:
                rv = torch.zeros((1, 6), device=DEVICE); rv[0, :3] = axis * args_cli.shove
                r.write_root_velocity_to_sim(rv)
                try:  # give the soft body the same initial velocity so the whole fish glides together
                    nv = sv.get_simulation_mesh_nodal_velocities()
                    nv[:] = (axis * args_cli.shove).view(1, 1, 3).to(nv.dtype)
                    sv.set_simulation_mesh_nodal_velocities(nv)
                except Exception as e:  # noqa: BLE001
                    print(f"[deform] (could not set soft nodal vel: {e})", flush=True)
                r.write_data_to_sim(); sim.step(render=live); r.update(DT)
                spd0 = float(sv.get_simulation_mesh_nodal_velocities()[0].mean(0).norm())
                print(f"[deform] >>> SHOVE {args_cli.shove:.2f} m/s at t={t:.2f}s (body speed={spd0:.3f}) <<<",
                      flush=True)
            F, T, semi, v_body = self.deform_fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            forces = (F / self.nb).unsqueeze(0).expand(self.nb, 3).contiguous()
            torques = torch.zeros((self.nb, 3), device=DEVICE); torques[0] = T
            r.set_external_force_and_torque(forces.unsqueeze(0), torques.unsqueeze(0), is_global=True)
            cen = r.data.body_link_pos_w[0].mean(0)
            if live and (not args_cli.free_cam) and step % max(int(args_cli.render_every), 1) == 0:
                sim.set_camera_view((cen + side * 1.6 + up * 0.7).tolist(), cen.tolist())
            # NO joint actuation (passive)
            r.write_data_to_sim(); sim.step(render=live); r.update(DT)
            Pn = r.data.body_link_pos_w[0]
            cen_now = Pn.mean(0)
            jump = 0.0 if cen_prev is None else float((cen_now - cen_prev).norm())
            if torch.isnan(Pn).any() or jump > 5.0 or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[deform] DIVERGED at step {step} (t={t:.2f}s) -- deformable+forces"
                      f"{'+live render' if live else ''} blew up", flush=True)
                break
            cen_prev = cen_now
            spd = float(v_body.norm())
            fwd = float(((cen_now - c0) * axis).sum())
            if not live:
                fwd_log[step] = fwd; spd_log[step] = spd
                if nodal_log is not None:
                    try:
                        nodal_log[step] = sv.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy()
                    except Exception:  # noqa: BLE001
                        pass
            if step % max(int(args_cli.physics_hz), 1) == 0:
                ph = "REST " if t < args_cli.shove_at else "GLIDE"
                print(f"[deform] {ph} t={t:5.2f}s  fwd={fwd:+.3f} m  body_speed={spd:.4f} m/s", flush=True)
            t += DT; step += 1
            if cap and step >= cap:
                print(f"[deform] auto-stop after {args_cli.live_seconds:.0f}s", flush=True)
                break
        if (not live) and fwd_log is not None:
            os.makedirs(args_cli.deform_log, exist_ok=True)
            npz = os.path.join(args_cli.deform_log, "deform_glide.npz")
            save = dict(fwd=fwd_log, spd=spd_log, axis=axis.cpu().numpy(), c0=c0.cpu().numpy(),
                        semi=semi0.cpu().numpy(), dt=DT, shove=args_cli.shove, shove_at=args_cli.shove_at)
            if nodal_log is not None:
                save["nodal"] = nodal_log
            np.savez(npz, **save)
            print(f"[deform] saved {npz}  (nodal log: {'yes' if nodal_log is not None else 'no'})", flush=True)
        print("=" * 64, flush=True)
        print(f"[deform] RESULT: blew_up={blew}  (drag sized by deformable body, "
              f"2*len={float(2*semi0.max()):.3f} m)", flush=True)
        print("=" * 64, flush=True)
        return dict(blew=blew)

    def run_deform_swim(self, n_steps):
        """DEFORMABLE fish, ACTUATED: rest -> external shove @--shove_at (glide) -> PD
        position-target travelling-wave gait @--gait_at (self-actuated swim). Drag sized by
        the DEFORMABLE volume (deform_fluid_wrench), distributed to the skeleton bones; the
        D6 drive must be authored (enable_d6_drive) so position control actuates the joints.
        headless logs nodal positions to npz; --live renders (follow cam or --free_cam)."""
        assert self.soft_view is not None, "deform_swim needs the deformable view"
        r = self.robot; sv = self.soft_view
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone(); c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        live = not args_cli.headless
        if live:
            sim.set_camera_view((c0 + side * 1.6 + up * 0.7 + axis * 0.3).tolist(),
                                (c0 + axis * 0.3).tolist())
        shove_step = int(args_cli.shove_at * args_cli.physics_hz)
        cap = int(args_cli.live_seconds * args_cli.physics_hz) if (live and args_cli.live_seconds > 0) else 0
        nodal_log = np.zeros((n_steps, self.n_soft, 3), np.float32) if (self.n_soft and not live) else None
        fwd_log = np.zeros(n_steps, np.float32) if not live else None
        _, _, semi0, _ = self.deform_fluid_wrench()
        print(f"[deform-swim] deformable ellipsoid 2*len={float(2*semi0.max()):.3f} x "
              f"2*thick={float(2*semi0.min()):.3f} m; {'LIVE' if live else 'headless'} "
              f"rest -> shove@{args_cli.shove_at:.0f}s -> PD gait@{args_cli.gait_at:.0f}s "
              f"(amp={args_cli.amp}, freq={args_cli.freq})", flush=True)
        t = 0.0; step = 0; blew = False; qmax = 0.0; cen_prev = None; spd0 = 0.0
        while (simulation_app.is_running() if live else step < n_steps):
            if step == shove_step and args_cli.shove > 0.0:
                rv = torch.zeros((1, 6), device=DEVICE); rv[0, :3] = axis * args_cli.shove
                r.write_root_velocity_to_sim(rv)
                try:
                    nv = sv.get_simulation_mesh_nodal_velocities()
                    nv[:] = (axis * args_cli.shove).view(1, 1, 3).to(nv.dtype)
                    sv.set_simulation_mesh_nodal_velocities(nv)
                except Exception as e:  # noqa: BLE001
                    print(f"[deform-swim] (could not set soft nodal vel: {e})", flush=True)
                r.write_data_to_sim(); sim.step(render=live); r.update(DT)
                spd0 = float(sv.get_simulation_mesh_nodal_velocities()[0].mean(0).norm())
                print(f"[deform-swim] >>> SHOVE {args_cli.shove:.2f} m/s at t={t:.2f}s "
                      f"(body speed={spd0:.3f}) <<<", flush=True)
            # PD position-target gait: hold straight until gait_at, then a head->tail wave
            q_des = torch.zeros(self.nj, device=DEVICE)
            qd_des = torch.zeros(self.nj, device=DEVICE)
            if t >= args_cli.gait_at:
                tw = t - args_cli.gait_at
                gate = min(1.0, tw / 0.5)
                phase = self.omega * tw - self.k_wave * self.sv
                q_des[0::self.stride] = args_cli.amp * torch.sin(phase) * gate
                qd_des[0::self.stride] = args_cli.amp * self.omega * torch.cos(phase) * gate
            r.set_joint_position_target(q_des.unsqueeze(0))
            r.set_joint_velocity_target(qd_des.unsqueeze(0))
            # deformable-volume drag -> distribute to bones (soft body rides along)
            F, T, semi, v_body = self.deform_fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            forces = (F / self.nb).unsqueeze(0).expand(self.nb, 3).contiguous()
            torques = torch.zeros((self.nb, 3), device=DEVICE); torques[0] = T
            r.set_external_force_and_torque(forces.unsqueeze(0), torques.unsqueeze(0), is_global=True)
            cen = r.data.body_link_pos_w[0].mean(0)
            if live and (not args_cli.free_cam) and step % max(int(args_cli.render_every), 1) == 0:
                sim.set_camera_view((cen + side * 1.6 + up * 0.7).tolist(), cen.tolist())
            r.write_data_to_sim(); sim.step(render=live); r.update(DT)
            Pn = r.data.body_link_pos_w[0]
            cen_now = Pn.mean(0)
            jump = 0.0 if cen_prev is None else float((cen_now - cen_prev).norm())
            if torch.isnan(Pn).any() or jump > 5.0 or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[deform-swim] DIVERGED at step {step} (t={t:.2f}s)", flush=True)
                break
            cen_prev = cen_now
            if t >= args_cli.gait_at:
                qmax = max(qmax, float(r.data.joint_pos[0, 0::self.stride].abs().max()))
            fwd = float(((cen_now - c0) * axis).sum())
            if not live:
                fwd_log[step] = fwd
                if nodal_log is not None:
                    try:
                        nodal_log[step] = sv.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy()
                    except Exception:  # noqa: BLE001
                        pass
            if step % max(int(args_cli.physics_hz), 1) == 0:
                ph = "REST " if t < args_cli.shove_at else ("GLIDE" if t < args_cli.gait_at else "SWIM ")
                print(f"[deform-swim] {ph} t={t:5.2f}s  fwd={fwd:+.3f} m  "
                      f"gait peak|q|={qmax:.3f}/{args_cli.amp:.2f}", flush=True)
            t += DT; step += 1
            if cap and step >= cap:
                print(f"[deform-swim] auto-stop after {args_cli.live_seconds:.0f}s", flush=True)
                break
        if (not live) and fwd_log is not None:
            os.makedirs(args_cli.deform_log, exist_ok=True)
            npz = os.path.join(args_cli.deform_log, "deform_swim.npz")
            save = dict(fwd=fwd_log, axis=axis.cpu().numpy(), c0=c0.cpu().numpy(),
                        semi=semi0.cpu().numpy(), dt=DT, shove=args_cli.shove,
                        shove_at=args_cli.shove_at, gait_at=args_cli.gait_at, amp=args_cli.amp)
            if nodal_log is not None:
                save["nodal"] = nodal_log
            np.savez(npz, **save)
            print(f"[deform-swim] saved {npz}", flush=True)
        print("=" * 64, flush=True)
        print(f"[deform-swim] RESULT: gait peak|q|={qmax:.3f}/{args_cli.amp:.2f}  blew_up={blew}", flush=True)
        print("=" * 64, flush=True)
        return dict(blew=blew)

    def kinetic_energy(self):
        """Translational + rotational KE (J). Gravity is zeroed and the body is
        neutrally buoyant, so total mechanical energy == KE: the passivity yardstick."""
        r = self.robot
        V = r.data.body_link_lin_vel_w[0]
        W = r.data.body_link_ang_vel_w[0]
        ke_lin = 0.5 * (self.body_mass * V.pow(2).sum(-1)).sum()
        # rotational: 0.5 w^T I w in link frame
        Wl = quat_rotate_inverse(r.data.body_link_quat_w[0], W)
        I = r.data.default_inertia[0].to(DEVICE).float().reshape(-1, 3, 3)
        ke_ang = 0.5 * torch.einsum("ni,nij,nj->n", Wl, I, Wl).sum()
        return float(ke_lin + ke_ang)

    def run_passive(self, n_steps, shove):
        """PASSIVITY TEST. No joint actuation (joints carry only implicit damping);
        the body is driven ONLY by the fluid wrench. Verifies the 'correct rest'
        behaviour: from rest the body must NOT move (no self-propulsion), and with an
        initial shove it must dissipate to rest with energy never exceeding the start."""
        r = self.robot
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        if shove > 0.0:
            rv = torch.zeros((1, 6), device=DEVICE)
            rv[0, :3] = axis * shove
            r.write_root_velocity_to_sim(rv)
            r.write_data_to_sim(); sim.step(render=False); r.update(DT)
        ke0 = self.kinetic_energy()
        ke_peak = ke0
        max_disp = 0.0
        max_spd = 0.0
        t = 0.0
        blew = False
        prev_ke = ke0
        mono_violations = 0
        print(f"[passive] shove={shove:.3f} m/s  KE0={ke0:.3e} J  "
              f"({n_steps} steps, {n_steps*DT:.1f}s)", flush=True)
        for step in range(n_steps):
            F, T = self.fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)
            # NO joint targets -> joints carry only implicit damping (passive)
            r.write_data_to_sim(); sim.step(render=False); r.update(DT)
            Pn = r.data.body_link_pos_w[0]
            if torch.isnan(Pn).any() or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[passive] DIVERGED at step {step} (t={t:.2f}s)", flush=True)
                break
            ke = self.kinetic_energy()
            ke_peak = max(ke_peak, ke)
            # count meaningful energy *injection* (growth beyond a small numerical floor)
            if ke > prev_ke * 1.02 + 1e-9:
                mono_violations += 1
            prev_ke = ke
            disp = float((Pn.mean(0) - c0).norm())
            max_disp = max(max_disp, disp)
            max_spd = max(max_spd, float(r.data.body_link_lin_vel_w[0].norm(dim=-1).max()))
            if step % max(int(args_cli.physics_hz), 1) == 0:
                fwd = float(((Pn.mean(0) - c0) * axis).sum())
                print(f"[passive] t={t:5.2f}s  KE={ke:.3e} J  centroid_disp={disp:.5f} m  "
                      f"fwd={fwd:+.5f} m  max_link_spd={max_spd:.4f}", flush=True)
            t += DT
        ke_end = self.kinetic_energy() if not blew else float("nan")
        net_fwd = float(((r.data.body_link_pos_w[0].mean(0) - c0) * axis).sum())
        print("=" * 64, flush=True)
        print(f"[passive] RESULT  model={args_cli.model}  shove={shove:.3f} m/s  "
              f"soft={'yes' if self.soft_view is not None else 'no'}", flush=True)
        print(f"          KE0={ke0:.3e}  KE_peak={ke_peak:.3e}  KE_end={ke_end:.3e} J", flush=True)
        print(f"          max_centroid_disp={max_disp:.5f} m  net_forward={net_fwd:+.5f} m  "
              f"max_link_spd={max_spd:.4f} m/s  energy_injection_steps={mono_violations}", flush=True)
        if blew:
            verdict = "FAIL (diverged)"
        elif shove <= 0.0:
            # rest test: must barely move and KE must stay negligible
            ok = (max_disp < 5e-3) and (ke_peak < max(ke0, 1e-9) * 5 + 1e-7) and (mono_violations == 0)
            verdict = "PASS (stays at rest, no self-propulsion)" if ok else \
                      "FAIL (moved / gained energy at rest)"
        else:
            # decay test: energy must not exceed the start (passive) and must decay
            ok = (ke_peak <= ke0 * 1.05 + 1e-9) and (ke_end <= ke0 * 0.5) and (mono_violations <= 2)
            verdict = "PASS (dissipates to rest, energy non-increasing)" if ok else \
                      "FAIL (energy grew -> model injects energy)"
        print(f"          VERDICT: {verdict}", flush=True)
        print("=" * 64, flush=True)
        return dict(ke0=ke0, ke_peak=ke_peak, ke_end=ke_end, max_disp=max_disp,
                    net_fwd=net_fwd, blew=blew, mono_violations=mono_violations)

    def reset_state(self):
        r = self.robot
        r.write_root_pose_to_sim(self.init_pose.unsqueeze(0))
        r.write_root_velocity_to_sim(torch.zeros((1, 6), device=DEVICE))
        r.write_joint_state_to_sim(torch.zeros((1, self.nj), device=DEVICE),
                                   torch.zeros((1, self.nj), device=DEVICE))
        for _ in range(2):
            r.set_joint_effort_target(torch.zeros((1, self.nj), device=DEVICE))
            r.write_data_to_sim(); sim.step(render=False); r.update(DT)

    def run(self, dof, isotropic, n_steps, log=False):
        r = self.robot
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        proj0 = (P0 * axis).sum(-1)
        order = torch.argsort(proj0)
        fish_len = max(float((P0.max(0).values - P0.min(0).values).norm()), 1e-3)
        warm = max(int(args_cli.warmup * args_cli.physics_hz), 1)
        pos_log = np.zeros((n_steps, self.nb, 3), np.float32) if log else None
        fwd_log = np.zeros(n_steps, np.float32) if log else None
        soft_log = (np.zeros((n_steps, self.n_soft, 3), np.float32)
                    if (log and self.soft_view is not None and self.n_soft) else None)
        t = 0.0
        blew_up = False
        for step in range(n_steps):
            gate = min(1.0, step / warm)
            # KINEMATIC body wave: prescribe joint angles + rates directly (no actuation
            # dynamics -> cannot explode). The free base then translates purely under the
            # analytic hydro forces -- standard resistive-force-theory swimming.
            phase = torch.tensor(self.omega * t, device=DEVICE) - self.k_wave * self.sv
            q_des = torch.zeros(self.nj, device=DEVICE)
            qd_des = torch.zeros(self.nj, device=DEVICE)
            q_des[dof::self.stride] = args_cli.amp * torch.sin(phase) * gate
            qd_des[dof::self.stride] = args_cli.amp * self.omega * torch.cos(phase) * gate
            r.write_joint_state_to_sim(q_des.unsqueeze(0), qd_des.unsqueeze(0))

            P = r.data.body_link_pos_w[0]
            V = r.data.body_link_lin_vel_w[0]
            Wang = r.data.body_link_ang_vel_w[0]
            Ps = P[order]
            ts = torch.zeros_like(Ps)
            ts[1:-1] = Ps[2:] - Ps[:-2]; ts[0] = Ps[1] - Ps[0]; ts[-1] = Ps[-1] - Ps[-2]
            ts = normalize(ts)
            tang = torch.zeros_like(P); tang[order] = ts
            if isotropic:
                sp = V.norm(dim=-1, keepdim=True)
                F = -(args_cli.k_normal * sp * V) - args_cli.c_lin * V
            else:
                v_tan = (V * tang).sum(-1, keepdim=True) * tang
                v_nor = V - v_tan
                F = -(args_cli.k_normal * v_nor.norm(dim=-1, keepdim=True) * v_nor
                      + args_cli.k_tangent * v_tan.norm(dim=-1, keepdim=True) * v_tan) - args_cli.c_lin * V
            T = -(args_cli.c_ang * Wang.norm(dim=-1, keepdim=True) * Wang) - args_cli.c_ang_lin * Wang
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)

            r.write_data_to_sim(); sim.step(render=False); r.update(DT)
            Pn = r.data.body_link_pos_w[0]
            if torch.isnan(Pn).any():
                blew_up = True
                if log:
                    pos_log[step:] = pos_log[step - 1] if step > 0 else P0.cpu().numpy()
                break
            if log:
                pos_log[step] = Pn.detach().cpu().numpy()
                fwd_log[step] = float(((Pn.mean(0) - c0) * axis).sum())
                if soft_log is not None:
                    try:
                        sn = self.soft_view.get_simulation_mesh_nodal_positions()
                        soft_log[step] = sn[0].detach().cpu().numpy()
                    except Exception:  # noqa: BLE001
                        pass
            t += DT
        # final forward
        Pf = r.data.body_link_pos_w[0]
        net = 0.0 if blew_up else float(((Pf.mean(0) - c0) * axis).sum())
        return dict(net=net, blew_up=blew_up, axis=axis, c0=c0, order=order,
                    fish_len=fish_len, pos_log=pos_log, fwd_log=fwd_log, soft_log=soft_log)

    def run_live(self, dof, isotropic):
        """Drive the swim gait LIVE with rendering + a follow camera, until the
        viewport window is closed.

        DEFAULT = POSITION CONTROL: the body wave is a per-step joint *position
        target* (`set_joint_position_target`) tracked by the PD-anchored implicit
        actuator (stiffness/damping from --stiffness/--damping, i.e. the salmon_IL
        NaN-stable config). This is a STANDARD physics loop -- joints are actuated by
        real PD torque (the fish genuinely moves itself), it renders fine with the
        coupled `sim.step(render=True)`, and -- because there is no per-step kinematic
        teleport -- it also tolerates the analytic external hydro forces that broke the
        old kinematic path. So the fish swims FORWARD emergently under the hydro
        reaction, live, in the GUI.

        --kinematic_live falls back to the old teleport+glide path.
        """
        if args_cli.passive:
            return self.run_live_passive()
        r = self.robot
        self.reset_state()
        render_every = max(int(args_cli.render_every), 1)
        warm = max(int(args_cli.warmup * args_cli.physics_hz), 1)
        log_every = max(int(2 * args_cli.physics_hz), 1)
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        order = torch.argsort((P0 * axis).sum(-1))
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        drive = "kinematic+glide" if args_cli.kinematic_live else "position-control"
        print(f"[demo] LIVE viewer ({drive}, mode={'isotropic' if isotropic else 'aniso'}"
              f"{', soft' if self.soft_view is not None else ''}, hydro={not args_cli.no_hydro}); "
              f"close the viewport window to stop.", flush=True)
        t = 0.0
        step = 0
        cen_prev = None
        while simulation_app.is_running():
            gate = min(1.0, step / warm)
            phase = torch.tensor(self.omega * t, device=DEVICE) - self.k_wave * self.sv
            q_des = torch.zeros(self.nj, device=DEVICE)
            q_des[dof::self.stride] = args_cli.amp * torch.sin(phase) * gate
            qd_des = torch.zeros(self.nj, device=DEVICE)
            qd_des[dof::self.stride] = args_cli.amp * self.omega * torch.cos(phase) * gate
            control = "kinematic" if args_cli.kinematic_live else args_cli.control
            if control == "kinematic":
                r.write_joint_state_to_sim(q_des.unsqueeze(0), qd_des.unsqueeze(0))
            elif control == "position":
                # POSITION CONTROL: PD actuator drives the joints toward the wave. We feed
                # BOTH the position target AND the velocity feedforward so the damping term
                # tracks the moving trajectory instead of fighting it (without the velocity
                # target, d*qdot over-damps a fast gait and the joints barely move).
                r.set_joint_position_target(q_des.unsqueeze(0))
                r.set_joint_velocity_target(qd_des.unsqueeze(0))
            else:  # effort_pd -- computed-torque PD through the effort channel (D6-joint safe)
                tau = args_cli.kp * (q_des - r.data.joint_pos[0]) \
                    + args_cli.kd * (qd_des - r.data.joint_vel[0])
                tau = torch.nan_to_num(tau).clamp(-args_cli.torque_clip, args_cli.torque_clip)
                r.set_joint_effort_target(tau.unsqueeze(0))

            P = r.data.body_link_pos_w[0]
            V = r.data.body_link_lin_vel_w[0]
            Wang = r.data.body_link_ang_vel_w[0]
            Ps = P[order]
            ts = torch.zeros_like(Ps)
            ts[1:-1] = Ps[2:] - Ps[:-2]; ts[0] = Ps[1] - Ps[0]; ts[-1] = Ps[-1] - Ps[-2]
            ts = normalize(ts)
            tang = torch.zeros_like(P); tang[order] = ts
            if isotropic:
                sp = V.norm(dim=-1, keepdim=True)
                F = -(args_cli.k_normal * sp * V) - args_cli.c_lin * V
            else:
                v_tan = (V * tang).sum(-1, keepdim=True) * tang
                v_nor = V - v_tan
                F = -(args_cli.k_normal * v_nor.norm(dim=-1, keepdim=True) * v_nor
                      + args_cli.k_tangent * v_tan.norm(dim=-1, keepdim=True) * v_tan) - args_cli.c_lin * V
            T = -(args_cli.c_ang * Wang.norm(dim=-1, keepdim=True) * Wang) - args_cli.c_ang_lin * Wang
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            if args_cli.no_hydro:
                F = torch.zeros_like(F); T = torch.zeros_like(T)
            r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)

            # optional legacy kinematic forward glide (only meaningful in --kinematic_live)
            if args_cli.kinematic_live and args_cli.glide > 0.0 and step > 1:
                rv = r.data.root_vel_w.clone()
                v = rv[0, :3]
                rv[0, :3] = v - (v * axis).sum() * axis + axis * args_cli.glide
                r.write_root_velocity_to_sim(rv)

            do_render = (step % render_every == 0)
            if do_render and not args_cli.free_cam:
                cen = P.mean(0)
                eye = cen + side * 1.8 + up * 0.7
                sim.set_camera_view(eye.tolist(), cen.tolist())
            # standard physics+render step (position control needs no teleport, so the
            # coupled fabric flush is harmless here -- the IL task proves this path)
            r.write_data_to_sim(); sim.step(render=do_render); r.update(DT)

            Pn = r.data.body_link_pos_w[0]
            cen_now = Pn.mean(0)
            jump = 0.0 if cen_prev is None else float((cen_now - cen_prev).norm())
            diverged = bool(torch.isnan(Pn).any()) or jump > 5.0
            if diverged:
                print("[demo] diverged -> resetting and continuing", flush=True)
                self.reset_state(); t = 0.0; step = 0; cen_prev = None
                continue
            cen_prev = cen_now
            if step % log_every == 0:
                fwd = float(((Pn.mean(0) - c0) * axis).sum())
                spd = float(V.norm(dim=-1).max())  # max link speed (undulation cancels in the mean)
                jp = r.data.joint_pos[0]
                jq = float(jp[dof::self.stride].abs().max())  # how well joints track the wave (amp target)
                print(f"[demo] t={t:5.1f}s  forward={fwd:+.3f} m  max_link_spd={spd:.2f}  "
                      f"max|q|={jq:.3f}/{args_cli.amp:.2f} rad", flush=True)
            t += DT
            step += 1


    def run_live_passive(self):
        """LIVE GUI of the PASSIVE MuJoCo-fluid behaviour: NO joint actuation, only the
        fluid wrench. The fish sits still, gets a periodic shove, and glides to a stop --
        all visible in the interactive viewport. Renders EVERY step (the consistent-flush
        pattern proven stable with external forces in the off-screen capture). This is the
        real test of whether the INTERACTIVE viewport tolerates the (small, passive)
        external force; the divergence guard snaps the body back if the live
        fabric-flush+force conflict bites. Skeleton-only (soft body + live render-mixing
        is its own hazard)."""
        r = self.robot
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        # frame the glide path from the side (user can still orbit freely)
        sim.set_camera_view((c0 + side * 2.4 + up * 0.8 + axis * 0.4).tolist(),
                            (c0 + axis * 0.4).tolist())
        rest_steps = max(int(args_cli.rest_seconds * args_cli.physics_hz), 1)
        glide_steps = max(int(args_cli.seconds * args_cli.physics_hz), 1)
        cycle = rest_steps + glide_steps
        cap = int(args_cli.live_seconds * args_cli.physics_hz) if args_cli.live_seconds > 0 else 0

        # --wiggle: actuate the MIDDLE TWO joints' bending DOF at t=wiggle_at. With wiggle on
        # we DON'T cycle-reset (would zero the joints we are driving) -- single rest->shove->wiggle.
        n_j = self.nj // self.stride                       # number of physical joints (9)
        mid_j = [n_j // 2 - 1, n_j // 2]                   # the two central joints (e.g. 3,4 of 9)
        wdofs = [self.stride * j + args_cli.dof for j in mid_j]   # their bending-DOF global indices
        if args_cli.wiggle:
            cycle = 10 ** 9                                # disable periodic reset
            if args_cli.wiggle_mode == "position":
                # give ONLY these joints a PD gain so position control can act (rest stay passive)
                r.write_joint_stiffness_to_sim(
                    torch.full((1, len(wdofs)), args_cli.wiggle_stiffness, device=DEVICE), joint_ids=wdofs)
                r.write_joint_damping_to_sim(
                    torch.full((1, len(wdofs)), args_cli.wiggle_damping, device=DEVICE), joint_ids=wdofs)
            print(f"[demo] WIGGLE armed: joints {mid_j} (DOFs {wdofs}) start position control at "
                  f"t={args_cli.wiggle_at:.1f}s, amp={args_cli.wiggle_amp} rad, mode={args_cli.wiggle_mode}",
                  flush=True)
        print(f"[demo] LIVE PASSIVE (MuJoCo fluid, NO actuation): {args_cli.rest_seconds:.0f}s rest "
              f"-> shove {args_cli.shove:.2f} m/s -> glide-to-stop"
              f"{'' if args_cli.wiggle else ', repeating'}; "
              f"{'auto-stop ' + str(args_cli.live_seconds) + 's' if cap else 'close window to stop'}.",
              flush=True)

        def snap_to_start():
            r.write_root_pose_to_sim(self.init_pose.unsqueeze(0))
            r.write_root_velocity_to_sim(torch.zeros((1, 6), device=DEVICE))
            r.write_joint_state_to_sim(torch.zeros((1, self.nj), device=DEVICE),
                                       torch.zeros((1, self.nj), device=DEVICE))

        t = 0.0
        step = 0          # cycle phase counter (reset on divergence)
        total = 0         # absolute step counter (for the auto-stop cap)
        diverged = 0
        wig_qmax = 0.0    # running peak |q| on the wiggle joints (sampling-artifact-free)
        cen_prev = None
        while simulation_app.is_running():
            local = step % cycle
            if local == 0 and step > 0:
                snap_to_start(); cen_prev = None
            if local == rest_steps and args_cli.shove > 0.0:
                rv = torch.zeros((1, 6), device=DEVICE); rv[0, :3] = axis * args_cli.shove
                r.write_root_velocity_to_sim(rv)
            # WIGGLE: position-control the middle two joints from t=wiggle_at onward
            if args_cli.wiggle and t >= args_cli.wiggle_at:
                tw = t - args_cli.wiggle_at
                w = 2 * math.pi * args_cli.wiggle_freq
                ang = args_cli.wiggle_amp * math.sin(w * tw)
                angd = args_cli.wiggle_amp * w * math.cos(w * tw)
                if args_cli.wiggle_mode == "kinematic":
                    # prescribe the two middle joints' angle directly -- the only thing these
                    # D6 joints actually obey; the rest of the body stays passive and the
                    # fluid wrench rectifies the wiggle into a little real motion
                    r.write_joint_state_to_sim(torch.full((1, len(wdofs)), ang, device=DEVICE),
                                               torch.full((1, len(wdofs)), angd, device=DEVICE),
                                               joint_ids=wdofs)
                elif args_cli.wiggle_mode == "position":
                    r.set_joint_position_target(torch.full((1, len(wdofs)), ang, device=DEVICE),
                                                joint_ids=wdofs)
                    r.set_joint_velocity_target(torch.full((1, len(wdofs)), angd, device=DEVICE),
                                                joint_ids=wdofs)
                else:  # effort_pd -- the channel D6 joints actually respond to
                    q = r.data.joint_pos[0, wdofs]
                    qd = r.data.joint_vel[0, wdofs]
                    tau = args_cli.wiggle_kp * (ang - q) + args_cli.wiggle_kd * (angd - qd)
                    tau = torch.nan_to_num(tau).clamp(-args_cli.torque_clip, args_cli.torque_clip)
                    r.set_joint_effort_target(tau.unsqueeze(0), joint_ids=wdofs)
            F, T = self.fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            if not args_cli.no_hydro:
                r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)
            r.write_data_to_sim(); sim.step(render=True); r.update(DT)

            Pn = r.data.body_link_pos_w[0]
            cen_now = Pn.mean(0)
            jump = 0.0 if cen_prev is None else float((cen_now - cen_prev).norm())
            if bool(torch.isnan(Pn).any()) or jump > 5.0 or float(Pn.abs().max()) > 1e3:
                diverged += 1
                print(f"[demo] LIVE diverged (#{diverged}) at t={t:.2f}s (jump={jump:.1f}) "
                      f"-> snap back", flush=True)
                snap_to_start(); cen_prev = None; t = 0.0; step = 0; total += 1
                if cap and total >= cap:
                    break
                continue
            cen_prev = cen_now
            if args_cli.wiggle and t >= args_cli.wiggle_at:
                wig_qmax = max(wig_qmax, float(r.data.joint_pos[0, wdofs].abs().max()))
            if step % max(int(args_cli.physics_hz), 1) == 0:
                fwd = float(((cen_now - c0) * axis).sum())
                ph = "REST " if local < rest_steps else "GLIDE"
                wtag = ""
                if args_cli.wiggle:
                    on = "ON " if t >= args_cli.wiggle_at else "off"
                    wtag = f"  wiggle[{on}] peak|q|={wig_qmax:.3f}/{args_cli.wiggle_amp:.2f}"
                print(f"[demo] {ph} t={t:5.2f}s fwd={fwd:+.3f} m{wtag}  divergences={diverged}",
                      flush=True)
            t += DT; step += 1; total += 1
            if cap and total >= cap:
                print(f"[demo] LIVE PASSIVE auto-stop after {args_cli.live_seconds:.0f}s "
                      f"(total divergences={diverged})", flush=True)
                break

    def run_swim(self, camera=None, n_steps=None):
        """THE SWIM DEMO: rest -> external shove (glide) -> self-actuated swim via PD
        POSITION targets on the D6 joints (drive fixed by enable_d6_drive). MuJoCo fluid
        wrench every step, so the position gait rectifies into real forward thrust.
        camera != None -> off-screen RTX capture (write frames); else -> live GUI."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = self.robot
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        capture = camera is not None
        if capture:
            # static camera framing a generous forward path so all 3 phases + the
            # translation are visible (the fish traverses the frame)
            look = c0 + axis * 0.8
            cam_eye = look + side * 3.2 + up * 1.0
            camera.set_world_poses_from_view(cam_eye.unsqueeze(0), look.unsqueeze(0))
        else:
            sim.set_camera_view((c0 + side * 2.2 + up * 0.9 + axis * 0.5).tolist(),
                                (c0 + axis * 0.5).tolist())
        shove_step = int(args_cli.shove_at * args_cli.physics_hz)
        cap = int(args_cli.live_seconds * args_cli.physics_hz) if (not capture and args_cli.live_seconds > 0) else 0
        every = max(int(round(args_cli.physics_hz / max(args_cli.capture_fps, 1e-3))), 1)
        os.makedirs(args_cli.capture_dir, exist_ok=True)
        print(f"[demo] SWIM: rest -> shove@{args_cli.shove_at:.0f}s ({args_cli.shove:.2f} m/s) -> "
              f"PD-position gait@{args_cli.gait_at:.0f}s (amp={args_cli.amp}, freq={args_cli.freq}); "
              f"{'capture ' + str(n_steps) + ' steps' if capture else 'close window / auto-stop'}", flush=True)
        frame = 0; t = 0.0; step = 0; cen_prev = None; blew = False; qmax = 0.0
        while (step < n_steps) if capture else simulation_app.is_running():
            if step == shove_step and args_cli.shove > 0.0:
                rv = torch.zeros((1, 6), device=DEVICE); rv[0, :3] = axis * args_cli.shove
                r.write_root_velocity_to_sim(rv)
                print(f"[demo] >>> SHOVE {args_cli.shove:.2f} m/s at t={t:.2f}s <<<", flush=True)
            # position targets: hold straight until gait_at, then a head->tail travelling wave
            q_des = torch.zeros(self.nj, device=DEVICE)
            qd_des = torch.zeros(self.nj, device=DEVICE)
            if t >= args_cli.gait_at:
                tw = t - args_cli.gait_at
                gate = min(1.0, tw / 0.5)
                phase = self.omega * tw - self.k_wave * self.sv
                q_des[0::self.stride] = args_cli.amp * torch.sin(phase) * gate
                qd_des[0::self.stride] = args_cli.amp * self.omega * torch.cos(phase) * gate
            r.set_joint_position_target(q_des.unsqueeze(0))
            r.set_joint_velocity_target(qd_des.unsqueeze(0))
            # MuJoCo fluid wrench
            F, T = self.fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            if not args_cli.no_hydro:
                r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)
            cen = r.data.body_link_pos_w[0].mean(0)
            do_render = capture or (not args_cli.headless)   # headless+no-capture = fast physics check
            # live follow-cam re-aims at the fish each render step -- UNLESS --free_cam,
            # in which case we leave the viewport alone so you can orbit/zoom freely
            if do_render and (not capture) and (not args_cli.free_cam) \
                    and step % max(int(args_cli.render_every), 1) == 0:
                eye = cen + side * 1.8 + up * 0.7
                sim.set_camera_view(eye.tolist(), cen.tolist())
            r.write_data_to_sim(); sim.step(render=do_render); r.update(DT)

            Pn = r.data.body_link_pos_w[0]
            cen_now = Pn.mean(0)
            jump = 0.0 if cen_prev is None else float((cen_now - cen_prev).norm())
            if bool(torch.isnan(Pn).any()) or jump > 5.0 or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[demo] diverged at t={t:.2f}s", flush=True)
                if capture:
                    break
                self.reset_state(); t = 0.0; step = 0; cen_prev = None; continue
            cen_prev = cen_now
            if t >= args_cli.gait_at:
                qmax = max(qmax, float(r.data.joint_pos[0, 0::self.stride].abs().max()))
            if capture and step % every == 0:
                camera.update(DT)
                rgb = camera.data.output["rgb"]; rgb = rgb[0] if rgb.ndim == 4 else rgb
                rgb = rgb[..., :3].detach().to("cpu").numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                plt.imsave(os.path.join(args_cli.capture_dir, f"frame_{frame:05d}.png"), rgb)
                frame += 1
            if step % max(int(args_cli.physics_hz), 1) == 0:
                fwd = float(((cen_now - c0) * axis).sum())
                ph = "REST " if t < args_cli.shove_at else ("GLIDE" if t < args_cli.gait_at else "SWIM ")
                print(f"[demo] {ph} t={t:5.2f}s  fwd={fwd:+.3f} m  gait peak|q|={qmax:.3f}/{args_cli.amp:.2f}",
                      flush=True)
            t += DT; step += 1
            if cap and step >= cap:
                print(f"[demo] auto-stop after {args_cli.live_seconds:.0f}s", flush=True)
                break
        print(f"[demo] SWIM done: {frame} frames, blew_up={blew}", flush=True)
        return frame, blew

    def capture_run(self, dof, isotropic, camera, n_steps, every):
        """Headless physics + off-screen RTX capture. Uses the proven kinematic body
        wave + analytic hydro forces (the +2.17 m headless swim), renders EVERY step
        (consistent fabric flush -- avoids the render=True/False mixing that NaNs the
        soft body) and writes a PNG every `every` steps from a follow camera.
        Returns frames written (and whether the hydro+flush combo blew up)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = self.robot
        self.reset_state()
        warm = max(int(args_cli.warmup * args_cli.physics_hz), 1)
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        order = torch.argsort((P0 * axis).sum(-1))
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        os.makedirs(args_cli.capture_dir, exist_ok=True)
        frame = 0
        t = 0.0
        blew = False
        print(f"[capture] starting; first RTX frame compiles shaders (minutes on Blackwell)...",
              flush=True)
        for step in range(n_steps):
            gate = min(1.0, step / warm)
            phase = torch.tensor(self.omega * t, device=DEVICE) - self.k_wave * self.sv
            q_des = torch.zeros(self.nj, device=DEVICE)
            qd_des = torch.zeros(self.nj, device=DEVICE)
            q_des[dof::self.stride] = args_cli.amp * torch.sin(phase) * gate
            qd_des[dof::self.stride] = args_cli.amp * self.omega * torch.cos(phase) * gate
            r.write_joint_state_to_sim(q_des.unsqueeze(0), qd_des.unsqueeze(0))

            P = r.data.body_link_pos_w[0]
            V = r.data.body_link_lin_vel_w[0]
            Wang = r.data.body_link_ang_vel_w[0]
            Ps = P[order]
            ts = torch.zeros_like(Ps)
            ts[1:-1] = Ps[2:] - Ps[:-2]; ts[0] = Ps[1] - Ps[0]; ts[-1] = Ps[-1] - Ps[-2]
            ts = normalize(ts)
            tang = torch.zeros_like(P); tang[order] = ts
            if isotropic:
                sp = V.norm(dim=-1, keepdim=True)
                F = -(args_cli.k_normal * sp * V) - args_cli.c_lin * V
            else:
                v_tan = (V * tang).sum(-1, keepdim=True) * tang
                v_nor = V - v_tan
                F = -(args_cli.k_normal * v_nor.norm(dim=-1, keepdim=True) * v_nor
                      + args_cli.k_tangent * v_tan.norm(dim=-1, keepdim=True) * v_tan) - args_cli.c_lin * V
            T = -(args_cli.c_ang * Wang.norm(dim=-1, keepdim=True) * Wang) - args_cli.c_ang_lin * Wang
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            if args_cli.no_hydro:
                F = torch.zeros_like(F); T = torch.zeros_like(T)
            r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)

            do_cap = (step % every == 0)
            if do_cap:
                cen = P.mean(0)
                eye = cen + side * 1.0 + up * 0.4 - axis * 0.15
                camera.set_world_poses_from_view(eye.unsqueeze(0), cen.unsqueeze(0))
            r.write_data_to_sim(); sim.step(render=True); r.update(DT)

            Pn = r.data.body_link_pos_w[0]
            if bool(torch.isnan(Pn).any()) or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[capture] DIVERGED at step {step} (t={t:.2f}s) -- camera flush hit the "
                      f"hydro+force conflict. Captured {frame} frames before blow-up.", flush=True)
                break
            if do_cap:
                camera.update(DT)
                rgb = camera.data.output["rgb"]
                rgb = rgb[0] if rgb.ndim == 4 else rgb
                rgb = rgb[..., :3].detach().to("cpu").numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                plt.imsave(os.path.join(args_cli.capture_dir, f"frame_{frame:05d}.png"), rgb)
                if frame % 15 == 0:
                    fwd = float(((Pn.mean(0) - c0) * axis).sum())
                    print(f"[capture] frame {frame:04d}  t={t:5.2f}s  forward={fwd:+.3f} m", flush=True)
                frame += 1
            t += DT
        print(f"[capture] done: {frame} frames in {args_cli.capture_dir} (blew_up={blew})", flush=True)
        return frame, blew

    def capture_passive(self, camera, rest_steps, glide_steps, every, shove):
        """Off-screen RTX video of the PASSIVE MuJoCo-fluid behaviour, one shot:
        phase 1 the fish sits still (no actuation; fluid wrench == 0 at rest), then an
        impulse shoves it and phase 2 shows it GLIDE and decelerate to a stop under
        fluid drag alone. A STATIC camera frames the whole glide path so the absolute
        motion -- and the stop -- is visible (a follow-cam would hide the translation).
        Off-screen capture escapes the live-viewport hydro+flush divergence."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = self.robot
        self.reset_state()
        P0 = r.data.body_link_pos_w[0].clone()
        c0 = P0.mean(0)
        _, _, vh = torch.linalg.svd(P0 - c0, full_matrices=False)
        axis = normalize(vh[0])
        up = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        side = normalize(torch.linalg.cross(axis, up))
        # static camera framing the expected glide path (start c0 -> ~0.8 m along +axis)
        look = c0 + axis * 0.45
        eye = look + side * 2.4 + up * 0.7
        camera.set_world_poses_from_view(eye.unsqueeze(0), look.unsqueeze(0))
        os.makedirs(args_cli.capture_dir, exist_ok=True)
        frame = 0
        t = 0.0
        blew = False
        n_steps = rest_steps + glide_steps
        print(f"[capture] PASSIVE video: {rest_steps} rest + {glide_steps} glide steps, "
              f"shove={shove:.2f} m/s; first RTX frame compiles shaders (minutes on Blackwell)...",
              flush=True)
        for step in range(n_steps):
            if step == rest_steps and shove > 0.0:
                rv = torch.zeros((1, 6), device=DEVICE)
                rv[0, :3] = axis * shove
                r.write_root_velocity_to_sim(rv)
                print(f"[capture] >>> SHOVE applied at t={t:.2f}s ({shove:.2f} m/s along body axis) <<<",
                      flush=True)
            F, T = self.fluid_wrench()
            F = torch.nan_to_num(F).clamp(-args_cli.force_clip, args_cli.force_clip)
            T = torch.nan_to_num(T).clamp(-args_cli.force_clip, args_cli.force_clip)
            r.set_external_force_and_torque(F.unsqueeze(0), T.unsqueeze(0), is_global=True)
            # NO joint actuation -> joints carry only implicit damping (passive)
            r.write_data_to_sim(); sim.step(render=True); r.update(DT)

            Pn = r.data.body_link_pos_w[0]
            if bool(torch.isnan(Pn).any()) or float(Pn.abs().max()) > 1e3:
                blew = True
                print(f"[capture] DIVERGED at step {step} (t={t:.2f}s)", flush=True)
                break
            if step % every == 0:
                camera.update(DT)
                rgb = camera.data.output["rgb"]
                rgb = rgb[0] if rgb.ndim == 4 else rgb
                rgb = rgb[..., :3].detach().to("cpu").numpy()
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * (255.0 if rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                plt.imsave(os.path.join(args_cli.capture_dir, f"frame_{frame:05d}.png"), rgb)
                if frame % 15 == 0:
                    fwd = float(((Pn.mean(0) - c0) * axis).sum())
                    ph = "REST " if step < rest_steps else "GLIDE"
                    print(f"[capture] {ph} frame {frame:04d} t={t:5.2f}s fwd={fwd:+.3f} m", flush=True)
                frame += 1
            t += DT
        print(f"[capture] PASSIVE done: {frame} frames in {args_cli.capture_dir} (blew_up={blew})",
              flush=True)
        return frame, blew


def build_robot():
    cfg = ArticulationCfg(
        prim_path="/World/Fish/skeleton",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SALMON_STAGE_PATH),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16,
                solver_velocity_iteration_count=2, sleep_threshold=0.0, stabilization_threshold=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, max_linear_velocity=args_cli.max_lin_vel,
                max_angular_velocity=args_cli.max_ang_vel,
                max_depenetration_velocity=1.0, enable_gyroscopic_forces=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={"all": ImplicitActuatorCfg(
            joint_names_expr=[".*"], effort_limit_sim=1e5, armature=args_cli.armature,
            # swim: PD position gains on the (now-driveable) D6 joints; passive: stiffness=0
            # (NO restoring self-force), only damping; else PD-anchored
            stiffness=(args_cli.swim_stiffness if (args_cli.swim or args_cli.deform_swim)
                       else 0.0 if args_cli.passive else args_cli.stiffness),
            damping=(args_cli.swim_damping if (args_cli.swim or args_cli.deform_swim)
                     else args_cli.passive_joint_damp if args_cli.passive else args_cli.damping))})
    return Articulation(cfg)


def _assemble_video(frame_dir, fps):
    import glob
    import subprocess
    pngs = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not pngs:
        return
    out = os.path.join(frame_dir, "swim_rtx.mp4")
    fps_i = str(int(max(fps, 1)))
    try:
        subprocess.run(["ffmpeg", "-y", "-framerate", fps_i,
                        "-i", os.path.join(frame_dir, "frame_%05d.png"),
                        "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out],
                       check=True, capture_output=True)
        print(f"[capture] wrote {out} ({len(pngs)} frames)", flush=True)
        return
    except Exception as e:  # noqa: BLE001
        print(f"[capture] ffmpeg failed ({e}); trying imageio", flush=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(out, fps=int(max(fps, 1))) as w:
            for p in pngs:
                w.append_data(imageio.imread(p))
        print(f"[capture] wrote {out} ({len(pngs)} frames)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[capture] video assembly failed ({e}); PNGs remain in {frame_dir}", flush=True)


def main():
    global sim
    os.makedirs(args_cli.out, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, render_interval=1, gravity=(0.0, 0.0, 0.0), device=DEVICE,
        physx=sim_utils.PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)))
    if args_cli.live or args_cli.capture:
        # lights so the fish is visible (asset has no MDL shaders). Keep the dome dim
        # so the white background doesn't blow out; a key light gives the body shading.
        dome = sim_utils.DomeLightCfg(intensity=120.0, color=(0.35, 0.45, 0.6))
        dome.func("/World/DomeLight", dome)
        key = sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 0.98, 0.92))
        key.func("/World/KeyLight", key, orientation=(0.86, 0.35, 0.0, 0.36))
    robot = build_robot()
    if args_cli.swim or args_cli.deform_swim:
        # author the missing angular DriveAPI so set_joint_position_target actuates the
        # D6 joints (must be after spawn, before sim.reset())
        enable_d6_drive()
    if args_cli.capture or args_cli.live:
        # the bones have no material -> tint them so they read against the bright dome
        _color_skeleton((0.92, 0.45, 0.30))
    capture_cam = None
    if args_cli.capture:
        from isaaclab.sensors import Camera, CameraCfg
        cam_cfg = CameraCfg(
            prim_path="/World/capture_cam", update_period=0.0,
            height=args_cli.cam_h, width=args_cli.cam_w, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=35.0, clipping_range=(0.05, 1.0e5)))
        capture_cam = Camera(cam_cfg)
    if args_cli.with_deformable:
        soft_path = prepare_deformable_active()
    else:
        print(f"[demo] deactivated {deactivate_soft_body()} soft prims", flush=True)
        soft_path = None
    t0 = time.time(); sim.reset(); print(f"[demo] reset {time.time()-t0:.2f}s", flush=True)
    for _ in range(2):
        robot.write_data_to_sim(); sim.step(render=False); robot.update(DT)

    soft_view = None
    if soft_path is not None:
        from isaacsim.core.prims import DeformablePrim
        try:
            soft_view = DeformablePrim(prim_paths_expr=soft_path, reset_xform_properties=False)
            soft_view.initialize()
            nodal = soft_view.get_simulation_mesh_nodal_positions()
            print(f"[demo] deformable view OK: nodal shape {tuple(nodal.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[demo] deformable view init failed ({e}); continuing without soft log", flush=True)
            soft_view = None

    init_pose = robot.data.body_link_pose_w[0, 0].clone()  # root link pose (7,) wxyz
    print(f"[demo] num_bodies={robot.num_bodies} num_joints={robot.num_joints} "
          f"mass={float(robot.data.default_mass[0].sum()):.4f}kg", flush=True)
    swimmer = Swimmer(robot, init_pose, soft_view=soft_view)

    if args_cli.deform_glide:
        if soft_view is None:
            print("[demo] ERROR: deform_glide needs the deformable view but it failed to init", flush=True)
            return
        n = int(args_cli.seconds * args_cli.physics_hz)
        swimmer.run_deform_glide(n_steps=n)
        return

    if args_cli.deform_swim:
        if soft_view is None:
            print("[demo] ERROR: deform_swim needs the deformable view but it failed to init", flush=True)
            return
        n = int((args_cli.gait_at + args_cli.seconds) * args_cli.physics_hz)  # rest+glide+swim
        swimmer.run_deform_swim(n_steps=n)
        return

    if args_cli.passive_test:
        n = int(args_cli.seconds * args_cli.physics_hz)
        swimmer.run_passive(n_steps=n, shove=args_cli.shove)
        return

    if args_cli.swim:
        # total = rest+glide (gait_at) + swim (--seconds)
        n = int((args_cli.gait_at + args_cli.seconds) * args_cli.physics_hz)
        if args_cli.capture:
            for _ in range(3):
                robot.write_data_to_sim(); sim.step(render=True); robot.update(DT)
            capture_cam.update(0.0)
            frames, blew = swimmer.run_swim(camera=capture_cam, n_steps=n)
            if frames > 0:
                _assemble_video(args_cli.capture_dir, args_cli.capture_fps)
        else:
            swimmer.run_swim(camera=None)
        return

    if args_cli.capture:
        every = max(int(round(args_cli.physics_hz / max(args_cli.capture_fps, 1e-3))), 1)
        # warm the renderer + camera (first RTX frame compiles shaders -> minutes on Blackwell)
        for _ in range(3):
            robot.write_data_to_sim(); sim.step(render=True); robot.update(DT)
        capture_cam.update(0.0)
        if args_cli.passive:
            # PASSIVE MuJoCo-fluid video: sit still -> shove -> glide to a stop
            rest_n = int(args_cli.rest_seconds * args_cli.physics_hz)
            glide_n = int(args_cli.seconds * args_cli.physics_hz)
            frames, blew = swimmer.capture_passive(camera=capture_cam, rest_steps=rest_n,
                                                   glide_steps=glide_n, every=every,
                                                   shove=args_cli.shove)
        else:
            n = int(args_cli.seconds * args_cli.physics_hz)
            frames, blew = swimmer.capture_run(args_cli.dof, isotropic=args_cli.isotropic,
                                               camera=capture_cam, n_steps=n, every=every)
        if frames > 0:
            _assemble_video(args_cli.capture_dir, args_cli.capture_fps)
        return

    if args_cli.live:
        sim.set_camera_view([2.2, -2.2, 2.3], [0.0, 0.0, 1.5])
        swimmer.run_live(args_cli.dof, isotropic=args_cli.isotropic)
        return

    if args_cli.sweep_dof:
        n = int(min(args_cli.seconds, 5.0) * args_cli.physics_hz)
        print(f"[demo] DOF sweep ({n} steps each), aniso:", flush=True)
        for dof in range(swimmer.stride):
            res = swimmer.run(dof, isotropic=False, n_steps=n, log=False)
            print(f"  DOF{dof}: net_forward={res['net']:+.4f} m "
                  f"({res['net']/res['fish_len']:+.2f} BL)  blew_up={res['blew_up']}", flush=True)
        return

    n = int(args_cli.seconds * args_cli.physics_hz)
    mode = "isotropic" if args_cli.isotropic else "aniso"
    if args_cli.with_deformable:
        mode = mode + "_soft"
    print(f"[demo] run mode={mode} dof={args_cli.dof} n_steps={n}", flush=True)
    res = swimmer.run(args_cli.dof, isotropic=args_cli.isotropic, n_steps=n, log=True)
    npz = os.path.join(args_cli.out, f"run_{mode}.npz")
    save_kwargs = dict(pos=res["pos_log"], fwd=res["fwd_log"], axis=res["axis"].cpu().numpy(),
                       c0=res["c0"].cpu().numpy(), order=res["order"].cpu().numpy(),
                       fish_len=res["fish_len"], dt=DT, freq=args_cli.freq, amp=args_cli.amp,
                       mode=mode, k_normal=args_cli.k_normal, k_tangent=args_cli.k_tangent)
    if res.get("soft_log") is not None:
        save_kwargs["soft_pos"] = res["soft_log"]
        print(f"[demo] logged soft-body nodal positions: {res['soft_log'].shape}", flush=True)
    np.savez(npz, **save_kwargs)
    dur = n * DT
    print("=" * 60, flush=True)
    print(f"[demo] MODE={mode} dof={args_cli.dof} net_forward={res['net']:+.4f} m "
          f"({res['net']/res['fish_len']:+.2f} BL) speed={res['net']/dur:+.4f} m/s "
          f"blew_up={res['blew_up']}\n       saved {npz}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
    # Kit's simulation_app.close() HANGS in teardown on this setup (Blackwell + FEM
    # deformable) -- the process spins at ~110% CPU holding GPU memory and never exits
    # (a hang, not an exception, so it can't be caught). All deliverables are produced
    # before main() returns, so skip close() and hard-exit -- the OS reclaims the GPU
    # context on process death.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
