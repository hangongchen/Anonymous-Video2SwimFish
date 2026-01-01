"""ZeF PCA-mode PLAYBACK on the articulated+FEM Misty Minnow -- offline validation, NO RL.

Pipeline (one Isaac process, one env per experiment):
  PCA coefficients -> reconstructed curvature kappa_bl(s,t) -> curvature->joint-target map
  (least-squares on an IN-SIM calibrated influence matrix Phi) -> PD position targets ->
  articulated + FEM fish -> per-step state recording (analysis + videos are OFFLINE).

Phases:
  settle      zero action, lets the FEM body relax
  calib_j0..6 bend each lateral (:1 family) D6 joint alone, hold, measure the achieved
              joint angle + the midline curvature profile  ->  Phi (20 stations x 7 joints)
  resettle    zero action again
  play        every env runs its own experiment (see EXPERIMENTS below)

Experiments (env index = row):
   0 coast          zero action (drift/thrust baseline)
   1 pc1            clip playback, PC1 only
   2 pc12_95        clip playback, PC1+PC2  (= the 95% reconstruction, n95=2)
   3 pc123          clip playback, PC1..PC3
   4 pc1234_99      clip playback, PC1..PC4 (99%)
   5 pc_all         clip playback, all 20 PCs (full raw reconstruction)
   6..9 sweep_pc1-4 single-mode visualization: kappa = mean + 2*sigma_i*sin(2*pi*0.25Hz*t)*v_i
  10 pc12_lead      same kappa as env2, joint targets LEAD-compensated by tau = damping/stiffness
                    (separates PD-lag error from unrealizable-mode error)
  11 pc12_half      the 2-PC clip at HALF speed (actuator-bandwidth test)
  12 pc12_voronoi   the 2-PC clip through the simple integrate-kappa-over-joint-cell mapping
                    (the "hand-designed" comparator for the least-squares map)

The 8.4 s clip is the longest contiguous VALID run of ZeF-05 (frames 395:900, 60 fps);
full-speed envs play it twice (loop; the wrap step is recorded so analysis can mask it).

Runtime patches only -- the training env source is NOT modified:
  * success radius forced negative + curricula off (reaching must not teleport/reset the fish)
  * rewards stubbed to zero (skips the AMP discriminator entirely)
  * NEVER env.reset() with the FEM body (known CUDA crash) -- zero-action settling instead.

Run:
  ${FISH_PYTHON} scripts/zef_playback/run_playback.py \
      --device cuda:0            # add --smoke for a 2-minute end-to-end shakedown
Output: demo_out/zef_playback/recording.npz (+ meta.json)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(_P("${FISH_ROOT}"))
os.environ.setdefault("FISH_TEST_DT_HZ", "120")          # live-run-certified dt for this fish
for _p in (str(REPO / "source"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=str(REPO / "demo_out/zef_playback"))
parser.add_argument("--smoke", action="store_true", help="tiny step counts, end-to-end shakedown")
parser.add_argument("--calib_amp", type=float, default=0.6, help="calibration action (1.0 = 25 deg)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402

import FISH.tasks                # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import curvature_utils as cu     # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


TASK = "Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0"
BASIS_NPZ = REPO / "demo_out/zef_manifold/pca_basis.npz"
CLIP = (395, 900)                # longest contiguous valid ZeF-05 run, 60 fps
SWEEP_HZ = 0.25                  # quasi-static single-mode sweep frequency

EXPERIMENTS = [
    {"name": "coast",        "kind": "coast"},
    {"name": "pc1",          "kind": "clip",    "pcs": [0]},
    {"name": "pc12_95",      "kind": "clip",    "pcs": [0, 1]},
    {"name": "pc123",        "kind": "clip",    "pcs": [0, 1, 2]},
    {"name": "pc1234_99",    "kind": "clip",    "pcs": [0, 1, 2, 3]},
    {"name": "pc_all",       "kind": "clip",    "pcs": list(range(20))},
    {"name": "sweep_pc1",    "kind": "sweep",   "pc": 0},
    {"name": "sweep_pc2",    "kind": "sweep",   "pc": 1},
    {"name": "sweep_pc3",    "kind": "sweep",   "pc": 2},
    {"name": "sweep_pc4",    "kind": "sweep",   "pc": 3},
    {"name": "pc12_lead",    "kind": "clip",    "pcs": [0, 1], "lead": True},
    {"name": "pc12_half",    "kind": "clip",    "pcs": [0, 1], "half_speed": True},
    {"name": "pc12_voronoi", "kind": "clip",    "pcs": [0, 1], "mapping": "voronoi"},
]
E = len(EXPERIMENTS)

# ------------------------------------------------------------------------------- env creation
cfg = parse_env_cfg(TASK, device=args.device, num_envs=E)
env = gym.make(TASK, cfg=cfg).unwrapped
dev = env.device

# runtime neutralization of the TRAINING machinery (no source edits):
env.cfg.curriculum = False
env.cfg.dist_curriculum = False
env.cfg.multi_target = False
env._cur_radius = -1.0                            # distance < -1 is never true -> no success reset
env._get_rewards = lambda: torch.zeros(env.num_envs, device=dev)   # skip AMP disc + reward books
print("### playback patches: success OFF (radius -1), curricula OFF, rewards stubbed", flush=True)

names = [env.robot.data.joint_names[i] for i in env._control_joint_ids]
nact = len(names)
DT = env.cfg.decimation * env.cfg.sim.dt
scale = float(env.cfg.pos_action_scale)
stiff = float(env.robot.data.joint_stiffness[0, 0])
damp = float(env.robot.data.joint_damping[0, 0])
tau = damp / stiff                                 # first-order PD lag time constant
print(f"### {nact} dofs, control dt={DT:.4f}s, pos_action_scale={scale:.4f} rad "
      f"({np.rad2deg(scale):.1f} deg), k={stiff:.0f} c={damp:.0f} -> tau={tau*1e3:.1f} ms", flush=True)

# lateral swim family ':1', ordered head->tail. Chain NAME index 0 is the TAIL end (verified in
# the scripted-wave test), so head-first = descending chain number. Order only matters for the
# printed labels + the Voronoi comparator -- the least-squares map is order-independent.
def _chain(nm):
    b = nm.split(":")[0]
    return int(b.split("_")[-1]) if "_" in b else 0

fam1 = sorted([i for i, n in enumerate(names) if n.split(":")[-1] == "1"],
              key=lambda i: -_chain(names[i]))
J = len(fam1)
print(f"### lateral (:1) joints head->tail: {[names[i] for i in fam1]}", flush=True)
assert J == 7, f"expected 7 lateral joints, got {J}"

geo = cu.rest_geometry(env.cfg.panel_hydro_path)
BL = geo["body_length"]
print(f"### body_length={BL:.4f} m, link order head->tail = {geo['link_order'].tolist()}", flush=True)

# ------------------------------------------------------------------------------- phase schedule
if args.smoke:
    N_SETTLE, N_HOLD, N_RELAX, N_RESET = 20, 25, 10, 15
    T_PLAY = 60
else:
    N_SETTLE, N_HOLD, N_RELAX, N_RESET = 60, 75, 45, 60
    T_PLAY = 506                                  # 2x the 253-step full-speed clip
N_MEAS = 10 if args.smoke else 15                 # steps averaged at the end of each hold

phases = [("settle", N_SETTLE)]
phases += [(f"calib_j{j}", N_HOLD) for j in range(J) for _ in (0,)]
# interleave relax after each calib hold
sched = [("settle", N_SETTLE)]
for j in range(J):
    sched += [(f"calib_j{j}", N_HOLD), (f"relax_j{j}", N_RELAX)]
sched += [("resettle", N_RESET), ("play", T_PLAY)]
T_TOTAL = sum(n for _, n in sched)
print(f"### schedule: {sched} -> {T_TOTAL} control steps = {T_TOTAL*DT:.1f} s sim time", flush=True)

# ------------------------------------------------------------------------------- recorders
nb = env.robot.data.body_link_pos_w.shape[1]
rec = {
    "joint_pos":  np.zeros((T_TOTAL, E, nact), np.float32),
    "joint_vel_max": np.zeros((T_TOTAL, E), np.float32),
    "action":     np.zeros((T_TOTAL, E, nact), np.float32),
    "bone_pos":   np.zeros((T_TOTAL, E, nb, 3), np.float32),
    "bone_quat":  np.zeros((T_TOTAL, E, nb, 4), np.float32),
    "root_state": np.zeros((T_TOTAL, E, 13), np.float32),
    "terminated": np.zeros((T_TOTAL, E), bool),
    "ep_len":     np.zeros((T_TOTAL, E), np.int32),
}
phase_of_step = np.zeros(T_TOTAL, dtype="U12")
_step = 0


def step_record(a, tag):
    """env.step + record everything for every env."""
    global _step
    _, _, term, _, _ = env.step(a)
    d = env.robot.data
    rec["joint_pos"][_step] = d.joint_pos[:, env._control_joint_ids].cpu().numpy()
    rec["joint_vel_max"][_step] = d.joint_vel.abs().max(1).values.cpu().numpy()
    rec["action"][_step] = a.cpu().numpy()
    rec["bone_pos"][_step] = d.body_link_pos_w.cpu().numpy()
    rec["bone_quat"][_step] = d.body_link_quat_w.cpu().numpy()
    rec["root_state"][_step] = d.root_state_w.cpu().numpy()
    rec["terminated"][_step] = term.cpu().numpy()
    rec["ep_len"][_step] = env.episode_length_buf.cpu().numpy()
    phase_of_step[_step] = tag
    _step += 1
    if _step % 100 == 0:
        print(f"  [{_step}/{T_TOTAL}] phase={tag} jvel_max={rec['joint_vel_max'][_step-1].max():.1f} "
              f"blew={int(rec['terminated'][:_step].sum())}", flush=True)


def kappa_all_envs(t_idx):
    """Median-over-envs kappa profile at recorded step t_idx (uses ALL envs -- identical commands
    during calibration)."""
    ks = []
    for e in range(E):
        p = cu.kappa_from_bones(rec["bone_pos"][t_idx, e], rec["bone_quat"][t_idx, e], geo)
        if p is not None:
            ks.append(p["kappa_bl"])
    return np.median(np.stack(ks), 0)


zero = torch.zeros(E, nact, device=dev)
t0 = time.time()

# ------------------------------------------------------------------------------- settle
for _ in range(N_SETTLE):
    step_record(zero, "settle")
kappa_rest = np.mean([kappa_all_envs(_step - 1 - i) for i in range(N_MEAS)], 0)
print(f"### rest kappa_bl: max|.|={np.abs(kappa_rest).max():.3f}", flush=True)

# ------------------------------------------------------------------------------- calibration
# Bend each lateral joint alone, hold to steady state, measure achieved q (all 7) + kappa.
# Phi solves K = Q @ Phi^T over the 7 experiments -- the tiny off-diagonal coupling of the held
# joints is thereby accounted for, not assumed away.
Q_mat = np.zeros((J, J))                          # achieved q per experiment (rows) x joint (cols)
K_mat = np.zeros((J, cu.K_STATIONS))              # measured kappa deviation per experiment
calib_cmd = args.calib_amp * scale                # commanded angle (rad)
for j in range(J):
    a = zero.clone()
    a[:, fam1[j]] = args.calib_amp
    for _ in range(N_HOLD):
        step_record(a, f"calib_j{j}")
    qm = rec["joint_pos"][_step - N_MEAS:_step].mean(0)          # (E, nact) steady-state
    Q_mat[j] = np.median(qm[:, fam1], 0)
    K_mat[j] = np.mean([kappa_all_envs(_step - 1 - i) for i in range(N_MEAS)], 0) - kappa_rest
    track = Q_mat[j, j] / calib_cmd
    speak = int(np.argmax(np.abs(K_mat[j])))
    print(f"### calib {names[fam1[j]]}: cmd {np.rad2deg(calib_cmd):.1f} deg -> ach "
          f"{np.rad2deg(Q_mat[j, j]):.1f} deg ({100*track:.0f}%), kappa peak at s="
          f"{cu.S_STATIONS[speak]:.2f} (|{K_mat[j][speak]:.2f}|)", flush=True)
    for _ in range(N_RELAX):
        step_record(zero, f"relax_j{j}")

Phi = np.linalg.lstsq(Q_mat, K_mat, rcond=None)[0].T             # (20, J): kappa per rad of joint
# joint s-location = |Phi|-weighted centroid, NOT argmax: peak stations tie at the spline ends
# (verification found argmax ties gave 2 joints EMPTY Voronoi cells -> zero commands in the
# recorded 2026-08-08 run; the recorded env12 therefore over/under-drives joints -- see REPORT).
s_joint = (cu.S_STATIONS[:, None] * np.abs(Phi)).sum(0) / np.abs(Phi).sum(0)
g_int = Phi.sum(0) * (1.0 / (cu.K_STATIONS - 1))                 # integrated angle per rad (Voronoi)
print(f"### Phi built. joint peak-s head->tail: {np.round(s_joint, 2).tolist()}", flush=True)
print(f"### integrated gain g_j (should be ~+/-1): {np.round(g_int, 2).tolist()}", flush=True)

# ------------------------------------------------------------------------------- command builders
basis = np.load(BASIS_NPZ)
comp = basis["components"].astype(np.float64)
mean_k = basis["mean"].astype(np.float64)
coeffs = basis["coeffs"][CLIP[0]:CLIP[1]].astype(np.float64)     # (505, 20) all-finite clip
sigma = basis["coeffs"][np.isfinite(basis["coeffs"]).all(1)].std(0)
assert np.isfinite(coeffs).all(), "clip has invalid frames"

lam = 1e-3 * np.mean(np.diag(Phi.T @ Phi))
Phi_pinv = np.linalg.solve(Phi.T @ Phi + lam * np.eye(J), Phi.T)  # ridge LS solve (J, 20)

# Voronoi cells: stations assigned to the nearest joint peak-s
cell_of_station = np.argmin(np.abs(cu.S_STATIONS[:, None] - s_joint[None, :]), 1)


def kappa_to_q(kap, mapping="ls"):
    """kappa_bl (T,20) -> joint targets (T,J) in rad."""
    dk = kap - kappa_rest[None, :]
    if mapping == "voronoi":
        th = np.zeros((kap.shape[0], J))
        ds = 1.0 / (cu.K_STATIONS - 1)
        for j in range(J):
            th[:, j] = dk[:, cell_of_station == j].sum(1) * ds
        # floor |g| at 0.5 (sign kept): a near-zero integrated gain (self-cancelling Phi column)
        # otherwise amplifies that joint's command ~8x into the clip (verification finding).
        g_safe = np.sign(g_int) * np.maximum(np.abs(g_int), 0.5)
        return th / g_safe[None, :]
    return dk @ Phi_pinv.T


def build_kappa(exp, T):
    """Target kappa_bl (T,20) for one experiment at the 30 Hz control rate."""
    if exp["kind"] == "sweep":
        t = np.arange(T) * DT
        c = 2.0 * sigma[exp["pc"]] * np.sin(2 * np.pi * SWEEP_HZ * t)
        return mean_k[None, :] + np.outer(c, comp[exp["pc"]])
    # clip playback
    if exp.get("half_speed"):
        cc = coeffs                                   # 60 fps frames played at 30 Hz = half speed
    else:
        cc = coeffs[::2]                              # every 2nd frame = real time
    ksel = np.zeros((cc.shape[0], cu.K_STATIONS))
    ksel += mean_k[None, :]
    for p in exp["pcs"]:
        ksel += np.outer(cc[:, p], comp[p])
    reps = int(np.ceil(T / ksel.shape[0]))
    return np.tile(ksel, (reps, 1))[:T]


A_play = np.zeros((T_PLAY, E, nact), np.float32)
clip_frac = np.zeros(E)
kappa_targets = np.zeros((T_PLAY, E, cu.K_STATIONS), np.float32)
for e, exp in enumerate(EXPERIMENTS):
    if exp["kind"] == "coast":
        continue
    kap = build_kappa(exp, T_PLAY)
    kappa_targets[:, e] = kap
    q = kappa_to_q(kap, exp.get("mapping", "ls"))
    if exp.get("lead"):
        dq = np.gradient(q, DT, axis=0)
        q = q + tau * dq
    a = q / scale
    clip_frac[e] = float((np.abs(a) > 1.0).mean())
    A_play[:, e, fam1] = np.clip(a, -1.0, 1.0)
    print(f"### {exp['name']:13s}: |q| p95={np.rad2deg(np.percentile(np.abs(q), 95)):.1f} deg, "
          f"clipped {100*clip_frac[e]:.1f}% of commands", flush=True)

wrap_step = (0 if args.smoke else 253)               # full-speed clip loops here (mask in analysis)

# ------------------------------------------------------------------------------- resettle + play
for _ in range(N_RESET):
    step_record(zero, "resettle")
A_t = torch.tensor(A_play, device=dev)
for t in range(T_PLAY):
    step_record(A_t[t], "play")

print(f"### sim done in {time.time()-t0:.0f}s wall, blow-up terminations: "
      f"{int(rec['terminated'].sum())}", flush=True)

# ------------------------------------------------------------------------------- save
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = "recording_smoke" if args.smoke else "recording"
np.savez_compressed(
    out / f"{tag}.npz",
    **rec, phase_of_step=phase_of_step,
    kappa_rest=kappa_rest, Phi=Phi, Q_mat=Q_mat, K_mat=K_mat, s_joint=s_joint, g_int=g_int,
    kappa_targets=kappa_targets, A_play=A_play, clip_frac=clip_frac,
    link_order=geo["link_order"], body_length=BL,
    tip_nose_bone=geo["tips"]["nose"][0], tip_nose_r=geo["tips"]["nose"][1],
    tip_tail_bone=geo["tips"]["tail"][0], tip_tail_r=geo["tips"]["tail"][1],
    fam1=np.array(fam1), joint_names=np.array(names), dt=DT, pos_action_scale=scale,
    stiffness=stiff, damping=damp, tau=tau, wrap_step=wrap_step, clip_range=np.array(CLIP),
    sigma=sigma, sweep_hz=SWEEP_HZ,
)
meta = {"experiments": EXPERIMENTS, "schedule": sched, "task": TASK, "smoke": bool(args.smoke),
        "dt": DT, "T_total": T_TOTAL, "clip_frames": CLIP, "sweep_hz": SWEEP_HZ,
        "calib_amp": args.calib_amp, "body_length": BL}
(out / ("meta_smoke.json" if args.smoke else "meta.json")).write_text(json.dumps(meta, indent=2))
print(f"### saved {out / (tag + '.npz')}", flush=True)
print("PLAYBACK_DONE", flush=True)
# NOTE: simulation_app.close() HANGS on this box (Blackwell + FEM) -- exit hard instead.
os._exit(0)
