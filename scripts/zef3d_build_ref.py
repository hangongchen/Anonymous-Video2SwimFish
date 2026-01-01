#!/usr/bin/env python
"""Phase A: build the 3D AMP reference -> demo_out/zef05_amp_ref/amp_reference_3d_v2.npz.

Phi per frame (46 numbers):
  bend  = 40 : left/right + up/down at each of K=20 body points, in the body's own frame,
               / TRUE 3D body length (tilt-free). Head at point 0.  UNCHANGED from v1.
  motion=  6 : ORIENTATION-AWARE (new in v2). See the ORIENTATION note below.
                 40 v_nose   signed speed along the fish's NOSE axis (BL/s). >0 head-first,
                             <0 tail-first.  (v1 stored |v_horizontal| -- a magnitude.)
                 41 v_left   signed speed along the body's LEFT axis (BL/s) = sideways slide.
                 42 v_up     signed speed along the body's UP axis (BL/s).
                 43 travel_cos  cos(angle between travel direction and the nose axis):
                             +1 head-first, 0 broadside, -1 tail-first. Scale-free.
                 44 yaw_body    rate of turn of the BODY's nose azimuth (rad/s).
                 45 pitch_body  rate of turn of the BODY's nose elevation (rad/s).

Only broadside frames (hx>0.5, both views unoccluded) have a 3D bend; segment_id groups consecutive
runs so the discriminator only pairs neighbouring frames. Smoothed, downsampled 60->30 fps.

ORIENTATION (why v2 exists)
---------------------------
v1's four motion channels were `[|v_horizontal|, v_z_world, d/dt(heading of the VELOCITY),
d/dt(pitch of the VELOCITY)]`. Every one of them is measured in the WORLD frame or is a bare
magnitude, and the 40 bend channels are expressed in the body's OWN frame -- so NOTHING in the
44-dim feature told the discriminator how the body is oriented RELATIVE TO ITS DIRECTION OF TRAVEL.
A fish swimming head-first at 0.5 BL/s and a fish sliding BROADSIDE at 0.5 BL/s with the same body
bend produced the SAME Phi. The discriminator was structurally blind to the exact failure the policy
had learned (measured: median 70 deg between travel direction and the body line, sideways in 98% of
frames), and the two reward hacks `rew_scales.backward` / `rew_scales.offaxis` existed only to patch
what the style reward could not see.

v2 projects the velocity onto the body's OWN basis -- the SAME (nose, left, up) triad the bend
profile is already measured in -- so head-first / sideways / tail-first are three different points in
feature space, and the turn rates come from the BODY's nose axis instead of the (noisy, and at low
speed meaningless) direction of travel.

SIGN TRAP: the historical `fwd` inside bend3d() is the anterior TANGENT, which runs head -> tail, so
it points BACKWARD out of the fish's nose. Projecting velocity onto it would have made head-first
swimming read NEGATIVE. v2 defines `nose = -tangent` explicitly and projects on that; the builder
asserts the real fish comes out swimming head-first (v_nose > 0 in the large majority of frames),
which is a self-check on this sign.
"""
import argparse
import json, re
import numpy as np
import cv2

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


ROOT = _P("${ZEF_ROOT}")
SEG = _P("${FISH_ROOT}/demo_out/zef05_amp_ref").replace("zef05_amp_ref", "") + "zef05_seg/ZebraFish-05"
OUT = _P("${FISH_ROOT}/demo_out/zef05_amp_ref/amp_reference_3d_v2.npz")
K = 20
BL_FIX_CM = 2.27          # mean measured 3D body length (for velocity BL/s normalization)
# below this speed the travel DIRECTION is meaningless (it is all tracking noise), so travel_cos is
# reported as 0 ("no opinion") rather than a random unit vector.
TRAVEL_COS_MIN_BL = 0.05


def load_jsonc(p):
    s = re.sub(r"/\*.*?\*/", "", open(p).read(), flags=re.S)
    return json.loads(re.sub(r"//.*", "", s))


def calib(tag):
    intr = load_jsonc(f"{ROOT}/cam{tag}_intrinsic.json"); refs = load_jsonc(f"{ROOT}/cam{tag}_references.json")
    Kc = np.array(intr["K"], float); dist = np.array(intr["Distortion"][0], float)
    world = np.array([[p["world"]["x"], p["world"]["y"], p["world"]["z"]] for p in refs], float)
    cam = np.array([[p["camera"]["x"], p["camera"]["y"]] for p in refs], float)
    ok, rvec, tvec = cv2.solvePnP(world, cam, Kc, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return dict(K=Kc, dist=dist, P=Kc @ np.hstack([R, tvec]))


def clean_mask(mask, head, bbox, margin=0.20):
    m = (mask > 127).astype(np.uint8); l, t, w, h = [float(v) for v in bbox]
    mx, my = int(margin * w), int(margin * h); keep = np.zeros_like(m)
    keep[max(0, int(t - my)):int(t + h + my), max(0, int(l - mx)):int(l + w + mx)] = 1
    m *= keep
    n, lab, st, ct = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return m
    hx, hy = int(round(head[0])), int(round(head[1]))
    L = lab[hy, hx] if (0 <= hy < lab.shape[0] and 0 <= hx < lab.shape[1]) else 0
    if L == 0:
        L = int(np.argmax(st[1:, cv2.CC_STAT_AREA])) + 1
    return (lab == L).astype(np.uint8)


def midline_px(mask, head, bbox, nb=20):
    m = clean_mask(mask, head, bbox); ys, xs = np.nonzero(m)
    if len(xs) < 40:
        return None
    P = np.stack([xs, ys], 1).astype(float); ax = P.mean(0) - head; ax /= np.linalg.norm(ax) + 1e-9
    al = (P - head) @ ax; e = np.linspace(np.percentile(al, 1), np.percentile(al, 99), nb + 1)
    pts = [P[(al >= e[i]) & (al <= e[i + 1])].mean(0) for i in range(nb) if ((al >= e[i]) & (al <= e[i + 1])).sum() > 3]
    if len(pts) < 6:
        return None
    pts = np.array(pts); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    s = cum / cum[-1]; tgt = np.linspace(0, 1, K)
    return np.stack([np.interp(tgt, s, pts[:, 0]), np.interp(tgt, s, pts[:, 1])], 1)


def undist(pts, cam):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(float), cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)


def triangulate(mT, mF, cT, cF):
    X = cv2.triangulatePoints(cT["P"], cF["P"], undist(mT, cT).T, undist(mF, cF).T)
    return (X[:3] / X[3]).T


def bend3d(M3d):
    """Body-frame bend profile AND the basis it was measured in.

    Returns (lat, ver, bl, basis) with basis = (nose, left, up), unit vectors in world coords.
    The bend channels are byte-identical to v1 -- only the basis is newly RETURNED (v1 built it,
    used it, and threw it away, which is why the velocity could never be expressed in it).

    `tan` is the anterior tangent and runs HEAD -> TAIL; the nose points the other way, so
    nose = -tan. Keep using `tan` for lat/ver so the bend stays exactly as before.
    """
    head = M3d[0]; rel = M3d - head
    tan = rel[max(1, K // 4)] - rel[0]; tan /= np.linalg.norm(tan) + 1e-9
    up = np.array([0.0, 0, 1.0]); up = up - up.dot(tan) * tan; up /= np.linalg.norm(up) + 1e-9
    left = np.cross(up, tan)
    bl = np.linalg.norm(np.diff(M3d, axis=0), axis=1).sum()
    return (rel @ left) / bl, (rel @ up) / bl, bl, (-tan, left, up)


def body_rates(nose, dt, clip_yaw=10.0, clip_pitch=8.0):
    """Turn rates of the BODY's nose axis over one contiguous run of frames.

    nose: (T,3) unit vectors. Returns (yaw, pitch) each (T,) in rad/s.
    v1 differentiated the direction of the VELOCITY vector instead; that is ~meaningless when the
    fish is slow or sliding (the velocity direction is then decoupled from where the fish points),
    and it is what let a broadside-drifting policy still post 'normal-looking' turn rates."""
    T = len(nose)
    if T < 2:
        return np.zeros(T), np.zeros(T)
    psi = np.unwrap(np.arctan2(nose[:, 1], nose[:, 0]))
    th = np.arcsin(np.clip(nose[:, 2], -1.0, 1.0))
    w = 5 if T >= 5 else 1
    yaw = np.clip(np.gradient(smooth(psi, w)) / dt, -clip_yaw, clip_yaw)
    pitch = np.clip(np.gradient(smooth(th, w)) / dt, -clip_pitch, clip_pitch)
    return yaw, pitch


def smooth(a, w=5):
    """Box filter with EDGE-REPLICATE padding.

    PREPROCESSING FIX (v2): this used `np.convolve(..., mode="same")`, which ZERO-pads. The first and
    last ~w/2 samples of every array were therefore averaged against zeros and pulled toward 0. That
    is harmless on the 900-frame global head track, but it is applied AGAIN per contiguous run when
    the feature is smoothed -- and the runs here have a MEDIAN LENGTH OF 1 FRAME (mean 4.8), so for
    most runs EVERY sample sat in the attenuated edge region. The stored reference amplitudes were
    being shrunk by up to ~40% on short runs, and the shrink varied with run length, which injects a
    spurious run-length-correlated signal the discriminator can key on. Replicate-padding keeps the
    signal level flat across the whole run."""
    w = int(max(1, w))
    if w == 1:
        return a.astype(float).copy()
    k = np.ones(w) / w
    pad = w // 2

    def _one(x):
        xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
        return np.convolve(xp, k, mode="valid")

    if a.ndim == 1:
        return _one(np.asarray(a, float))
    a = np.asarray(a, float)
    return np.stack([_one(a[:, i]) for i in range(a.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT, help="output npz (v2, orientation-aware)")
    args = ap.parse_args()

    cT, cF = calib("T"), calib("F")
    gt = np.loadtxt(f"{ROOT}/gt/gt.txt", delimiter=","); gt = gt[np.argsort(gt[:, 0])]
    FR = gt[:, 0].astype(int); dt = 1.0 / 60.0
    # ---- 3D head trajectory -> world velocity (every frame) ----
    # MEDIAN filter first to kill single-frame head-tracking jumps (they gave v_fwd spikes to 145 BL/s),
    # then smooth. CLIPPING now happens per BODY-FRAME channel below, after the projection.
    from scipy.signal import medfilt
    posm = np.stack([medfilt(gt[:, 2 + k], 5) for k in range(3)], 1)
    pos = smooth(posm, 9); vel = np.gradient(pos, axis=0) / dt              # cm/s, world
    sp_h = np.linalg.norm(vel[:, :2], axis=1)
    # v1's four WORLD-frame / magnitude channels, kept ONLY so the same rebuild can emit the legacy
    # feature for a like-for-like old-vs-new comparison (see `legacy_motion` in the npz). They are
    # NOT part of Phi any more -- none of them can distinguish head-first from broadside travel.
    lg_v_fwd = np.clip(sp_h / BL_FIX_CM, 0.0, 10.0)                         # BL/s (fish burst max ~10)
    lg_v_vert = np.clip(vel[:, 2] / BL_FIX_CM, -5.0, 5.0)
    hd = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    lg_yaw = np.clip(smooth(np.gradient(smooth(hd, 9)) / dt, 5), -10, 10)
    pit = np.unwrap(np.arctan2(vel[:, 2], sp_h + 1e-9))
    lg_pitch = np.clip(smooth(np.gradient(smooth(pit, 9)) / dt, 5), -8, 8)
    # ---- broadside mask + runs ----
    velxy = np.gradient(gt[:, 2:4], axis=0); spd = np.linalg.norm(velxy, axis=1) + 1e-9
    hx = np.abs(velxy[:, 0]) / spd
    broad = (gt[:, 11] == 0) & (gt[:, 18] == 0) & (hx > 0.5) & (spd > 0.05)
    # WHERE THE DATA GOES. The reference is small (a few hundred frames out of 900) and it matters
    # which filter is responsible, because AMP sees only within-run TRANSITIONS -- a filter that
    # punches single-frame holes destroys far more pairs than its frame count suggests.
    print(f"[build3d] broadside filter on {len(FR)} annotated frames: "
          f"occluded-top {int((gt[:,11]!=0).sum())}, occluded-front {int((gt[:,18]!=0).sum())}, "
          f"not-broadside(hx<=0.5) {int((hx<=0.5).sum())}, too-slow(spd<=0.05) {int((spd<=0.05).sum())} "
          f"-> {int(broad.sum())} pass")
    # ---- 3D bend on broadside frames ----
    bend, seg, kept, blens = [], [], [], []
    basis, velw, mids, legacy = [], [], [], []
    run = 0; prev = False
    for i, fr in enumerate(FR):
        if not broad[i]:
            prev = False; continue
        r = gt[i]
        mT = midline_px(cv2.imread(f"{SEG}/imgT/mask/{fr:06d}.png", 0), r[5:7], r[7:11])
        mF = midline_px(cv2.imread(f"{SEG}/imgF/mask/{fr:06d}.png", 0), r[12:14], r[14:18])
        if mT is None or mF is None:
            prev = False; continue
        M3d = triangulate(mT, mF, cT, cF)
        lat, ver, bl, (nose, left, up) = bend3d(M3d)
        if not (1.5 < bl < 4.0):
            prev = False; continue
        if not prev:
            run += 1
        prev = True
        bend.append(np.concatenate([lat, ver]).astype(np.float32))
        basis.append(np.stack([nose, left, up]))                            # (3,3) rows nose/left/up
        velw.append(vel[i]); mids.append(M3d)
        legacy.append([lg_v_fwd[i], lg_v_vert[i], lg_yaw[i], lg_pitch[i]])
        seg.append(run); kept.append(fr); blens.append(bl)
    bend = np.array(bend); seg = np.array(seg, np.int64); kept = np.array(kept, np.int64)
    basis = np.array(basis); velw = np.array(velw); mids = np.array(mids)
    legacy = np.array(legacy, np.float32)

    # ---- ORIENTATION-AWARE motion: project the world velocity onto the BODY's own (nose,left,up) ----
    # This is the whole point of v2. The bend profile already lives in this basis; the velocity now
    # does too, so "which way is the fish pointing relative to where it is going" is IN the feature.
    #
    # SPLIT INTO DIRECTION + MAGNITUDE, deliberately. The obvious encoding is the signed body-frame
    # velocity [v_nose, v_left, v_up] in BL/s, but all three of those scale with SPEED -- and the real
    # fish here cruises at ~3 BL/s while the sim fish manages ~0.1-0.5 BL/s. That domain gap would sit
    # on top of the orientation signal in every one of the three channels, and a discriminator can win
    # on the speed gap alone (a saturated D gives r_style==0 everywhere and NO gradient -- the
    # documented failure mode of the ZebraFish-01 reference). Splitting confines the speed gap to ONE
    # channel and leaves THREE clean, scale-free channels that say only "which way is the fish
    # pointing relative to where it is going" -- exactly the thing the old feature could not express.
    v_body = np.einsum("nij,nj->ni", basis, velw) / BL_FIX_CM               # (N,3) BL/s [nose,left,up]
    speed_bl = np.clip(np.linalg.norm(velw, axis=1) / BL_FIX_CM, 0.0, 10.0)
    # unit travel direction IN THE BODY FRAME. dir_nose is the cosine between travel and the nose:
    # +1 head-first, 0 broadside, -1 tail-first. Below TRAVEL_COS_MIN_BL the direction is pure
    # tracking noise, so report the zero vector ("no opinion") instead of a random unit vector.
    _n = np.linalg.norm(v_body, axis=1, keepdims=True)
    vdir = np.where(_n > TRAVEL_COS_MIN_BL, v_body / np.maximum(_n, 1e-9), 0.0)
    dir_nose, dir_left, dir_up = np.clip(vdir, -1.0, 1.0).T
    # ---- turn rates from the BODY's nose axis, per contiguous run (dt = 1/60 within a run) ----
    yaw_b = np.zeros(len(seg)); pitch_b = np.zeros(len(seg))
    for s in np.unique(seg):
        m = seg == s
        yaw_b[m], pitch_b[m] = body_rates(basis[m, 0], dt)
    motion = np.stack([speed_bl, dir_nose, dir_left, dir_up, yaw_b, pitch_b], 1).astype(np.float32)
    feat = np.concatenate([bend, motion], 1).astype(np.float32)             # (N, 2K+6)

    # SELF-CHECK on the nose sign: real zebrafish swim head-first, so the travel direction must lie
    # along +nose in the large majority of frames. If this trips, `nose = -tan` in bend3d() has the
    # wrong sign and the whole reference would teach the policy to swim BACKWARDS.
    frac_head_first = float((dir_nose > 0).mean())
    assert frac_head_first > 0.8, (
        f"nose-axis sign check FAILED: only {frac_head_first:.1%} of reference frames are head-first "
        f"(dir_nose>0). The real fish does not swim backwards -- `nose` in bend3d() is flipped.")

    # ---- temporal smooth WITHIN each run, then downsample 60->30 fps (stride 2) ----
    fs = feat.copy(); lgs = legacy.copy()
    for s in np.unique(seg):
        m = seg == s
        if m.sum() >= 5:
            fs[m] = smooth(feat[m], 5)
            lgs[m] = smooth(legacy[m], 5)
    take = np.zeros(len(fs), bool)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        take[idx[::2]] = True
    fs, seg2, kept2 = fs[take], seg[take], kept[take]
    # DROP RUNS TOO SHORT TO FORM A TRANSITION. AMP trains on (f_t, f_t+1) pairs WITHIN a segment, so
    # a 1-frame run contributes no pair -- but it still entered feat_mean/feat_std in _init_disc, and
    # its body turn rates are forced to 0 (body_rates needs >=2 frames). That put an 11%-of-rows spike
    # at yaw=pitch=0 into the normalization statistics of channels that are otherwise wide. Dropping
    # them changes the training pairs not at all and makes the z-scoring reflect the data D sees.
    keep_run = np.isin(seg2, [s for s in np.unique(seg2) if (seg2 == s).sum() >= 2])
    n_drop = int((~keep_run).sum())
    fs, seg2, kept2 = fs[keep_run], seg2[keep_run], kept2[keep_run]
    take_idx = np.where(take)[0][keep_run]
    pairs = int(np.sum(seg2[1:] == seg2[:-1]))
    names = ["speed_bl", "dir_nose", "dir_left", "dir_up", "yaw_body", "pitch_body"]
    np.savez(args.out, feat=fs, segment_id=seg2, frame=kept2, K=K, n_bend_per_station=2,
             n_vel=len(names), body_len_cm=float(np.mean(blens)),
             vel_names=np.array(names),
             # v1's motion block on the SAME frames, for the old-vs-new comparison / video overlay
             legacy_motion=lgs[take_idx],
             legacy_vel_names=np.array(["v_fwd", "v_vert", "yaw", "pitch"]),
             # geometry for the dataset video (world cm): the triangulated midline and the body basis
             midline3d=mids[take_idx].astype(np.float32), basis=basis[take_idx].astype(np.float32),
             vel_world=velw[take_idx].astype(np.float32),
             bl_cm=np.array(blens, np.float32)[take_idx])
    print(f"[build3d] saved {args.out}")
    print(f"[build3d] feat {fs.shape} (40 bend + {len(names)} motion) | {len(np.unique(seg2))} runs "
          f"| {pairs} transition pairs")
    print(f"[build3d] bend tail amp: left/right {np.abs(fs[:, K-1]).mean()*np.sqrt(2):.3f}  "
          f"up/down {np.abs(fs[:, 2*K-1]).mean()*np.sqrt(2):.3f} BL")
    print(f"[build3d] ORIENTATION self-check: {frac_head_first:.1%} of frames head-first (dir_nose>0)")
    for j, n in enumerate(names):
        c = fs[:, 2 * K + j]
        print(f"[build3d]   ch{2*K+j} {n:11s} mean {c.mean():+7.3f} std {c.std():6.3f} "
              f"min {c.min():+7.3f} max {c.max():+7.3f}")
    print(f"[build3d] median |angle(travel, nose)| = "
          f"{np.degrees(np.arccos(np.clip(np.median(fs[:, 2*K+1]), -1, 1))):.1f} deg "
          f"(0 = perfect head-first swimming)")
    # SEGMENT HEALTH: AMP only ever sees TRANSITIONS, so a 1-frame run contributes nothing. Report it
    # -- half the runs here are single frames, which is why 50 runs yield only ~190 pairs.
    lens = np.array([int((seg2 == s).sum()) for s in np.unique(seg2)])
    print(f"[build3d] segments: {len(lens)} runs kept, lengths min {lens.min()} median "
          f"{int(np.median(lens))} max {lens.max()}; dropped {n_drop} pair-less single-frame rows")
    print(f"[build3d] frame yield: {len(FR)} annotated -> {len(bend)} usable ({len(bend)/len(FR):.1%}) "
          f"-> {len(fs)} after 60->30 fps downsample")


if __name__ == "__main__":
    main()
