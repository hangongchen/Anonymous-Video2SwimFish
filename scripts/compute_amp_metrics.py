#!/usr/bin/env python
"""Full steady-window metrics for an AMP swim rollout, including the AMP score from the TRAINED disc.

Computes everything the task spec asks for:
  forward speed, lateral-speed ratio, angle of attack, yaw drift, tail-beat frequency, oscillation
  amplitude, AMP score mean/std, energy per distance (cost of transport), blow-up count.

The AMP score is recomputed OFFLINE from the saved discriminator (demo_out/.../disc_state.pth): the
bend feature is extracted from the rollout EXACTLY as the env does (world-horizontal, heading-relative,
head-anchored), normalized with the disc's stored feat_mean/std, paired into transitions, scored, and
mapped through the LSGAN reward r = clamp(1 - 0.25(D-1)^2, 0, 1).

Usage:
  python scripts/compute_amp_metrics.py --demo <rollout.npz> \
      --disc demo_out/zef05_amp_ref/disc_state.pth \
      --reference demo_out/zef05_amp_ref/amp_reference.npz [--label amp05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_amp_features import head_anchored_bend_profile

BL = 0.5   # sim asset body length (m)


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bend_features(pc, root, K):
    """world-horizontal, heading-relative, head-anchored -- identical to env _bend_feature."""
    out = []
    for t in range(len(pc)):
        rel = pc[t] - root[t, 0:3]
        fwd = rot(root[t, 3:7]) @ np.array([-1.0, 0.0, 0.0]); fwd[2] = 0
        fwd = fwd / (np.linalg.norm(fwd[:2]) + 1e-9)
        perp = np.array([-fwd[1], fwd[0], 0.0])
        along = rel @ fwd; lateral = rel @ perp
        out.append(head_anchored_bend_profile(np.stack([along, lateral, 0 * along], 1), K))
    return np.stack(out)


class MLP(nn.Module):
    def __init__(self, sd):
        super().__init__()
        # reconstruct a [Linear,ReLU]* + Linear stack from the saved state_dict shapes
        idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net.")})
        layers, weights = [], [(i, sd[f"net.{i}.weight"]) for i in idxs if f"net.{i}.weight" in sd]
        for n, (i, w) in enumerate(weights):
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if n < len(weights) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def amp_score(feat, disc_path, K):
    ck = torch.load(disc_path, map_location="cpu")
    sd = ck["disc"]
    mean = ck["feat_mean"].cpu().numpy(); std = ck["feat_std"].cpu().numpy()
    disc = MLP(sd); disc.load_state_dict(sd); disc.eval()
    fn = (feat - mean) / std
    trans = np.concatenate([fn[:-1], fn[1:]], axis=1).astype(np.float32)
    with torch.no_grad():
        score = disc(torch.tensor(trans)).squeeze(-1).numpy()
    r_style = np.clip(1.0 - 0.25 * (score - 1.0) ** 2, 0.0, 1.0)
    return score, r_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--disc", default="demo_out/zef05_amp_ref/disc_state.pth")
    ap.add_argument("--reference", default="demo_out/zef05_amp_ref/amp_reference.npz")
    ap.add_argument("--label", default="policy")
    ap.add_argument("--K", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.demo)
    pc, root = d["pc"].astype(float), d["root"].astype(float)
    jvel = d["joint_vel"].astype(float) if "joint_vel" in d.files else None
    tau = d["applied_torque"].astype(float) if "applied_torque" in d.files else None
    dt = float(d["step_dt"]) if "step_dt" in d.files else 1 / 30.
    T = len(pc)
    pos = root[:, 0:3]
    vel = root[:, 7:10]
    speed = np.linalg.norm(vel, axis=1)

    # STEADY WINDOW: drop first 15% (transient) and exclude the terminal hydro-lift launch (a sustained
    # speed ramp to >3x the median -- a force-model artifact, not swimming; see the launch memo).
    lo = int(0.15 * T)
    steady_med = np.median(speed[lo:])
    launch = speed > max(3.0 * steady_med, 0.6)
    hi = T
    for t in range(int(0.6 * T), T):            # find the launch onset in the last 40%
        if launch[t]:
            hi = t; break
    S = slice(lo, hi)
    n = hi - lo

    # heading (body -X) in world, per frame
    fwd = np.stack([rot(root[t, 3:7]) @ np.array([-1., 0, 0]) for t in range(T)])
    v_fwd = np.sum(fwd * vel, axis=1)
    v_lat = vel - v_fwd[:, None] * fwd
    lat_speed = np.linalg.norm(v_lat, axis=1)
    slip_ratio = lat_speed / np.maximum(speed, 1e-6)
    cosang = np.clip(v_fwd / np.maximum(speed, 1e-6), -1, 1)
    aoa = np.degrees(np.arccos(cosang))
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    yaw_rate = np.gradient(yaw, dt)

    # tail-beat + oscillation amplitude from the bend feature (same extractor as the env/reference).
    # Linear-detrend (removes DC + slow drift) then pick the dominant peak in a PHYSIOLOGICAL band
    # [0.15, 8] Hz -- this survives a slow (bad) gait AND a fast one, and rejects both the maneuvering
    # (<0.15 Hz) and the frame-to-frame jitter that piles up near Nyquist.
    feat = bend_features(pc, root, args.K)
    tail = feat[S, -1]
    tt = np.arange(len(tail))
    tail_dt = tail - np.polyval(np.polyfit(tt, tail, 1), tt)
    F = np.abs(np.fft.rfft((tail_dt - tail_dt.mean()) * np.hanning(len(tail_dt)))) ** 2
    freqs = np.fft.rfftfreq(len(tail_dt), dt)
    band = (freqs >= 0.15) & (freqs <= 8.0)
    tail_beat = freqs[band][F[band].argmax()] if band.any() else float("nan")
    osc_amp = feat[S, -1].std()                 # BL (head-anchored, so this is tail swing)

    # AMP score from the trained disc
    if Path(args.disc).exists():
        _, r_style = amp_score(feat, args.disc, args.K)
        rs = r_style[lo:min(hi, len(r_style))]
        amp_mean, amp_std = float(rs.mean()), float(rs.std())
    else:
        amp_mean = amp_std = float("nan")

    # energy per distance (cost of transport proxy): integral |tau.omega| dt / path length
    dist = np.linalg.norm(np.diff(pos[S], axis=0), axis=1).sum()
    if tau is not None and jvel is not None:
        power = (np.abs(tau) * np.abs(jvel)).sum(1)
        energy = power[S].sum() * dt
        cot = energy / max(dist, 1e-6)
    else:
        cot = float("nan")

    # blow-ups: joint-vel spikes past the limit
    blowups = int((np.abs(jvel).max(1) > 190).sum()) if jvel is not None else 0

    net_yaw = np.degrees(yaw[hi - 1] - yaw[lo])

    print(f"\n===== AMP swim metrics: {args.label} =====")
    print(f"  rollout {T} frames, steady window [{lo},{hi}) = {n*dt:.1f}s (launch excluded from {hi})")
    print(f"  forward speed       {np.median(v_fwd[S]):.4f} m/s = {np.median(v_fwd[S])/BL:.3f} BL/s")
    print(f"  total speed         {np.median(speed[S]):.4f} m/s = {np.median(speed[S])/BL:.3f} BL/s")
    print(f"  lateral-speed ratio {np.median(slip_ratio[S]):.3f}  (|v_lat|/|v|; 0=pure forward)")
    print(f"  angle of attack     {np.median(aoa[S]):.1f} deg  (median; 0=body aligned with travel)")
    print(f"  yaw drift           {net_yaw:+.0f} deg net, |rate| med {np.median(np.abs(yaw_rate[S])):.2f} rad/s")
    print(f"  tail-beat frequency {tail_beat:.2f} Hz")
    print(f"  oscillation amp     {osc_amp:.4f} BL (tail)")
    print(f"  AMP score r_style   {amp_mean:.3f} +/- {amp_std:.3f}  (trained disc; 0.75=GAN equilibrium)")
    print(f"  energy per distance {cot:.3f} J/m  (cost of transport proxy)")
    print(f"  blow-ups            {blowups}")
    # machine-readable line for the comparison table
    print(f"METRICS_JSON {args.label} " + str({
        "v_fwd_bl": round(float(np.median(v_fwd[S]) / BL), 3),
        "slip_ratio": round(float(np.median(slip_ratio[S])), 3),
        "aoa_deg": round(float(np.median(aoa[S])), 1),
        "net_yaw_deg": round(float(net_yaw), 0),
        "yaw_rate": round(float(np.median(np.abs(yaw_rate[S]))), 2),
        "tail_beat_hz": round(float(tail_beat), 2),
        "osc_amp_bl": round(float(osc_amp), 4),
        "amp_score": round(amp_mean, 3), "amp_score_std": round(amp_std, 3),
        "cot": round(float(cot), 3) if cot == cot else None,
        "blowups": blowups}))


if __name__ == "__main__":
    main()
