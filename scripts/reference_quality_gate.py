#!/usr/bin/env python
"""Reference-quality GATE for AMP motion data.

The ZebraFish-01 reference passed silently into training as a discriminator target while being
frame-to-frame WHITE NOISE (2 fish merged by occlusion corrupted the carved midline). The
discriminator then separated it from *itself* at AUC 0.99 and the style reward pinned flat.

This gate runs the checks that would have caught that, on a carved cloud sequence, and returns a
hard PASS/FAIL BEFORE the sequence is allowed to become an AMP reference. Run it on every new carve.

Checks (all vs. what a real coherent undulation must look like):
  1. TEMPORAL COHERENCE  -- lag-1 autocorrelation of the bend profile > 0.6
                            (white noise ~0; a clean 2-3 Hz gait @30fps ~0.8)
  2. SPECTRAL PEAKEDNESS  -- tail-station peak/mean power > 4  AND  spectral flatness < 0.4
                            (white noise flatness ~0.55)
  3. AMPLITUDE ENVELOPE   -- tail/head oscillation-amplitude ratio > 2.5
                            (real fish ~4 per Di Santo 2021; jitter ~1 = uniform)
  4. ENVELOPE SHAPE       -- amplitude minimum in the anterior third (s < 0.4), not mid/posterior

Usage:
  python scripts/reference_quality_gate.py --clouds demo_out/zef05_amp_ref/pointclouds --fps 30
Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_amp_features import spine_bend_profile


def bend_series(cloud_files, K, canonical=True):
    """Per-frame spine-bend profile. Clouds are assumed already in canonical body frame
    (X=length, Y=lateral) as produced by the carve; if not, pass --pca to PCA-align."""
    profs = []
    for f in cloud_files:
        a = np.load(f).astype(np.float64)
        if not canonical:
            c = a - a.mean(0)
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            a = c @ Vt.T
        profs.append(spine_bend_profile(a, K))
    return np.stack(profs)


def _coherence(block, K):
    """lag-1 autocorr, averaged over stations, on ONE contiguous block (sign-aligned)."""
    Pc = block.copy()
    for t in range(1, len(Pc)):
        if np.dot(Pc[t], Pc[t - 1]) < 0:
            Pc[t] = -Pc[t]
    ac = []
    for k in range(K):
        x = Pc[:, k] - Pc[:, k].mean()
        if x.std() > 1e-9:
            ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return float(np.mean(ac)), Pc


def _spectrum(block, fps):
    sig = block[:, -1] - block[:, -1].mean()
    # High-pass: a multi-second clip superimposes SLOW maneuvering (turns, coasting) on the fast
    # tail beat. Without detrending the peak lands on the maneuver (~0.5 Hz), not the gait. A ~0.5 s
    # moving-average removed (cutoff ~2 Hz) isolates the tail beat.
    w = max(3, int(round(fps * 0.5)) | 1)
    sig = sig - np.convolve(sig, np.ones(w) / w, mode="same")
    n = len(sig)
    F = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    F[0] = 0.0
    freqs = np.fft.rfftfreq(n, 1 / fps)
    pos = F[1:]
    gm = np.exp(np.mean(np.log(pos + 1e-20)))
    return float(freqs[F.argmax()]), float(F.max() / pos.mean()), float(gm / pos.mean())


def gate(P, fps, K, blocks=None):
    """blocks: list of (start,end) contiguous index ranges. Coherence/spectrum are computed PER
    block and the MEDIAN is taken -- concatenating disjoint bouts fabricates cross-boundary
    structure that inflates autocorr (this is exactly how the ZebraFish-01 carve fooled a naive
    whole-sequence check). Envelope is a per-frame statistic and is safe to pool."""
    out = {}
    if not blocks:
        blocks = [(0, len(P))]
    acs, pk_hz, pk_om, flat = [], [], [], []
    for a, b in blocks:
        if b - a < 16:
            continue
        ac, Pc = _coherence(P[a:b], K)
        acs.append(ac)
        h, om, fl = _spectrum(Pc, fps)
        pk_hz.append(h); pk_om.append(om); flat.append(fl)
    out["n_blocks"] = len(acs)
    out["autocorr"] = float(np.median(acs))
    out["peak_hz"] = float(np.median(pk_hz))
    out["peak_over_mean"] = float(np.median(pk_om))
    out["flatness"] = float(np.median(flat))

    # 3./4. amplitude envelope (std per station = oscillation amplitude)
    amp = P.std(0)
    out["tail_head_ratio"] = float(amp[-1] / max(amp[0], 1e-9))
    s = np.linspace(0, 1, K)
    out["min_station"] = float(s[amp.argmin()])

    checks = {
        "temporal_coherence (autocorr>0.6)": out["autocorr"] > 0.6,
        "spectral_peak (pk/mean>4)": out["peak_over_mean"] > 4.0,
        "not_white (flatness<0.4)": out["flatness"] < 0.4,
        "envelope_rises (tail/head>2.5)": out["tail_head_ratio"] > 2.5,
        "anterior_node (min_station<0.4)": out["min_station"] < 0.4,
    }
    return out, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", help="dir of per-frame .npy clouds")
    ap.add_argument("--profiles", help="precomputed (T,K) bend-profile .npy (skips carving)")
    ap.add_argument("--glob", default="*.npy")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--pca", action="store_true", help="PCA-align clouds (if not already canonical)")
    args = ap.parse_args()

    if args.profiles:
        P = np.load(args.profiles)
        if P.ndim != 2:
            print(f"FAIL: profiles array must be (T,K), got {P.shape}")
            sys.exit(1)
        blocks = [(0, len(P))]
        nfr = len(P)
    else:
        files = sorted(glob.glob(str(Path(args.clouds) / args.glob)))
        files = [f for f in files if "carved_frames" not in f]
        if len(files) < 20:
            print(f"FAIL: only {len(files)} clouds (<20) -- not enough to assess a gait")
            sys.exit(1)

        # contiguous blocks from filename indices (gaps>1 in the trailing number start a new block)
        def fnum(f):
            digs = "".join(ch for ch in Path(f).stem if ch.isdigit())
            return int(digs) if digs else -1
        nums = [fnum(f) for f in files]
        blocks, start = [], 0
        for i in range(1, len(nums)):
            if nums[i] < 0 or nums[i - 1] < 0 or (nums[i] - nums[i - 1]) not in (1, 2):
                blocks.append((start, i)); start = i
        blocks.append((start, len(files)))
        P = bend_series(files, args.K, canonical=not args.pca)
        nfr = len(files)
    K = P.shape[1]
    out, checks = gate(P, args.fps, K, blocks=blocks)

    print(f"=== reference-quality gate: {nfr} frames @ {args.fps} fps, "
          f"{out['n_blocks']} block(s) ===")
    print(f"  autocorr(lag1)      {out['autocorr']:+.3f}   (>0.6 real, ~0 noise)")
    print(f"  tail beat           {out['peak_hz']:.2f} Hz, peak/mean {out['peak_over_mean']:.1f}   (>4)")
    print(f"  spectral flatness   {out['flatness']:.3f}   (<0.4 tonal, ~0.55 white)")
    print(f"  envelope tail/head  {out['tail_head_ratio']:.2f}   (>2.5; real ~4)")
    print(f"  amplitude min at    s={out['min_station']:.2f}   (<0.4 = anterior node)")
    print("  ---")
    allpass = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allpass = allpass and ok
    print("  ===")
    print(f"GATE: {'PASS -- usable as an AMP reference' if allpass else 'FAIL -- do NOT train on this'}")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
