"""Calibrate the traveling-wave CPG from the SAME ZeF-05 curvature data the PCA basis uses.

The CPG is defined in CURVATURE space so both controllers share the identical downstream
decoder (ridge-pinv(Phi) curvature->joint map):

    kappa_cpg(s,t) = kappa_mean(s) + A(t) * E(s) * sin(theta(t) - phi(s)) + b(t)
    theta'(t)      = 2*pi*f(t)

ZeF-derived FIXED parameters (this script):
    E(s)    amplitude envelope  = per-station Hilbert amplitude of the band-passed kappa,
            normalized to max 1
    phi(s)  spatial phase       = per-station circular-mean phase offset vs the reference
            station (the traveling-wave profile), unwrapped along s
    f0      baseline frequency  = dominant spectral peak of posterior-station kappa
    A_max   amplitude bound     = p99 of the instantaneous oscillation amplitude at the
            envelope-max station (so A*E reproduces the data's amplitude distribution)
    f_lo/hi frequency bounds    = p5/p95 of the instantaneous frequency (clamped to [0.5, 3.5]
            Hz -- the PD actuator corner is 3.2 Hz, unreachable above)
    b_max   turn-bias bound     = p99 of |body-mean kappa| (the data's low-frequency uniform-
            curvature component = how hard the real fish bends to turn)

RL-controlled parameters (3): A(t), f(t), b(t) via bounded-rate deltas.
Output: demo_out/zef_manifold/cpg_params.npz + a diagnostic plot.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = Path(_P("${FISH_ROOT}/demo_out/zef_manifold"))
d = np.load(OUT / "curvature_dataset.npz")
kb = d["kappa_bl"]                  # (900, 20)
valid = d["valid"].astype(bool)
fps = float(d["fps"])
S = kb.shape[1]

# longest contiguous valid run (same clip convention as everything else: 395:900)
seg = kb[395:900]                   # (505, 20), all finite
assert np.isfinite(seg).all()
mean_k = np.nanmean(kb[valid], axis=0)
x = seg - seg.mean(0, keepdims=True)

# ---- baseline frequency: spectral peak of the posterior half (the wave carrier) ----
post = x[:, S // 2:].mean(1)
freqs = np.fft.rfftfreq(len(post), 1.0 / fps)
P = np.abs(np.fft.rfft(post * np.hanning(len(post)))) ** 2
P[freqs < 0.5] = 0
f0 = float(freqs[int(np.argmax(P))])

# ---- band-pass around f0, Hilbert per station ----
lo, hi = max(0.4, 0.5 * f0), min(2.0 * f0, fps / 2 - 1)
b_, a_ = butter(3, [lo / (fps / 2), hi / (fps / 2)], btype="band")
xb = filtfilt(b_, a_, x, axis=0)
an = hilbert(xb, axis=0)
amp = np.abs(an)                    # (T, S) instantaneous amplitude
ph = np.angle(an)                   # (T, S)

E = amp.mean(0)
E = E / E.max()
s_ref = int(np.argmax(amp.mean(0)))              # reference station = strongest oscillation
dph = ph - ph[:, s_ref:s_ref + 1]
phi = np.angle(np.exp(1j * dph).mean(0))         # circular mean phase offset per station
phi = np.unwrap(phi)
phi = phi - phi[0]                               # head station = 0 reference

# ---- data-derived bounds ----
A_max = float(np.percentile(amp[:, int(np.argmax(E))], 99))
inst_f = np.diff(np.unwrap(ph[:, s_ref])) * fps / (2 * np.pi)
f_lo = float(np.clip(np.percentile(inst_f, 5), 0.5, 3.5))
f_hi = float(np.clip(np.percentile(inst_f, 95), 0.5, 3.5))
body_mean = seg.mean(1)                          # uniform-curvature (turning) component
b_max = float(np.percentile(np.abs(body_mean - body_mean.mean()), 99))

# ---- data-derived RATE bounds (p99 of the per-(1/30 s) change, same rule as the PCA da_max) ----
st = int(np.argmax(E))
dA = np.abs(np.diff(amp[::2, st]))               # per 2 frames = 1/30 s
dA_max = float(np.percentile(dA, 99))
df = np.abs(np.diff(inst_f[::2]))
df_max = float(np.clip(np.percentile(df, 99), 0.02, 0.5))
db = np.abs(np.diff(body_mean[::2]))
db_max = float(np.percentile(db, 99))

np.savez(OUT / "cpg_params.npz",
         envelope=E.astype(np.float32), phase=phi.astype(np.float32),
         kappa_mean=mean_k.astype(np.float32),
         f0=f0, f_lo=f_lo, f_hi=f_hi, A_max=A_max, b_max=b_max,
         dA_max=dA_max, df_max=df_max, db_max=db_max,
         s_ref=s_ref, band=(lo, hi), clip=(395, 900))
print(f"rates/step: dA={dA_max:.3f} df={df_max:.3f} db={db_max:.3f}")
print(f"f0={f0:.2f} Hz  f range=[{f_lo:.2f},{f_hi:.2f}]  A_max={A_max:.2f} kappa_bl  "
      f"b_max={b_max:.2f}  s_ref={s_ref}")
print("E(s) =", E.round(2).tolist())
print("phi(s)/pi =", (phi / np.pi).round(2).tolist())

s = np.linspace(0, 1, S)
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].plot(s, E, "o-"); axs[0].set_title("ZeF amplitude envelope E(s)"); axs[0].set_xlabel("s")
axs[1].plot(s, phi / np.pi, "o-"); axs[1].set_title("ZeF spatial phase phi(s) [pi]"); axs[1].set_xlabel("s")
axs[2].semilogy(freqs, P + 1e-12); axs[2].axvline(f0, color="r", ls="--", label=f"f0={f0:.2f} Hz")
axs[2].set_xlim(0, 8); axs[2].legend(); axs[2].set_title("posterior kappa spectrum")
fig.tight_layout(); fig.savefig(OUT / "cpg_calibration.png", dpi=130)
print(f"wrote {OUT/'cpg_params.npz'} and cpg_calibration.png")
