#!/usr/bin/env python
"""amp10 (3D AMP) learning curves: each reward term + swimming + discriminator health, over epochs."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wandb
RUN = "ANON-ENTITY/fish_articulation_analytic_water/xb86edde"
# weights (from cfg) so we can show the WEIGHTED contribution of each term
W = dict(r_style=0.5, r_displacement=1.0, jvel_pen=-0.01, pen_actrate_w=-1.0, w_alive=0.05)
keys = ["episode/r_style", "episode/r_displacement", "episode/jvel_pen", "episode/pen_actrate_w",
        "episode/net_disp_bl", "episode/v_forward_bl", "episode/cruise_speed",
        "episode/D_fake_spread", "episode/D_gap", "episode/D_real_mean", "episode/D_fake_mean",
        "episode/blowup_frac", "episode/z_local", "episode/bend_activity"]
h = wandb.Api(timeout=40).run(RUN).history(keys=keys, samples=3000)
ep = np.arange(len(h))
def col(k):
    c = "episode/" + k
    return h[c].values.astype(float) if c in h.columns else np.full(len(h), np.nan)

fig, axs = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("amp10 — 3D AMP retrain (warm-start from swimmer, pen_z removed)   reward-term learning curves", fontsize=13)

ax = axs[0, 0]
ax.plot(ep, W["r_style"] * col("r_style"), label="style (AMP 3D) × 0.5")
ax.plot(ep, W["r_displacement"] * col("r_displacement"), label="net-travel × 1.0")
ax.plot(ep, W["jvel_pen"] * col("jvel_pen"), label="joint-speed penalty × -0.01")
ax.plot(ep, np.full(len(ep), W["w_alive"]), "--", color="0.6", label="alive +0.05")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("(a) WEIGHTED reward terms (what actually drives training)")
ax.set_xlabel("epoch"); ax.set_ylabel("reward contribution"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[0, 1]
ax.plot(ep, col("net_disp_bl"), label="net travel (any dir)")
ax.plot(ep, col("v_forward_bl"), label="forward speed")
ax.plot(ep, col("cruise_speed"), label="cruise speed", alpha=0.6)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("(b) SWIMMING (BL/s) — did it keep swimming?")
ax.set_xlabel("epoch"); ax.set_ylabel("BL/s"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 0]
ax.plot(ep, col("r_style"), "purple", label="r_style (raw AMP score → reward)")
ax.plot(ep, col("D_fake_spread"), "orange", label="D_fake_spread (disc not collapsed)")
ax.plot(ep, col("D_gap"), "green", label="D_gap (real−fake separation)")
ax.set_title("(c) DISCRIMINATOR health (3D, input=88)")
ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axs[1, 1]
ax.plot(ep, col("blowup_frac"), "red", label="blowup fraction")
ax.plot(ep, col("z_local"), "blue", label="depth z (target 1.0, no pen_z!)")
ax.plot(ep, col("bend_activity"), "green", label="bend activity")
ax.axhline(1.0, color="blue", lw=0.5, ls=":")
ax.set_title("(d) STABILITY / depth (pen_z removed — disc holds depth?)")
ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("demo_out/amp10_curves.png", dpi=110)
print(f"[curves] saved demo_out/amp10_curves.png ({len(h)} epochs)")
for k in ["r_style", "r_displacement", "net_disp_bl", "z_local", "blowup_frac", "D_gap"]:
    v = col(k); v = v[np.isfinite(v)]
    if len(v):
        print(f"  {k:<16} {v[0]:+.4f} -> {v[-1]:+.4f}  (min {v.min():+.3f}, max {v.max():+.3f})")
