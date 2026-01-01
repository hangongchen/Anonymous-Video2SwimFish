#!/usr/bin/env python
"""Pull amp09 (warm-start refine) training curves from wandb and plot the story:
swimming is PRESERVED (net_disp / v_forward hold) while AMP refines style (r_style, D_fake_spread)."""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="ANON-ENTITY/fish_articulation_analytic_water/1nb892pl")
ap.add_argument("--out", default="demo_out/amp09_curves.png"); args = ap.parse_args()
import wandb
r = wandb.Api(timeout=40).run(args.run)
ks = ["episode/net_disp_bl", "episode/v_forward_bl", "episode/r_style", "episode/D_fake_spread",
      "episode/cruise_speed", "episode/blowup_frac", "episode/bend_activity"]
h = r.history(keys=ks, samples=2000)
ep = np.arange(len(h))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("amp09 — warm-start from July-18 swimmer + AMP style refine (obs=75, w_disp=1.0, w_style=0.5)", fontsize=12)
def col(k):
    v = h["episode/"+k].values.astype(float) if "episode/"+k in h.columns else np.full(len(h), np.nan)
    return v
ax = axs[0]
ax.plot(ep, col("net_disp_bl"), label="net_disp (any-dir)"); ax.plot(ep, col("v_forward_bl"), label="v_forward")
ax.plot(ep, col("cruise_speed"), label="cruise_speed", alpha=0.6)
ax.axhline(0, color="k", lw=0.5, ls=":"); ax.set_title("SWIMMING preserved (BL/s)"); ax.set_xlabel("refine epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axs[1]
ax.plot(ep, col("r_style"), "purple", label="r_style (AMP)"); ax.plot(ep, col("D_fake_spread"), "orange", label="D_fake_spread")
ax.set_title("AMP style refinement"); ax.set_xlabel("refine epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axs[2]
ax.plot(ep, col("bend_activity"), "green", label="bend_activity"); ax.plot(ep, col("blowup_frac"), "red", label="blowup_frac")
ax.set_title("undulation activity / stability"); ax.set_xlabel("refine epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
import os; os.makedirs(os.path.dirname(args.out), exist_ok=True); plt.savefig(args.out, dpi=110)
print(f"[curves] saved {args.out}  ({len(h)} epochs)")
for k in ["net_disp_bl","v_forward_bl","r_style","blowup_frac"]:
    v = col(k); v = v[np.isfinite(v)]
    if len(v): print(f"  {k:<16} first {v[0]:+.4f} -> last {v[-1]:+.4f}  (max {v.max():+.4f})")
