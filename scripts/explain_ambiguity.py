#!/usr/bin/env python
"""Show the ambiguity: the current bend feature (one left-right number per station) cannot tell apart
very different 3D body shapes. Many poses -> the same feature. You must encode the WHOLE midline."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = np.linspace(0, 1, 20)               # head(0) -> tail(1)
# three DIFFERENT 3D midline shapes, all with the SAME left-right (lateral) profile (here: zero)
up = 0.25 * np.sin(s * np.pi)           # curves UP in the vertical plane
down = -0.25 * np.sin(s * np.pi)        # curves DOWN
tilt = -0.35 * s                        # straight body but tilted (pitched) nose-down
lateral_same = 0.0 * s                  # all three have the SAME lateral profile (straight side-to-side)

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("The current bend feature is AMBIGUOUS: different 3D shapes → the same numbers", fontsize=13)

# LEFT: side view (along x up/down) = what the body REALLY does -- three clearly different shapes
ax = axs[0]
ax.plot(s, up, "-o", color="royalblue", ms=4, lw=2.2, label="shape A: body curves UP")
ax.plot(s, down, "-o", color="crimson", ms=4, lw=2.2, label="shape B: body curves DOWN")
ax.plot(s, tilt, "-o", color="green", ms=4, lw=2.2, label="shape C: straight but TILTED")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("SIDE view (up / down) — what the fish really does\nTHREE clearly different bodies")
ax.set_xlabel("along body (head → tail)"); ax.set_ylabel("up  /  down"); ax.legend(fontsize=9)
ax.set_ylim(-0.4, 0.4)

# RIGHT: top view (along x left/right) = the CURRENT feature -- all three are IDENTICAL
ax = axs[1]
for c, lab in [("royalblue", "A"), ("crimson", "B"), ("green", "C")]:
    ax.plot(s, lateral_same, "-o", color=c, ms=5, lw=2.2, alpha=0.6)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("TOP view (left / right) — the CURRENT feature\nALL THREE look the SAME (flat) → disc can't tell them apart")
ax.set_xlabel("along body (head → tail)"); ax.set_ylabel("left  /  right")
ax.set_ylim(-0.4, 0.4)
ax.text(0.5, 0.28, "A, B, C all lie on top of each other here", ha="center", fontsize=11,
        color="darkred", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("demo_out/explain_ambiguity.png", dpi=115)
print("saved demo_out/explain_ambiguity.png")
