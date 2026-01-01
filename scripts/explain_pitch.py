#!/usr/bin/env python
"""A simple picture explaining PITCH (nose up/down tilt) + FORESHORTENING (tilted things look shorter
from above), and why it breaks the body-length normalization -- for a non-native speaker."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Why a tilted (pitched) fish is measured too SHORT from the top camera", fontsize=14)


def draw_fish(ax, angle_deg, label):
    ax.set_xlim(-0.2, 2.6); ax.set_ylim(-0.3, 2.0); ax.set_aspect("equal"); ax.axis("off")
    # the "floor" = the horizontal plane the TOP camera measures on
    ax.plot([-0.1, 2.5], [0, 0], "k-", lw=2)
    ax.text(2.5, -0.18, "horizontal plane\n(what the top camera sees)", fontsize=9, ha="right", va="top", color="0.3")
    # the fish body: a line of true length L, centered at (cx, cy), tilted by angle
    L = 1.6; cx, cy = 0.9, 1.1
    a = np.deg2rad(angle_deg)
    nose = np.array([cx + 0.5 * L * np.cos(a), cy + 0.5 * L * np.sin(a)])
    tail = np.array([cx - 0.5 * L * np.cos(a), cy - 0.5 * L * np.sin(a)])
    ax.plot([tail[0], nose[0]], [tail[1], nose[1]], "-", color="steelblue", lw=9, solid_capstyle="round")
    ax.plot(nose[0], nose[1], "o", color="orange", ms=13)            # nose
    ax.text(nose[0] + 0.05, nose[1] + 0.05, "nose", fontsize=9, color="orange")
    ax.text(tail[0] - 0.05, tail[1] - 0.12, "tail", fontsize=9, color="steelblue", ha="right")
    # drop the two ends straight down to the floor = the "shadow" the top camera measures
    for p in (nose, tail):
        ax.plot([p[0], p[0]], [p[1], 0], "--", color="0.6", lw=1.2)
    sx0, sx1 = tail[0], nose[0]
    ax.plot([sx0, sx1], [0, 0], "-", color="crimson", lw=6, solid_capstyle="butt")
    ax.annotate("", xy=(sx1, -0.12), xytext=(sx0, -0.12),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.5))
    shadow = abs(sx1 - sx0)
    ax.text((sx0 + sx1) / 2, -0.22, f"MEASURED length = {shadow/L:.0%} of the real length",
            fontsize=10, ha="center", va="top", color="crimson", fontweight="bold")
    # true length marker along the body
    mid = (nose + tail) / 2
    ax.text(mid[0] + 0.15 * np.sin(a) + 0.1, mid[1] - 0.15 * np.cos(a),
            f"real body\nlength = 100%", fontsize=9, color="steelblue")
    ax.set_title(label, fontsize=12)


draw_fish(axs[0], 0, "FLAT fish (nose level)  ✓ measured right")
draw_fish(axs[1], 47, "PITCHED fish (nose tilted up 47°)  ✗ measured too short")
axs[1].annotate("", xy=(1.45, 1.55), xytext=(1.0, 1.05),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
axs[1].text(1.5, 1.6, "'pitch' =\nnose tilts\nup or down", fontsize=10, color="green")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("demo_out/explain_pitch.png", dpi=115)
print("saved demo_out/explain_pitch.png")
