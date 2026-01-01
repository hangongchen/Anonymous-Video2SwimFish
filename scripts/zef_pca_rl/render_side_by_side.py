"""Side-by-side video of two reach-eval recordings (same seed => same spawn + first target).

Usage: <env python> scripts/zef_pca_rl/render_side_by_side.py <dirA> <dirB> <out.mp4>
Each dir must contain recording.npz from eval_pca_reach.py. Renders env0 of both on a shared
clock: FEM cloud, nose arrow, current target + success ring, path, reach rings, live counters.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def main():
    da, db, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    A = dict(np.load(da / "recording.npz"))
    B = dict(np.load(db / "recording.npz"))
    dt = float(A["dt"])
    T = min(A["root"].shape[0], B["root"].shape[0])
    t0a, t0b = A["tgt"][0, 0], B["tgt"][0, 0]
    print(f"first target A={t0a.round(3)} B={t0b.round(3)} "
          f"match={np.allclose(t0a, t0b, atol=1e-3)}")

    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    arts = []
    for ax, R, name in ((axs[0], A, da.name), (axs[1], B, db.name)):
        sc = ax.scatter(R["pts0"][0][:, 0], R["pts0"][0][:, 1], s=4, c="#4a90c4", alpha=0.8)
        (tr,) = ax.plot([], [], "-", color="gray", lw=1)
        (st,) = ax.plot([], [], "*", ms=18, color="red")
        ci = plt.Circle((0, 0), 0.2, fill=False, color="red", ls="--"); ax.add_patch(ci)
        ar = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="black", lw=2))
        (rp,) = ax.plot([], [], "o", ms=9, mfc="none", mec="green", mew=2)
        tx = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_aspect("equal"); ax.set_title(name)
        arts.append((ax, R, sc, tr, st, ci, ar, rp, tx, [], []))
    w = FFMpegWriter(fps=30, bitrate=3500)
    with w.saving(fig, out, dpi=100):
        for t in range(T):
            for ax, R, sc, tr, st, ci, ar, rp, tx, rx, ry in arts:
                sc.set_offsets(R["pts0"][t][:, :2])
                tr.set_data(R["root"][: t + 1, 0, 0], R["root"][: t + 1, 0, 1])
                tg = R["tgt"][t, 0]
                st.set_data([tg[0]], [tg[1]]); ci.center = (tg[0], tg[1])
                p, q = R["root"][t, 0], R["quat"][t, 0]
                nose = (1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]))
                ar.xy = (p[0] + 0.25 * nose[0], p[1] + 0.25 * nose[1]); ar.set_position((p[0], p[1]))
                if R["reach"][t, 0]:
                    rx.append(p[0]); ry.append(p[1]); rp.set_data(rx, ry)
                ax.set_xlim(p[0] - 1.4, p[0] + 1.4); ax.set_ylim(p[1] - 1.4, p[1] + 1.4)
                tx.set_text(f"t={t*dt:5.1f}s dist={R['dist'][t,0]:.2f} "
                            f"hcos={R['hcos'][t,0]:+.2f} reaches={int(R['nreach'][t,0])}")
            w.grab_frame()
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
