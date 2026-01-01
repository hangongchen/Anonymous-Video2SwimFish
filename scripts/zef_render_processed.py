#!/usr/bin/env python
"""Render the processed ZeF real-fish frames into a video: both views (top + front), with the extracted
midline, a FORWARD ARROW (length = speed), and a TURNING arrow (where the heading is turning toward).

--frame N  -> save one test frame to demo_out/zef_proc_test.png (to check before rendering the video).
otherwise  -> render frames [--a, --b] into demo_out/zef_processed.mp4.
"""
import argparse, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


ROOT = _P("${ZEF_ROOT}")
SEG = _P("${FISH_ROOT}/demo_out/zef05_seg/ZebraFish-05")
BL_CM = 3.4
FPS = 60.0
CW, CH = 520, 420           # fixed crop size (px), centered on the head

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=-1)
ap.add_argument("--a", type=int, default=30)
ap.add_argument("--b", type=int, default=450)
ap.add_argument("--out", default="demo_out/zef_processed.mp4")
args = ap.parse_args()

gt = np.loadtxt(f"{ROOT}/gt/gt.txt", delimiter=",")
gt = gt[np.argsort(gt[:, 0])]
FR = gt[:, 0].astype(int)
row = {int(r[0]): r for r in gt}


def smooth(a, w=5):
    k = np.ones(w) / w
    return np.stack([np.convolve(a[:, i], k, mode="same") for i in range(a.shape[1])], 1)


# speed + yaw from the 3D head trajectory (same recipe as build_zef05_reference)
pos = smooth(gt[:, 2:5], 5)
dt = 1.0 / FPS
vel = np.gradient(pos, axis=0) / dt
speed_bl = np.linalg.norm(vel, axis=1) / BL_CM
hd = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
hd_s = np.convolve(hd, np.ones(9) / 9, mode="same")       # smooth the HEADING (angle), not the rate
yaw = np.gradient(hd_s) / dt
yaw = np.convolve(yaw, np.ones(5) / 5, mode="same")
yaw[speed_bl < 0.3] = 0.0                                 # heading is undefined when the fish is nearly still
# NO clamp: the real turning rate is shown, including genuine fast turns during darts
# per-view head-pixel velocity (for the on-image arrow DIRECTION); smoothed more so it doesn't hit zero
headT = smooth(gt[:, 5:7], 9); velT = smooth(np.gradient(headT, axis=0), 5)
headF = smooth(gt[:, 12:14], 9); velF = smooth(np.gradient(headF, axis=0), 5)
idx_of = {int(f): i for i, f in enumerate(FR)}
lastdir = {"T": np.array([1.0, 0.0]), "F": np.array([1.0, 0.0])}   # keep last good arrow direction


def clean_mask(mask, head, bbox, margin=0.20):
    m = (mask > 127).astype(np.uint8)
    l, t, w, h = [float(v) for v in bbox]
    mx, my = int(margin * w), int(margin * h)
    keep = np.zeros_like(m)
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


def midline(mask, head, bbox, nb=18):
    m = clean_mask(mask, head, bbox)
    ys, xs = np.nonzero(m)
    if len(xs) < 40:
        return None
    P = np.stack([xs, ys], 1).astype(float)
    ax = P.mean(0) - head
    ax /= np.linalg.norm(ax) + 1e-9
    al = (P - head) @ ax
    e = np.linspace(np.percentile(al, 1), np.percentile(al, 99), nb + 1)
    pts = [P[(al >= e[i]) & (al <= e[i + 1])].mean(0) for i in range(nb) if ((al >= e[i]) & (al <= e[i + 1])).sum() > 3]
    return np.array(pts) if len(pts) > 5 else None


def draw_view(ax, fr, view):
    r = row[fr]; i = idx_of[fr]
    if view == "T":
        img = cv2.imread(f"{ROOT}/imgT/{fr:06d}.jpg"); head = r[5:7]; bbox = r[7:11]; hv = velT[i]; mp = f"{SEG}/imgT/mask/{fr:06d}.png"
    else:
        img = cv2.imread(f"{ROOT}/imgF/{fr:06d}.jpg"); head = r[12:14]; bbox = r[14:18]; hv = velF[i]; mp = f"{SEG}/imgF/mask/{fr:06d}.png"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x0 = int(head[0] - CW / 2); y0 = int(head[1] - CH / 2)
    x0 = max(0, min(x0, img.shape[1] - CW)); y0 = max(0, min(y0, img.shape[0] - CH))
    ax.clear(); ax.imshow(img[y0:y0 + CH, x0:x0 + CW]); ax.set_xticks([]); ax.set_yticks([])
    hx, hy = head[0] - x0, head[1] - y0
    # midline
    mask = cv2.imread(mp, 0)
    if mask is not None:
        mids = midline(mask, head, bbox)
        if mids is not None:
            ax.plot(mids[:, 0] - x0, mids[:, 1] - y0, "-", color="lime", lw=2.2)
    # FORWARD ARROW: direction = head-pixel motion (fall back to last good dir so it never vanishes),
    # length grows with the real 3D speed and has a floor so it is always visible.
    sp = np.linalg.norm(hv)
    if sp > 0.4:
        d = hv / sp; lastdir[view] = d
    else:
        d = lastdir[view]                                           # fish paused / moving toward camera
    # length grows with speed but is CAPPED so the arrow TIP always stays inside the panel (half-height
    # = CH/2 = 210 px). A tip outside the axes makes matplotlib drop the whole arrow -> it vanishes.
    L = 25 + 155 * min(speed_bl[i] / 4.0, 1.0)                       # px, range 25..180 (< 210 half-height)
    ax.annotate("", xy=(hx + d[0] * L, hy + d[1] * L), xytext=(hx, hy),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=3, mutation_scale=22))
    # TURNING ARROW (top view): where the heading points after 0.35 s (rotated by yaw*dt_preview)
    if view == "T":
        a = np.clip(yaw[i] * 0.35, -0.7, 0.7)             # preview rotation, capped so the arrow is sane
        c, s = np.cos(a), np.sin(a)
        d2 = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        ax.annotate("", xy=(hx + d2[0] * L, hy + d2[1] * L), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="-|>", color="orange", lw=2.5, ls="--", mutation_scale=18, alpha=0.9))
    ax.plot(hx, hy, "*", color="yellow", ms=15, mec="k")
    ax.set_title(f"{'TOP (x,y)' if view=='T' else 'FRONT (x,z)'}", fontsize=11)


def render(fr, fig, axs):
    draw_view(axs[0], fr, "T"); draw_view(axs[1], fr, "F")
    i = idx_of[fr]
    turn = "straight" if abs(yaw[i]) < 0.3 else ("turning LEFT" if yaw[i] > 0 else "turning RIGHT")
    fig.suptitle(f"ZeF real fish  frame {fr}    red = speed arrow ({speed_bl[i]:.2f} BL/s)    "
                 f"orange = turning ({turn}, {yaw[i]:+.2f} rad/s)    green = midline", fontsize=12)


fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
if args.frame >= 0:
    render(args.frame, fig, axs)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("demo_out/zef_proc_test.png", dpi=110)
    print("saved demo_out/zef_proc_test.png")
else:
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    w = imageio.get_writer(args.out, fps=30, macro_block_size=None)
    frames = [f for f in range(args.a, args.b + 1) if f in row]
    for j, fr in enumerate(frames):
        render(fr, fig, axs)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        w.append_data(buf)
        if j % 60 == 0:
            print(f"  {j}/{len(frames)} frames", flush=True)
    w.close()
    print(f"saved {args.out}  ({len(frames)} frames)")
