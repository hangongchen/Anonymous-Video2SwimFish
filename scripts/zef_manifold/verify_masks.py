#!/usr/bin/env python
"""Step 0: VERIFY the precomputed ZebraFish-05 top-view masks before trusting them.

Checks, for every one of the 900 frames:
  1. mask exists and has exactly one usable connected component (after size filter)
  2. mask area is stable over time (fish size should not jump frame-to-frame)
  3. the GT head point lies inside (or within a few px of) the mask
  4. mask bbox against the GT bbox (IoU) -- the GT bbox is human-annotated truth
Visual: overlay grid of mask contour + GT head/bbox on the raw frame for sampled
frames (uniform + the WORST frames by each metric), for human inspection.
"""
import argparse, os
import cv2
import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


DATA = _P("${ZEF_ROOT}")
MASKS = _P("${FISH_ROOT}/demo_out/zef05_seg/ZebraFish-05/imgT/mask")
OUT = _P("${FISH_ROOT}/demo_out/zef_manifold")


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = np.loadtxt(f"{DATA}/gt/gt.txt", delimiter=",")
    rows = {int(r[0]): r for r in gt}

    stats = []  # frame, area, n_comp, head_dist_px, bbox_iou
    for fr in sorted(rows):
        r = rows[fr]
        m = cv2.imread(f"{MASKS}/{fr:06d}.png", cv2.IMREAD_GRAYSCALE)
        if m is None:
            stats.append((fr, 0, 0, np.inf, 0.0))
            continue
        mb = (m > 127).astype(np.uint8)
        n, lab, st, cent = cv2.connectedComponentsWithStats(mb)
        comps = [i for i in range(1, n) if st[i, 4] >= 30]
        area = int(mb.sum())
        # distance of GT head to nearest mask pixel
        hx, hy = r[5], r[6]
        ys, xs = np.nonzero(mb)
        hd = float(np.min(np.hypot(xs - hx, ys - hy))) if len(xs) else np.inf
        # mask bbox vs GT bbox IoU
        if len(xs):
            mx0, mx1, my0, my1 = xs.min(), xs.max(), ys.min(), ys.max()
            gx0, gy0, gw, gh = r[7], r[8], r[9], r[10]
            gx1, gy1 = gx0 + gw, gy0 + gh
            ix = max(0, min(mx1, gx1) - max(mx0, gx0))
            iy = max(0, min(my1, gy1) - max(my0, gy0))
            inter = ix * iy
            union = (mx1 - mx0) * (my1 - my0) + gw * gh - inter
            iou = float(inter / max(union, 1e-9))
        else:
            iou = 0.0
        stats.append((fr, area, len(comps), hd, iou))

    S = np.array(stats)
    fr_a, area, ncomp, hdist, iou = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4]
    med = np.median(area)
    print(f"frames with mask: {(area > 0).sum()}/900")
    print(f"area: median {med:.0f}px  p5 {np.percentile(area,5):.0f}  p95 {np.percentile(area,95):.0f}"
          f"  outliers(<0.4x or >2.5x median): {int(((area < 0.4*med) | (area > 2.5*med)).sum())}")
    print(f"components(>=30px): 1-comp {(ncomp==1).sum()}  2+ {(ncomp>1).sum()}  0 {(ncomp==0).sum()}")
    print(f"gt-head->mask dist px: median {np.median(hdist):.1f}  p95 {np.percentile(hdist,95):.1f}"
          f"  >30px: {int((hdist>30).sum())}")
    print(f"bbox IoU vs GT: median {np.median(iou):.2f}  p5 {np.percentile(iou,5):.2f}"
          f"  <0.3: {int((iou<0.3).sum())}")
    np.savez(f"{OUT}/mask_qc.npz", frame=fr_a, area=area, ncomp=ncomp, hdist=hdist, iou=iou)

    # overlay grid: 8 uniform + 4 worst-by-IoU frames
    worst = fr_a[np.argsort(iou)[:4]].astype(int)
    sample = sorted(set(np.linspace(1, 900, 8, dtype=int)) | set(worst))
    tiles = []
    for fr in sample:
        r = rows[fr]
        img = cv2.imread(f"{DATA}/imgT/{fr:06d}.jpg")
        m = cv2.imread(f"{MASKS}/{fr:06d}.png", cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        cont, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cont, -1, (0, 0, 255), 2)
        gx, gy, gw, gh = (int(v) for v in r[7:11])
        cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
        cv2.circle(img, (int(r[5]), int(r[6])), 6, (255, 0, 0), -1)
        cx, cy = int(r[5]), int(r[6])
        h = 180
        x0, y0 = max(0, cx - h), max(0, cy - h)
        crop = img[y0:y0 + 2 * h, x0:x0 + 2 * h]
        crop = cv2.copyMakeBorder(crop, 0, 2 * h - crop.shape[0], 0, 2 * h - crop.shape[1],
                                  cv2.BORDER_CONSTANT)
        cv2.putText(crop, f"f{fr} iou={iou[fr_a==fr][0]:.2f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    rows_n = (len(tiles) + 3) // 4
    grid = np.zeros((rows_n * 2 * 180, 4 * 2 * 180, 3), np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, 4)
        grid[rr * 360:(rr + 1) * 360, cc * 360:(cc + 1) * 360] = t
    cv2.imwrite(f"{OUT}/mask_qc_overlay.png", grid)
    print(f"overlay grid -> {OUT}/mask_qc_overlay.png")


if __name__ == "__main__":
    main()
