#!/usr/bin/env python
"""Segment the single swimming fish out of a 3D-ZeF sequence and export a
black-background, fish-centered video + image frames for BOTH annotated views
(imgF = front cam, imgT = top cam).

Pipeline (per view):
  1. Median background over sampled frames (static camera -> fish disappears).
  2. Per frame: abs-diff vs background -> threshold -> morphology -> connected
     components. Pick the fish blob = largest component whose centroid is near
     the ground-truth head position (from gt/gt.txt), so reflections / feeder
     noise are rejected.
  3. Fill holes, smooth -> binary fish mask -> composite fish onto black.
  4. Crop a fixed square window centered on the fish each frame (consistent
     scale, fish fills the frame) and resize to a fixed canvas.

Memory-frugal: computes the background strip-wise over a few sampled frames and
processes one frame at a time (never holds all masks in RAM), so it runs in a
~2 GB budget.

Outputs (under outputs/zef_fish_seg/<SEQ>/):
  <view>/frames/000001.png ...   black-bg fish frames (cropped, centered)
  <view>_fish.mp4                 the same as a video
  <view>/mask/000001.png ...     (optional) full-frame masks for debugging

Usage:
  python scripts/segment_zef_fish.py --seq ZebraFish-05 --save-mask
"""
import argparse, gc, glob, os
import cv2
import numpy as np

# gt.txt column indices (0-based) per view: head x/y and bbox l/t/w/h
VIEW_COLS = {
    "imgF": dict(cx=12, cy=13, bl=14, bt=15, bw=16, bh=17),  # front cam
    "imgT": dict(cx=5,  cy=6,  bl=7,  bt=8,  bw=9,  bh=10),  # top cam
}


def median_background(files, sample=30, strips=30):
    """Memory-frugal median background: sample frames, median per horizontal
    strip so the float64 temporary stays small (fits a ~2 GB budget)."""
    sel = files[:: max(1, len(files) // sample)]
    imgs = [cv2.imread(f) for f in sel]
    h, w, _ = imgs[0].shape
    stack = np.stack(imgs)  # (N,h,w,3) uint8
    del imgs
    bg = np.empty((h, w, 3), np.uint8)
    step = (h + strips - 1) // strips
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        bg[y0:y1] = np.median(stack[:, y0:y1], axis=0).astype(np.uint8)
    del stack
    gc.collect()
    return bg


def segment(img, bg, cx, cy, roi_half=250, thresh=None, min_area=60):
    """Return a full-frame binary mask of the single fish, segmented inside a
    ROI centered on the gt head (cx, cy).

    ROI-local Otsu adapts to the fish/tank contrast and captures the whole body
    (the earlier global threshold + open-erosion lost most of the fish). The ROI
    also rejects far-away reflections / feeder noise robustly.
    """
    H, W = img.shape[:2]
    x0, y0 = max(0, int(cx) - roi_half), max(0, int(cy) - roi_half)
    x1, y1 = min(W, int(cx) + roi_half), min(H, int(cy) + roi_half)
    d = cv2.absdiff(img[y0:y1, x0:x1], bg[y0:y1, x0:x1])
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    if thresh is None:
        t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # guard: if Otsu picks an absurdly low cut on a near-empty ROI, floor it
        if t < 8:
            _, m = cv2.threshold(g, 8, 255, cv2.THRESH_BINARY)
    else:
        _, m = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    hx, hy = cx - x0, cy - y0  # gt head in ROI coords
    best, best_score = -1, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        dist = np.hypot(cent[i, 0] - hx, cent[i, 1] - hy)
        # prefer large blobs near the gt head
        score = area / (1.0 + dist)
        if score > best_score:
            best_score, best = score, i
    roi_mask = np.zeros(m.shape, np.uint8)
    if best > 0:
        roi_mask[lab == best] = 255
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    # fill interior holes via flood fill from a corner
    ff = roi_mask.copy()
    rh, rw = roi_mask.shape
    ffmask = np.zeros((rh + 2, rw + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)
    roi_mask = roi_mask | cv2.bitwise_not(ff)
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = roi_mask
    return mask


def mask_center_extent(mask, fallback):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return fallback, 0
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    extent = max(xs.max() - xs.min(), ys.max() - ys.min())
    return (cx, cy), int(extent)


def crop_centered(img, cx, cy, win, canvas):
    """Crop a square window of side `win` centered on (cx,cy) with black padding,
    then resize to canvas x canvas."""
    half = win // 2
    x0, y0 = int(cx) - half, int(cy) - half
    out = np.zeros((win, win, 3), img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.shape[1], x0 + win), min(img.shape[0], y0 + win)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (canvas, canvas), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/work/yzha1/SimFishLib/Dataset/3D-ZeF/data")
    ap.add_argument("--seq", default="ZebraFish-05")
    ap.add_argument("--out", default="/work/yzha1/SimFishLib/outputs/zef_fish_seg")
    ap.add_argument("--views", nargs="+", default=["imgF", "imgT"])
    ap.add_argument("--canvas", type=int, default=768)
    ap.add_argument("--margin", type=float, default=1.3, help="crop window = fish_extent * margin (fish fills ~1/margin of the canvas)")
    ap.add_argument("--probe-step", type=int, default=8, help="frame stride for window estimation")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--thresh", type=int, default=None, help="fixed diff threshold; default None = ROI-local Otsu")
    ap.add_argument("--save-mask", action="store_true")
    args = ap.parse_args()

    seq_dir = os.path.join(args.root, args.seq)
    gt = np.loadtxt(os.path.join(seq_dir, "gt", "gt.txt"), delimiter=",")
    ff = os.environ.get("FFMPEG", "")

    def gt_head(fr, cols):
        row = gt[gt[:, 0] == fr]
        return None if len(row) == 0 else (row[0][cols["cx"]], row[0][cols["cy"]])

    for view in args.views:
        cols = VIEW_COLS[view]
        files = sorted(glob.glob(os.path.join(seq_dir, view, "*.jpg")))
        print(f"[{view}] {len(files)} frames -> building median background...", flush=True)
        bg = median_background(files)

        out_view = os.path.join(args.out, args.seq, view)
        frames_dir = os.path.join(out_view, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        if args.save_mask:
            os.makedirs(os.path.join(out_view, "mask"), exist_ok=True)

        # ---- Probe pass: measure fish-extent distribution ----
        extents = []
        for f in files[:: args.probe_step]:
            fr = int(os.path.splitext(os.path.basename(f))[0])
            head = gt_head(fr, cols)
            if head is None:
                continue
            img = cv2.imread(f)
            mask = segment(img, bg, head[0], head[1], thresh=args.thresh)
            _, ext = mask_center_extent(mask, head)
            if ext:
                extents.append(ext)
        extents = np.array(extents) if extents else np.array([200])
        # per-frame crop window = this frame's fish extent * margin, but not
        # smaller than floor_win (so head-on/compact poses aren't magnified into
        # a pixelated mess) -> fish always fills the majority of the canvas.
        floor_win = int(np.percentile(extents, 40) * args.margin)
        floor_win = max(96, floor_win + (floor_win % 2))
        print(f"[{view}] probed {len(extents)} frames, extent p50={np.percentile(extents,50):.0f} "
              f"p99={np.percentile(extents,99):.0f}px -> floor window {floor_win}px, margin {args.margin}", flush=True)

        # ---- Main pass: segment, composite on black, centered crop, write ----
        vid_path = os.path.join(args.out, args.seq, f"{view}_fish.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (args.canvas, args.canvas))
        n_hit = 0
        for k, f in enumerate(files):
            fr = int(os.path.splitext(os.path.basename(f))[0])
            head = gt_head(fr, cols)
            img = cv2.imread(f)
            if head is None:
                out = np.zeros((args.canvas, args.canvas, 3), np.uint8)
                mask = None
            else:
                mask = segment(img, bg, head[0], head[1], thresh=args.thresh)
                cen, ext = mask_center_extent(mask, head)
                fish = cv2.bitwise_and(img, img, mask=mask)
                win = max(floor_win, int(ext * args.margin))
                win += win % 2
                out = crop_centered(fish, cen[0], cen[1], win, args.canvas)
                if ext:
                    n_hit += 1
            cv2.imwrite(os.path.join(frames_dir, f"{fr:06d}.png"), out)
            if args.save_mask and mask is not None:
                cv2.imwrite(os.path.join(out_view, "mask", f"{fr:06d}.png"), mask)
            writer.write(out)
            if (k + 1) % 100 == 0:
                print(f"[{view}] {k+1}/{len(files)} frames (fish found {n_hit})", flush=True)
        writer.release()

        # re-encode to h264 for portability if ffmpeg available
        if ff:
            h264 = vid_path.replace(".mp4", "_h264.mp4")
            rc = os.system(f'{ff} -y -i "{vid_path}" -c:v libx264 -pix_fmt yuv420p -crf 20 "{h264}" >/dev/null 2>&1')
            if rc == 0 and os.path.exists(h264):
                os.replace(h264, vid_path)
        print(f"[{view}] DONE  fish found in {n_hit}/{len(files)} frames\n"
              f"        frames -> {frames_dir}\n        video  -> {vid_path}", flush=True)

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
