#!/usr/bin/env python
"""Post-process: crop each already-selected canonical frame down to just the fish
region, using Qwen3-VL to localize a bounding box (no classical CV segmentation
needed -- reuses the same model already used to pick the frame).

For each image, asks Qwen for the fish's bounding box in NORMALIZED 0-1000
coordinates (standard Qwen-VL grounding convention: x1,y1,x2,y2, top-left
origin), scales to pixel coords, pads by --margin, and crops (in place by
default -- pass --out to write elsewhere and keep the originals).

Usage:
  python crop_canonical_to_fish_qwen.py --in_dir ../../raw_datasets/catfish_canonical_frames
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simfishlib.inference.qwen3vl import Qwen3VLClient  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return _expandvars(s)


DEFAULT_MODEL = _P("${FISH_ROOT}/SimFishLib/models/Qwen3-VL-32B-Instruct")

PROMPT = """Find the single fish in this image. Output ONLY its bounding box as
four integers x1,y1,x2,y2 (top-left and bottom-right corners), on a 0-1000
normalized scale where (0,0) is the top-left corner of the image and (1000,1000)
is the bottom-right corner.

The box MUST fully contain the ENTIRE fish: the nose tip, the tail tip, and all
fins, with NONE of them touching or cut off by the box edges. Look carefully
for the tail -- it is often thin, pale, or blends into the background, and is
the single most commonly missed part. If in doubt, make the box LARGER rather
than smaller: it is far better to include some extra tank/water background
than to clip off any part of the fish's body. Do not worry about tightness.

Respond with ONLY the four numbers, comma-separated, nothing else."""

# Used on RETRIES (previous attempt's crop was rejected by the verifier): forces explicit
# nose/tail localization in words BEFORE committing to coordinates -- chain-of-thought grounding
# is measurably more accurate than jumping straight to numbers, and is worth the extra tokens
# only on the harder cases that already failed once.
PROMPT_COT = """Find the single fish in this image.

Step 1: In one short sentence, describe WHERE the fish's nose/head tip is located
in the image (e.g. "near the left edge, partly faded into the background").
Step 2: In one short sentence, describe WHERE the fish's tail tip is located.
The tail is often thin, pale, low-contrast, or partially blended into the
background/substrate -- look very carefully along the body's long axis past
where the body appears to "end" at a glance.
Step 3: Output a bounding box, as four integers x1,y1,x2,y2 (top-left and
bottom-right corners), on a 0-1000 normalized scale (0,0)=top-left,
(1000,1000)=bottom-right, that fully contains BOTH points you just described
plus the whole body between them, with generous margin. It is far better to
include extra background than to clip the nose or tail.

End your response with a final line in EXACTLY this format:
BOX: x1,y1,x2,y2"""

VERIFY_PROMPT = ("Is the ENTIRE fish visible in this image -- nose tip, tail tip, and all fins, with "
                  "NONE of them cut off by the image edges? Answer with exactly one word: YES or NO.")


def parse_box(reply: str) -> tuple[int, int, int, int] | None:
    m = re.search(r"BOX:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", reply, re.IGNORECASE)
    if m:
        nums = m.groups()
    else:
        nums = re.findall(r"\d+", reply)
        if len(nums) < 4:
            return None
        nums = nums[:4]
    x1, y1, x2, y2 = (int(n) for n in nums)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_with_margin(im, box, margin: float):
    W, H = im.size
    x1, y1, x2, y2 = box
    px1, py1 = x1 / 1000 * W, y1 / 1000 * H
    px2, py2 = x2 / 1000 * W, y2 / 1000 * H
    bw, bh = px2 - px1, py2 - py1
    mx, my = bw * margin, bh * margin
    cx1 = max(0, int(px1 - mx))
    cy1 = max(0, int(py1 - my))
    cx2 = min(W, int(px2 + mx))
    cy2 = min(H, int(py2 + my))
    return im.crop((cx1, cy1, cx2, cy2))


def verify_crop(client, cropped) -> bool:
    """True if Qwen confirms nothing is clipped."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
        cropped.save(tf.name, quality=95)
        reply = client.generate_from_images([tf.name], VERIFY_PROMPT, max_new_tokens=8)
    return "NO" not in reply.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", default=None, help="default: crop in place")
    ap.add_argument("--margin", type=float, default=0.35, help="fractional padding around the detected box "
                    "-- kept generous ON PURPOSE: a VLM-grounded box that's too TIGHT risks silently "
                    "clipping part of the fish (nose/tail), which is much worse for 3D reconstruction "
                    "than a bit of extra background; measured failure case: lake_sturgeon_fish020 lost "
                    "its tail at margin=0.12 even though the full body was visible in the source frame")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--max_retries", type=int, default=4,
                     help="attempts per image before giving up and keeping the uncropped original. "
                          "Attempt 0 = deterministic PROMPT (temperature=0). Attempts 1+ = chain-of-"
                          "thought PROMPT_COT with sampling (temperature>0, a fresh draw each time) "
                          "and a progressively larger margin, so a retry can recover from a box that "
                          "was simply mis-localized, not just one that was merely too tight.")
    args = ap.parse_args()

    from PIL import Image

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    assert images, f"no images found in {in_dir}"
    print(f"[crop_canonical] {len(images)} images in {in_dir}", flush=True)

    print(f"[crop_canonical] loading Qwen3-VL from {args.model_path} ...", flush=True)
    client = Qwen3VLClient(model_path=args.model_path)
    client.load()
    print("[crop_canonical] model loaded", flush=True)

    n_ok, n_fallback = 0, 0
    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        out_path = out_dir / img_path.name
        success = False
        for attempt in range(args.max_retries):
            if attempt == 0:
                prompt, temperature, margin = PROMPT, 0.0, args.margin
            else:
                prompt, temperature, margin = PROMPT_COT, 0.9, min(0.35 + 0.2 * attempt, 1.0)
            reply = client.generate_from_images([img_path], prompt, max_new_tokens=160, temperature=temperature)
            box = parse_box(reply)
            if box is None:
                print(f"[crop_canonical] {img_path.name} attempt {attempt}: could not parse box from "
                      f"reply={reply.strip()!r}", flush=True)
                continue
            cropped = crop_with_margin(im, box, margin)
            ok = verify_crop(client, cropped)
            print(f"[crop_canonical] {img_path.name} attempt {attempt}: box={box}/1000 margin={margin:.2f} "
                  f"-> crop {cropped.size} verify={'YES' if ok else 'NO'}", flush=True)
            if ok:
                cropped.save(out_path, quality=95)
                success = True
                break
        if success:
            n_ok += 1
        else:
            im.save(out_path, quality=95)
            n_fallback += 1
            print(f"[crop_canonical] {img_path.name}: all {args.max_retries} attempts rejected "
                  f"-> keeping UNCROPPED original -> {out_path}", flush=True)

    print(f"[crop_canonical] done: {n_ok} cropped, {n_fallback} kept uncropped (all retries failed)", flush=True)


if __name__ == "__main__":
    main()
