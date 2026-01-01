"""Convert rigging trajectories (episodes.jsonl + rendered states) into
ms-swift SFT rows for Qwen3-VL.

Each episode step becomes one training row:
  system: rigging instructions + template menu
  user:   <image> side view + <image> top view + state summary
  assistant: the expert JSON action

Usage:
  python3 build_sft_jsonl.py \
      --trajectories simfishlib_data/vlm_rigger/trajectories \
      --stage0 simfishlib_data/vlm_rigger/stage0 \
      --out simfishlib_data/vlm_rigger/sft/rig_sft.jsonl \
      [--val-samples zander_fish,tilapia_fish]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_loop import SYSTEM_PROMPT, template_menu  # noqa: E402


def state_summary(bones):
    if not bones:
        return "No bones placed yet."
    lines = ["%d bones placed:" % len(bones)]
    for i, b in enumerate(bones):
        lines.append("  bone %d: %s center=[%s] size=[%s]" % (
            i + 1, b["template_id"],
            ",".join("%.3f" % v for v in b["center_norm"]),
            ",".join("%.3f" % v for v in b["size_norm"])))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True)
    p.add_argument("--stage0", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-samples", default="zander_fish,tilapia_fish")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(os.path.join(args.stage0, "bone_template_library.json")) as f:
        library = {t["template_id"]: t for t in json.load(f)["templates"]}
    system = SYSTEM_PROMPT.format(template_menu=template_menu(library))
    val_samples = set(args.val_samples.split(","))

    rows_train, rows_val, skipped = [], [], 0
    with open(os.path.join(args.trajectories, "episodes.jsonl")) as f:
        for line in f:
            ep = json.loads(line)
            if not all(os.path.exists(img) for img in ep["images"]):
                skipped += 1
                continue
            user = ("SIDE view (image 1) and TOP view (image 2) of the fish "
                    "with the current skeleton.\n%s\nYour action:" %
                    state_summary(ep["state_bones"]))
            row = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "<image><image>" + user},
                    {"role": "assistant", "content": json.dumps(ep["target_action"])},
                ],
                "images": [os.path.abspath(i) for i in ep["images"]],
            }
            (rows_val if ep["sample"] in val_samples else rows_train).append(row)

    rng = random.Random(args.seed)
    rng.shuffle(rows_train)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows_train:
            f.write(json.dumps(r) + "\n")
    val_path = args.out.replace(".jsonl", "_val.jsonl")
    with open(val_path, "w") as f:
        for r in rows_val:
            f.write(json.dumps(r) + "\n")
    print("train rows: %d -> %s" % (len(rows_train), args.out))
    print("val rows:   %d -> %s" % (len(rows_val), val_path))
    print("skipped (missing renders): %d" % skipped)


if __name__ == "__main__":
    main()
