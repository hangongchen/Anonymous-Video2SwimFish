"""Generate expert action trajectories from GT skeleton layouts.

Host-side python (no bpy). For each GT fish:
  - construction episodes: build the skeleton bone-by-bone in several orders;
    each step is (partial state renders) -> next expert action (add/stop)
  - correction episodes: perturb one GT bone; target action = adjust back to GT
  - stop supervision comes from the final construction step

Outputs (under --out-dir):
  episodes.jsonl              one line per training step
  render_specs/<sample>.json  batch spec for render_overlay.py (one per fish)
  states/<sample>/<step>/     render output dirs (filled by render_overlay)

Usage:
  python3 gen_trajectories.py --stage0 simfishlib_data/vlm_rigger/stage0 \
      --gt-dir fish_asset_pipeline/dataset/dataset/supervised_GT_dataset/fish_bone \
      --out-dir simfishlib_data/vlm_rigger/trajectories \
      [--orders 3] [--perturbs-per-bone 3] [--samples name1,name2]
"""

import argparse
import json
import os
import random


def rnd3(x):
    return [round(v, 3) for v in x]


def add_action(bone):
    return {
        "action": "add",
        "template": bone["template_id"],
        "center_norm": rnd3(bone["center_norm"]),
        "size_norm": rnd3(bone["size_norm"]),
    }


def adjust_action(idx, bone):
    return {
        "action": "adjust",
        "bone": idx + 1,  # 1-based, matches render labels
        "center_norm": rnd3(bone["center_norm"]),
        "size_norm": rnd3(bone["size_norm"]),
    }


def perturb(bone, rng):
    b = json.loads(json.dumps(bone))
    kind = rng.choice(["shift", "scale", "both"])
    if kind in ("shift", "both"):
        axis_weights = [0.6, 0.3, 0.1]  # mostly along length, some height
        for i in range(3):
            if rng.random() < axis_weights[i]:
                b["center_norm"][i] += rng.uniform(0.04, 0.14) * rng.choice([-1, 1])
        b["center_norm"] = [min(max(v, 0.02), 0.98) for v in b["center_norm"]]
    if kind in ("scale", "both"):
        f = rng.choice([rng.uniform(0.5, 0.75), rng.uniform(1.3, 1.9)])
        b["size_norm"] = [min(v * f, 1.5) for v in b["size_norm"]]
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage0", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--orders", type=int, default=3,
                   help="construction orderings per fish (1st is always tail->head spatial)")
    p.add_argument("--perturbs-per-bone", type=int, default=3)
    p.add_argument("--samples", default=None, help="comma list to restrict")
    p.add_argument("--views", default="side,top")
    args = p.parse_args()

    views = args.views.split(",")
    gt = json.load(open(os.path.join(args.stage0, "gt_layouts.json")))["samples"]
    if args.samples:
        keep = set(args.samples.split(","))
        gt = {k: v for k, v in gt.items() if k in keep}

    spec_dir = os.path.join(args.out_dir, "render_specs")
    os.makedirs(spec_dir, exist_ok=True)
    episodes_path = os.path.join(args.out_dir, "episodes.jsonl")

    n_ep = 0
    with open(episodes_path, "w") as ef:
        for sample, lay in sorted(gt.items()):
            rng = random.Random("traj:" + sample)
            bones_gt = [
                {"template_id": b["template_id"],
                 "center_norm": b["center_norm"],
                 "size_norm": b["size_norm"]}
                for b in lay["bones"]
            ]
            n = len(bones_gt)
            states = []  # for the batch render spec

            def emit(step_id, state_bones, action, kind):
                nonlocal n_ep
                out_dir = os.path.join(args.out_dir, "states", sample, step_id)
                states.append({
                    "id": step_id,
                    "bones": state_bones,
                    "out_dir": out_dir,
                    "views": views,
                })
                ef.write(json.dumps({
                    "sample": sample,
                    "step_id": step_id,
                    "kind": kind,
                    "state_bones": state_bones,
                    "images": [os.path.join(out_dir, v + ".png") for v in views],
                    "target_action": action,
                }) + "\n")
                n_ep += 1

            # --- construction episodes ---
            orders = [list(range(n))]  # spatial order (gt_layouts already sorted along length)
            for _ in range(args.orders - 1):
                o = list(range(n))
                rng.shuffle(o)
                orders.append(o)
            for oi, order in enumerate(orders):
                for k in range(n + 1):
                    placed = [bones_gt[j] for j in order[:k]]
                    if k < n:
                        act = add_action(bones_gt[order[k]])
                    else:
                        act = {"action": "stop"}
                    emit("build_o%d_s%02d" % (oi, k), placed, act, "construction")

            # --- correction episodes ---
            for bi in range(n):
                for pi in range(args.perturbs_per_bone):
                    bad = perturb(bones_gt[bi], rng)
                    state = [bad if j == bi else bones_gt[j] for j in range(n)]
                    emit("fix_b%d_p%d" % (bi, pi), state,
                         adjust_action(bi, bones_gt[bi]), "correction")

            # batch render spec for this fish
            spec = {
                "fish": {"blend": os.path.abspath(os.path.join(args.gt_dir, lay["blend_file"])),
                         "object": lay["fish_mesh_object"]},
                "templates_dir": os.path.abspath(os.path.join(args.stage0, "templates")),
                "resolution": 1024,
                "states": states,
            }
            with open(os.path.join(spec_dir, sample + ".json"), "w") as f:
                json.dump(spec, f, indent=1)
            print("%s: %d bones -> %d steps" % (sample, n, len(states)))

    print("TOTAL episodes: %d -> %s" % (n_ep, episodes_path))


if __name__ == "__main__":
    main()
