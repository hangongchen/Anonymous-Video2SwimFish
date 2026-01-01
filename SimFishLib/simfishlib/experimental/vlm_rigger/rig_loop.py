"""Stage 2-3: sequential VLM skeleton-construction loop (inference driver).

The VLM sees overlay renders of the current skeleton state and emits one JSON
action per turn:
  {"action":"add","template":"tpl_06_c01123","center_norm":[x,y,z],"size_norm":[x,y,z]}
  {"action":"adjust","bone":3,"center_norm":[...],"size_norm":[...]}
  {"action":"remove","bone":5}
  {"action":"stop"}

A deterministic executor applies each action under hard constraints, re-renders
via Blender, and loops until "stop" or the turn cap. The final state is saved
as a GT-style blend (fish mesh + bone1..boneN) for the fish_asset_pipeline.

Usage (needs GPU for the VLM):
  python3 rig_loop.py --fish path/to/fish.blend \
      --stage0 simfishlib_data/vlm_rigger/stage0 \
      --out-dir out/my_fish \
      --model Qwen/Qwen3-VL-8B-Instruct [--adapter <lora_dir>] \
      [--max-turns 25] [--max-bones 10] [--dry-run]

--dry-run replays GT actions (for plumbing tests without a GPU).
"""

import argparse
import json
import os
import re
import subprocess
import sys

BLENDER = os.environ.get("BLENDER_BIN", "/work/yzha1/tools/blender/blender")
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_overlay.py")

SYSTEM_PROMPT = """You are a fish rigging expert. You place template bones inside a fish body to build its skeleton, one action per turn.

You see two renders of the fish (translucent gray) with currently placed bones (colored, numbered head labels): a SIDE view (X = body length, Y = height) and a TOP view (X = body length, Z = thickness).

Coordinates are normalized to the fish bounding box: center_norm and size_norm are [x, y, z] with each component in 0..1 (size may slightly exceed 1). Bones must lie along the spine (y around 0.55-0.75), be ordered along X with small gaps, and their heights should roughly follow the local body height.

Available bone templates:
{template_menu}

Reply with EXACTLY one JSON action, nothing else:
{{"action":"add","template":"<id>","center_norm":[x,y,z],"size_norm":[x,y,z]}}
{{"action":"adjust","bone":<1-based index>,"center_norm":[x,y,z],"size_norm":[x,y,z]}}
{{"action":"remove","bone":<1-based index>}}
{{"action":"stop"}}

Add bones from tail to head. When the skeleton covers the body trunk well (typically 4-8 bones), reply {{"action":"stop"}}."""


def template_menu(library):
    lines = []
    for t in sorted(library.values(), key=lambda t: -t["usage_count"]):
        lines.append('  %s  (unit size %.2f x %.2f x %.2f, used %dx in GT)' % (
            t["template_id"], *[round(v, 2) for v in t["unit_size"]], t["usage_count"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- executor

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def enforce_gaps(bones, gap=0.005):
    """Deterministic V3-style spacing: sweep left->right along the length axis,
    shifting bones right just enough to keep a small positive gap. Returns the
    1-based indices of bones that were moved."""
    moved = []
    for j in range(1, len(bones)):
        prev_r = bones[j - 1]["center_norm"][0] + bones[j - 1]["size_norm"][0] / 2
        half = bones[j]["size_norm"][0] / 2
        min_c = prev_r + gap + half
        if bones[j]["center_norm"][0] < min_c:
            bones[j]["center_norm"][0] = min(min_c, 0.98)
            moved.append(j + 1)
    return moved


def validate_and_apply(state, action, library, max_bones):
    """Apply one action to state['bones']. Returns (ok, feedback_message)."""
    bones = state["bones"]
    a = action.get("action")

    if a == "stop":
        return True, "stopped"

    if a == "remove":
        i = int(action.get("bone", 0)) - 1
        if not (0 <= i < len(bones)):
            return False, "invalid bone index %s (have %d bones)" % (action.get("bone"), len(bones))
        bones.pop(i)
        return True, "removed bone %d" % (i + 1)

    if a in ("add", "adjust"):
        if a == "add" and len(bones) >= max_bones:
            return False, "bone limit reached (%d); adjust or stop" % max_bones
        tpl = action.get("template")
        if a == "add" and tpl not in library:
            return False, "unknown template %r" % tpl
        try:
            center = [float(v) for v in action["center_norm"]]
            size = [float(v) for v in action["size_norm"]]
        except (KeyError, TypeError, ValueError):
            return False, "center_norm/size_norm must be [x,y,z] numbers"
        center = [clamp(v, 0.02, 0.98) for v in center]
        size = [clamp(v, 0.02, 1.5) for v in size]

        entry = {"center_norm": center, "size_norm": size}
        if a == "add":
            entry["template_id"] = tpl
            bones.append(entry)
            bones.sort(key=lambda b: b["center_norm"][0])
            msg = "added bone (now %d, renumbered along length)" % len(bones)
        else:
            i = int(action.get("bone", 0)) - 1
            if not (0 <= i < len(bones)):
                return False, "invalid bone index %s (have %d bones)" % (action.get("bone"), len(bones))
            entry["template_id"] = bones[i]["template_id"]
            if action.get("template") in library:
                entry["template_id"] = action["template"]
            bones[i] = entry
            bones.sort(key=lambda b: b["center_norm"][0])
            msg = "adjusted bone %d" % (i + 1)

        fixed = enforce_gaps(bones)
        if fixed:
            msg += "; auto-spaced bones %s to remove overlap" % fixed
        return True, msg

    return False, "unknown action %r" % a


def render(state, spec_base, out_dir, save_blend=None):
    """Render current state; enforcement of the inside-the-mesh hard
    constraint happens in Blender and the corrected bones are synced back
    into `state`. Returns (images, meta)."""
    spec = dict(spec_base)
    spec.update({"bones": state["bones"], "out_dir": out_dir,
                 "views": ["side", "top"], "save_blend": save_blend,
                 "enforce_inside": True})
    spec_path = os.path.join(out_dir, "spec.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    r = subprocess.run(
        [BLENDER, "--background", "--python", RENDER_SCRIPT, "--", "--spec", spec_path],
        capture_output=True, text=True)
    if "DONE render" not in r.stdout:
        raise RuntimeError("render failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    with open(os.path.join(out_dir, "render_meta.json")) as f:
        meta = json.load(f)
    # sync auto-shrunk bones back into the loop state
    shrunk = []
    for i, bf in enumerate(meta.get("bones_final", [])):
        state["bones"][i]["center_norm"] = [round(v, 4) for v in bf["center_norm"]]
        state["bones"][i]["size_norm"] = [round(v, 4) for v in bf["size_norm"]]
        if bf["shrunk"]:
            shrunk.append(i + 1)
    return [os.path.join(out_dir, "side.png"), os.path.join(out_dir, "top.png")], meta, shrunk


def parse_action(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- drivers

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


class GTReplayPolicy:
    """--dry-run: replays the GT layout for a known sample, then stops."""

    def __init__(self, stage0, sample):
        gt = json.load(open(os.path.join(stage0, "gt_layouts.json")))["samples"]
        self.queue = [
            {"action": "add", "template": b["template_id"],
             "center_norm": b["center_norm"], "size_norm": b["size_norm"]}
            for b in gt[sample]["bones"]
        ]

    def __call__(self, images, history, state):
        if self.queue:
            return json.dumps(self.queue.pop(0))
        return '{"action":"stop"}'


class VLMPolicy:
    """Prompt layout mirrors the SFT rows built by build_sft_jsonl.py."""

    def __init__(self, model, adapter, library):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        from simfishlib.inference.qwen3vl import Qwen3VLClient
        self.client = Qwen3VLClient(model_path=model, adapter_path=adapter)
        self.client.load()
        self.system = SYSTEM_PROMPT.format(template_menu=template_menu(library))

    def __call__(self, images, history, state):
        user = ("SIDE view (image 1) and TOP view (image 2) of the fish "
                "with the current skeleton.\n%s\nYour action:" %
                state_summary(state["bones"]))
        prompt = self.system + "\n\n" + user
        return self.client.generate_from_images(images, prompt, max_new_tokens=256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fish", required=True, help=".blend or .obj with the fish mesh")
    p.add_argument("--fish-object", default=None)
    p.add_argument("--stage0", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--max-bones", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="replay GT actions (sample inferred from fish file name)")
    args = p.parse_args()

    with open(os.path.join(args.stage0, "bone_template_library.json")) as f:
        library = {t["template_id"]: t for t in json.load(f)["templates"]}

    fish_key = "obj" if args.fish.endswith(".obj") else "blend"
    spec_base = {
        "fish": {fish_key: os.path.abspath(args.fish), "object": args.fish_object},
        "templates_dir": os.path.abspath(os.path.join(args.stage0, "templates")),
        "resolution": 1024,
    }

    if args.dry_run:
        sample = os.path.splitext(os.path.basename(args.fish))[0]
        policy = GTReplayPolicy(args.stage0, sample)
    else:
        if not args.model:
            p.error("--model required unless --dry-run")
        policy = VLMPolicy(args.model, args.adapter, library)

    state = {"bones": []}
    history = []
    log_path = os.path.join(args.out_dir, "loop_log.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(log_path, "w") as log:
        for turn in range(args.max_turns):
            turn_dir = os.path.join(args.out_dir, "turn_%02d" % turn)
            images, meta, shrunk = render(state, spec_base, turn_dir)
            if shrunk:
                print("turn %02d: auto-shrunk bones %s to stay inside the fish mesh" % (turn, shrunk))
            raw = policy(images, history, state)
            action = parse_action(raw)
            if action is None:
                feedback = "could not parse action JSON; reply with exactly one JSON object"
                ok = False
            else:
                ok, feedback = validate_and_apply(state, action, library, args.max_bones)
            rec = {"turn": turn, "raw": raw, "action": action, "ok": ok,
                   "feedback": feedback, "bone_count": len(state["bones"])}
            history.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("turn %02d: %s -> %s" % (turn, action, feedback))
            if action and action.get("action") == "stop":
                break
            if len(history) >= 2 and history[-2]["action"] == action:
                print("converged: identical action repeated, stopping")
                break

    final_dir = os.path.join(args.out_dir, "final")
    blend_path = os.path.join(args.out_dir, "auto_skeleton.blend")
    _, meta, _ = render(state, spec_base, final_dir, save_blend=blend_path)
    bad = [i + 1 for i, bf in enumerate(meta.get("bones_final", [])) if not bf["contained"]]
    if bad:
        print("WARNING: bones %s could not be fully contained in the fish mesh" % bad)
    with open(os.path.join(args.out_dir, "final_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print("FINAL blend: %s (%d bones)" % (blend_path, len(state["bones"])))


if __name__ == "__main__":
    main()
