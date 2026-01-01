#!/usr/bin/env python3
"""Run the implicit-decoder flow refiner on Dataset/testset.

Per input image, writes:

  <output-root>/<slug>/
    8.obj  / 8.png       - 8^3 Qwen coarse, upsampled to 64^3 boxy mesh
    64.obj / 64.png      - 64^3 Qwen refined voxel, boxy mesh
    mesh.obj / mesh.png  - flow-refiner-implicit output (marching cubes on TSDF)
    log/input.<ext>      - copy of the source image
    log/qwen/            - full Qwen hierarchical_inference output dir
    log/flow/            - sdf.npy + summary.json

Same testset pipeline as scripts/experimental/infer_testset_v2.py — only the
refiner stage swaps in the flow-implicit model (sample_implicit_volume +
marching cubes at level=0 on the TSDF).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from simfishlib.experimental.flow_transformer_refiner import (
    build_implicit_model,
    load_checkpoint,
    sample_implicit_volume,
    tsdf_to_mesh,
    write_obj,
)
from simfishlib.experimental.flow_transformer_refiner.io import write_json
from simfishlib.experimental.voxel_refiner.dataset import _load_pil_image, _normalize_image
from simfishlib.voxel import voxel_to_obj


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "image"


def upsample_coarse_8_to_64(coarse8: np.ndarray) -> np.ndarray:
    if coarse8.shape != (8, 8, 8):
        raise ValueError(f"Expected (8,8,8), got {coarse8.shape}")
    out = np.repeat(np.repeat(np.repeat(coarse8, 8, axis=0), 8, axis=1), 8, axis=2)
    return out.astype(bool)


def fake_video_tensor(image_path: Path, image_size: int, num_frames: int) -> torch.Tensor:
    img = _load_pil_image(image_path, image_size)
    frames = np.repeat(img[None, ...], num_frames, axis=0)
    arr = _normalize_image(frames)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float()


def render_obj_to_png(obj_path: Path, png_path: Path, elev: float = 22, azim: float = -58) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    try:
        import trimesh
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") else np.zeros((0, 3), dtype=np.int64)
    except Exception as exc:
        print(f"[render] failed to load {obj_path}: {exc}", file=sys.stderr)
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    if len(faces) > 0:
        tris = verts[faces]
        pc = Poly3DCollection(tris, facecolor="#4da3ff", edgecolor="#0a2540", linewidth=0.05, alpha=0.92)
        ax.add_collection3d(pc)
        mn = verts.min(axis=0); mx = verts.max(axis=0)
        pad = 0.02 * (mx - mn).max()
        ax.set_xlim(mn[0] - pad, mx[0] + pad)
        ax.set_ylim(mn[1] - pad, mx[1] + pad)
        ax.set_zlim(mn[2] - pad, mx[2] + pad)
        ax.set_box_aspect(((mx - mn).tolist()))
    else:
        ax.text2D(0.5, 0.5, "empty mesh", ha="center", va="center", transform=ax.transAxes)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, transparent=False, facecolor="white")
    plt.close(fig)


def iter_testset_images(testset_dir: Path) -> Iterable[Path]:
    exts = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    for p in sorted(testset_dir.iterdir()):
        if p.is_file() and p.suffix in exts:
            yield p


def run_qwen_hierarchical(image_path: Path, qwen_out_dir: Path, qwen_script: Path) -> None:
    qwen_out_dir.mkdir(parents=True, exist_ok=True)
    env = {"IMAGE_PATH": str(image_path), "OUTPUT_DIR": str(qwen_out_dir)}
    print(f"  [qwen] running {qwen_script.name} -> {qwen_out_dir}", flush=True)
    full_env = os.environ.copy()
    full_env.update(env)
    proc = subprocess.run(
        ["bash", str(qwen_script)],
        env=full_env, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (qwen_out_dir / "_launcher_stdout.log").write_bytes(proc.stdout or b"")
    if proc.returncode != 0:
        tail = (proc.stdout or b"").decode("utf-8", errors="ignore").splitlines()[-30:]
        raise RuntimeError(
            f"qwen launcher failed (exit {proc.returncode}); tail:\n" + "\n".join(tail)
        )


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testset-dir", type=Path, default=Path("Dataset/testset"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qwen-launcher", type=Path, default=Path("scripts/reproduce_checkpoint3350_infer.sh"))
    p.add_argument("--query-resolution", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-query", type=int, default=65536)
    p.add_argument("--skip-qwen-if-cached", action="store_true",
                   help="Reuse <out>/log/qwen/voxel64.npy if present.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = (ckpt.get("meta") or {}).get("args", {})
    visual_mode = train_args.get("visual_mode", "video")
    use_visual = visual_mode != "none"

    model = build_implicit_model(
        coarse_resolution=int(train_args.get("coarse_resolution", 64)),
        patch=int(train_args.get("patch", 4)),
        hidden_dim=int(train_args.get("hidden_dim", 384)),
        depth=int(train_args.get("depth", 6)),
        num_heads=int(train_args.get("num_heads", 6)),
        feature_dim=int(train_args.get("feature_dim", 128)),
        decoder_hidden=int(train_args.get("decoder_hidden", 256)),
        decoder_layers=int(train_args.get("decoder_layers", 5)),
        decoder_num_freqs=int(train_args.get("decoder_num_freqs", 10)),
        use_visual=use_visual,
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    n_params = sum(t.numel() for t in model.parameters())
    print(f"[infer] loaded flow-implicit from {args.checkpoint}  params={n_params/1e6:.1f}M  device={device}")
    print(f"[infer] train args: visual_mode={visual_mode}  num_video_frames={train_args.get('num_video_frames', 4)}")

    images = list(iter_testset_images(args.testset_dir))
    if not images:
        raise SystemExit(f"No images found under {args.testset_dir}")
    print(f"[infer] {len(images)} testset images -> {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    image_size = int(train_args.get("image_size", 224))
    num_frames = int(train_args.get("num_video_frames", 4))

    for i, img_path in enumerate(images):
        slug = slugify(img_path.stem)
        out_dir = args.output_root / slug
        log_dir = out_dir / "log"
        qwen_dir = log_dir / "qwen"
        flow_dir = log_dir / "flow"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        flow_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{i + 1}/{len(images)}] {img_path.name} -> {out_dir} ===", flush=True)

        try:
            shutil.copy2(img_path, log_dir / f"input{img_path.suffix.lower()}")
        except Exception as exc:
            print(f"  [warn] could not copy input image: {exc}", file=sys.stderr)

        voxel64_path = qwen_dir / "voxel64.npy"
        coarse8_path = qwen_dir / "coarse8.npy"
        skip = args.skip_qwen_if_cached and voxel64_path.exists() and coarse8_path.exists()
        if skip:
            print(f"  [qwen] cached at {qwen_dir}, skipping subprocess")
        else:
            run_qwen_hierarchical(img_path, qwen_dir, args.qwen_launcher)
        if not voxel64_path.exists():
            print(f"  [error] qwen did not produce {voxel64_path}; skipping", file=sys.stderr)
            continue

        if coarse8_path.exists():
            coarse8 = np.load(coarse8_path).astype(bool)
            voxel_to_obj(upsample_coarse_8_to_64(coarse8), out_dir / "8.obj")
            print(f"  [8.obj]  wrote (coarse cells = {int(coarse8.sum())})")

        voxel64 = np.load(voxel64_path).astype(bool)
        voxel_to_obj(voxel64, out_dir / "64.obj")
        print(f"  [64.obj] wrote (occupied = {int(voxel64.sum())})")

        coarse_t = torch.from_numpy(voxel64.astype(np.float32))[None, None].to(device)
        visual_in = None
        if use_visual:
            visual_in = fake_video_tensor(img_path, image_size, num_frames).to(device)
        try:
            sdf_pred = sample_implicit_volume(
                model, coarse_t, visual_in,
                query_resolution=args.query_resolution,
                num_steps=args.num_steps,
                batch_query=args.batch_query,
                device=device,
            ).squeeze().cpu().numpy()
        except Exception as exc:
            print(f"  [error] flow refiner failed: {exc}", file=sys.stderr)
            continue

        verts, faces = tsdf_to_mesh(sdf_pred)
        if len(verts):
            write_obj(out_dir / "mesh.obj", verts, faces)
        else:
            print(f"  [warn] marching cubes returned empty mesh", file=sys.stderr)

        np.save(flow_dir / "sdf.npy", sdf_pred.astype(np.float32))
        write_json(flow_dir / "summary.json", {
            "image": str(img_path),
            "checkpoint": str(args.checkpoint),
            "query_resolution": args.query_resolution,
            "num_steps": args.num_steps,
            "mesh_verts": int(len(verts)),
            "mesh_faces": int(len(faces)),
            "sdf_min": float(sdf_pred.min()),
            "sdf_max": float(sdf_pred.max()),
            "inside_frac": float((sdf_pred < 0.0).mean()),
            "coarse_occupied64": int(voxel64.sum()),
        })
        print(f"  [mesh.obj] verts={len(verts)}  faces={len(faces)}  "
              f"sdf[min,med,max]=[{sdf_pred.min():.3f},{float(np.median(sdf_pred)):.3f},{sdf_pred.max():.3f}]")

        for obj_name in ("8.obj", "64.obj", "mesh.obj"):
            obj = out_dir / obj_name
            if obj.exists():
                render_obj_to_png(obj, out_dir / (obj.stem + ".png"))

    print(f"\n[infer] done. {len(images)} images under {args.output_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
