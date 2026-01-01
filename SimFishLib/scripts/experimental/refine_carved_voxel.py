#!/usr/bin/env python
"""Feed a hand-carved 64^3 coarse voxel (+ a conditioning video) through the
trained implicit flow-transformer refiner and write the refined mesh.

Mirrors evaluate_flow_refiner_implicit.py's model build / input formatting, but
takes the coarse voxel from an arbitrary .npy (e.g. our two-view carving) and
the visual condition from an arbitrary video, so no dataset/GT is required.

Defaults match outputs/.../flow_transformer_refiner_implicit_thicken_v3/args.json.

Run on a GPU node:
  python scripts/experimental/refine_carved_voxel.py \
    --checkpoint outputs/experimental/flow_transformer_refiner_implicit_thicken_v3/checkpoint_best.pt \
    --coarse outputs/zef_carve/ZebraFish-05_f0532_voxel64.npy \
    --video  outputs/zef_fish_seg/ZebraFish-05/imgF_fish.mp4 \
    --out-dir outputs/zef_carve/refine_f0532
"""
import argparse, os
from pathlib import Path
import numpy as np
import torch

from simfishlib.experimental.flow_transformer_refiner import (
    build_implicit_model,
    sample_implicit_volume,
    tsdf_to_mesh,
)
from simfishlib.experimental.flow_transformer_refiner.io import load_checkpoint, write_obj
from simfishlib.experimental.flow_transformer_refiner.target import downsample_voxel
from simfishlib.experimental.voxel_refiner.dataset import _sample_video_frames, _normalize_image


def build_visual(video, num_frames, image_size):
    if not video:
        return None
    frames = _sample_video_frames(Path(video), num_frames, image_size)   # [K,H,W,3] uint8
    arr = _normalize_image(frames)                                       # [-1,1], channels-last
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()[None]  # [1,K,3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--coarse", required=True, help="64^3 boolean voxel .npy")
    ap.add_argument("--video", default=None, help="conditioning video (mp4); omit for no-visual*")
    ap.add_argument("--out-dir", required=True)
    # arch (v3 defaults)
    ap.add_argument("--coarse-resolution", type=int, default=64)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--num-heads", type=int, default=12)
    ap.add_argument("--feature-dim", type=int, default=128)
    ap.add_argument("--decoder-hidden", type=int, default=256)
    ap.add_argument("--decoder-layers", type=int, default=5)
    ap.add_argument("--decoder-num-freqs", type=int, default=10)
    ap.add_argument("--dit-xt-channels", type=int, default=0)
    ap.add_argument("--num-video-frames", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    # sampling
    ap.add_argument("--query-resolution", type=int, default=128)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--batch-query", type=int, default=65536)
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smooth-sigma", type=float, default=0.7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    use_visual = args.video is not None
    model = build_implicit_model(
        coarse_resolution=args.coarse_resolution, patch=args.patch,
        hidden_dim=args.hidden_dim, depth=args.depth, num_heads=args.num_heads,
        feature_dim=args.feature_dim, decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers, decoder_num_freqs=args.decoder_num_freqs,
        use_visual=use_visual, dit_xt_channels=int(args.dit_xt_channels),
    ).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=device)
    model.eval()

    # coarse: bool 64^3 -> signed [-1,1] -> [1,1,R,R,R]
    occ = np.load(args.coarse).astype(np.float32)
    if occ.shape[0] != args.coarse_resolution:
        occ = downsample_voxel(occ, args.coarse_resolution)
        occ = (occ > 0.5).astype(np.float32)
    coarse = torch.from_numpy(occ * 2.0 - 1.0)[None, None].to(device)
    print(f"coarse {tuple(coarse.shape)}  occupied={int((occ>0.5).sum())}")

    visual = build_visual(args.video, args.num_video_frames, args.image_size)
    if visual is not None:
        visual = visual.to(device)
        print(f"visual {tuple(visual.shape)} from {args.video}")

    acc = None
    for s in range(max(1, args.num_seeds)):
        gen = torch.Generator(device=device).manual_seed(args.seed + 1000 * s)
        vol = sample_implicit_volume(
            model, coarse, visual,
            query_resolution=args.query_resolution, num_steps=args.num_steps,
            batch_query=args.batch_query, device=device, generator=gen,
        ).squeeze().cpu().numpy()
        acc = vol if acc is None else acc + vol
        print(f"  seed {s} done")
    sdf = (acc / max(1, args.num_seeds)).astype(np.float32)

    if args.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        sdf = gaussian_filter(sdf, sigma=args.smooth_sigma).astype(np.float32)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sdf.npy", sdf)
    occ_hi = sdf < 0.0
    np.save(out / "occupancy_hi.npy", occ_hi)
    occ64 = downsample_voxel(occ_hi.astype(np.float32), 64) > 0.5
    np.save(out / "occ64.npy", occ64)
    verts, faces = tsdf_to_mesh(sdf)
    if len(verts):
        write_obj(out / "final_mesh.obj", verts, faces)
    print(f"[DONE] refined occupied(128^3)={int(occ_hi.sum())}  occupied(64^3)={int(occ64.sum())}")
    print(f"       -> {out/'final_mesh.obj'}  (+ sdf.npy, occ64.npy)")


if __name__ == "__main__":
    main()
