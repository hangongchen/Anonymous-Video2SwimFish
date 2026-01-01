#!/usr/bin/env python3
"""Evaluate a trained implicit-decoder flow refiner.

Per val asset:
  - run sample_implicit_volume on a query_resolution^3 grid (default 128)
  - march cubes the resulting dense SDF
  - voxel IoU at 64^3 vs GT for an apples-to-apples comparison with the dense
    refiner (the prediction is downsampled to 64^3 for the IoU bucket)
  - dump mesh.obj, sdf.npy, occupancy.npy, summary.json per asset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simfishlib.experimental.voxel_refiner.dataset import CoarseSourceMix
from simfishlib.experimental.flow_transformer_refiner import (
    ImplicitFlowRefinerConfig,
    ImplicitFlowRefinerDataset,
    build_implicit_model,
    discover_samples,
    load_checkpoint,
    sample_implicit_volume,
    split_by_asset,
    tsdf_to_mesh,
    write_obj,
)
from simfishlib.experimental.flow_transformer_refiner.io import write_json
from simfishlib.experimental.flow_transformer_refiner.target import (
    downsample_voxel,
    mesh_to_signed_distance,
)
from simfishlib.experimental.voxel_refiner.mesh_target import load_normalized_mesh


def _voxel_to_mesh_obj(occ: np.ndarray):
    """Marching cubes on a boolean 64^3 occupancy, returning (verts, faces) in [0,1]^3."""
    pseudo = (0.5 - occ.astype(np.float32))  # inside -> -0.5, outside -> +0.5
    return tsdf_to_mesh(pseudo)


def _keep_largest_component(verts: np.ndarray, faces: np.ndarray):
    """Return the largest connected component (by face count) of a mesh.

    Used to drop floating speckle islands while keeping the main body. Falls
    back to the input if trimesh is unavailable or the split fails."""
    if len(faces) == 0:
        return verts, faces
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        comps = m.split(only_watertight=False)
        if len(comps) <= 1:
            return verts, faces
        biggest = max(comps, key=lambda c: len(c.faces))
        return (
            np.asarray(biggest.vertices, dtype=np.float32),
            np.asarray(biggest.faces, dtype=np.int64),
        )
    except Exception as exc:
        print(f"[flow-eval-implicit] keep-largest failed ({exc}); keeping full mesh")
        return verts, faces


def _mesh_distance_metrics(
    pred_v: np.ndarray, pred_f: np.ndarray,
    gt_v: np.ndarray, gt_f: np.ndarray,
    *,
    num_samples: int = 50000,
    taus: tuple[float, ...] = (0.01, 0.02),
    seed: int = 0,
) -> dict:
    """Direct 3D surface-to-surface comparison between two meshes.

    Both meshes are assumed to live in the same [0,1]^3 frame. We sample points
    uniformly over each surface and use nearest-neighbour distances to compute:

      - chamfer_l2  : symmetric mean squared NN distance (lower=better)
      - chamfer_l1  : symmetric mean NN distance, same units as the coords
      - hausdorff   : max NN distance (worst-case surface error)
      - fscore@tau  : harmonic mean of precision (pred pts within tau of GT) and
                      recall (GT pts within tau of pred). higher=better, in [0,1].
                      tau is a fraction of the unit-cube edge.

    Distances are in [0,1] coordinate units (the unit cube edge = 1.0).
    """
    out: dict = {"chamfer_l1": float("nan"), "chamfer_l2": float("nan"),
                 "hausdorff": float("nan")}
    for t in taus:
        out[f"fscore@{t}"] = float("nan")
    if len(pred_f) == 0 or len(gt_f) == 0:
        return out
    try:
        import trimesh
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(seed)
        pm = trimesh.Trimesh(vertices=pred_v, faces=pred_f, process=False)
        gm = trimesh.Trimesh(vertices=gt_v, faces=gt_f, process=False)
        # area-weighted uniform surface sampling
        ps, _ = trimesh.sample.sample_surface(pm, num_samples, seed=int(seed))
        gs, _ = trimesh.sample.sample_surface(gm, num_samples, seed=int(seed) + 1)
        d_pg, _ = cKDTree(gs).query(ps)   # pred -> gt  (precision side)
        d_gp, _ = cKDTree(ps).query(gs)   # gt -> pred  (recall side)
        out["chamfer_l1"] = float(d_pg.mean() + d_gp.mean())
        out["chamfer_l2"] = float((d_pg ** 2).mean() + (d_gp ** 2).mean())
        out["hausdorff"] = float(max(d_pg.max(), d_gp.max()))
        for t in taus:
            precision = float((d_pg < t).mean())
            recall = float((d_gp < t).mean())
            f = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            out[f"fscore@{t}"] = f
    except Exception as exc:
        print(f"[flow-eval-implicit] mesh-distance metrics failed: {exc}")
    return out


def _save_visual(visual_tensor, asset_dir: Path) -> None:
    """Dump the (normalized-to-[-1,1]) visual tensor the model actually saw as
    PNGs alongside the obj files. Single image -> visual.png; video stack -> a
    visual_frame_{k:02d}.png per frame."""
    try:
        from PIL import Image
    except Exception:
        return
    arr = visual_tensor.detach().cpu().numpy()
    # denormalize: x in [-1,1] -> uint8 [0,255]
    def _to_u8(chw: np.ndarray) -> np.ndarray:
        hwc = np.transpose(chw, (1, 2, 0))
        u8 = np.clip((hwc * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return u8
    if arr.ndim == 3:
        Image.fromarray(_to_u8(arr)).save(asset_dir / "visual.png")
    elif arr.ndim == 4:
        for k in range(arr.shape[0]):
            Image.fromarray(_to_u8(arr[k])).save(asset_dir / f"visual_frame_{k:02d}.png")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Dataset/Fish2VoxelMergedV2Processed")
    p.add_argument("--qwen-coarse-root", default="Dataset/Fish2VoxelMergedV2QwenCoarse")
    p.add_argument("--video-root", default="Dataset/Fish2VoxelMergedV2Video")
    p.add_argument("--canonical-root", default="Dataset/Fish2VoxelMergedV2CanonicalViews")
    p.add_argument("--angled-root", default="Dataset/Fish2VoxelMergedV2AngledViews")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-assets", type=int, default=None)
    p.add_argument("--coarse-resolution", type=int, default=64)
    p.add_argument("--query-resolution", type=int, default=128)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--decoder-layers", type=int, default=5)
    p.add_argument("--decoder-num-freqs", type=int, default=10)
    p.add_argument("--visual-mode", choices=["video", "image", "both", "none"], default="video")
    p.add_argument("--no-visual", action="store_true")
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-query", type=int, default=65536)
    p.add_argument("--num-seeds", type=int, default=1,
                   help="average the predicted SDF volume over N independent noise "
                        "seeds. Cancels per-voxel sampling noise ~1/sqrt(N) toward "
                        "the conditional-mean (clean) SDF. 1 = current behavior.")
    p.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="Gaussian blur (in voxels) applied to the SDF volume before "
                        "marching cubes. ~0.7-1.0 removes high-freq surface fuzz. "
                        "0 = off.")
    p.add_argument("--keep-largest", action="store_true",
                   help="keep only the largest connected component of the final "
                        "mesh, dropping floating speckle islands.")
    p.add_argument("--metric-samples", type=int, default=50000,
                   help="surface points sampled per mesh for the 3D mesh-to-mesh "
                        "Chamfer / F-score metrics (predicted mesh vs GT mesh).")
    p.add_argument("--fscore-taus", type=float, nargs="+", default=[0.01, 0.02],
                   help="F-score distance thresholds, as a fraction of the unit-cube "
                        "edge (e.g. 0.01 = 1%% of the bounding box).")
    p.add_argument("--dit-xt-channels", type=int, default=0,
                   help="must match the value used at training time")
    p.add_argument("--thicken-fins", action="store_true",
                   help="must match the value used at training time (the dataset cache key includes it)")
    p.add_argument("--fin-thickness", type=float, default=0.015)
    p.add_argument("--thin-threshold", type=float, default=0.01)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        Path(args.data_root),
        resolution=64,
        max_assets=args.max_assets,
        qwen_coarse_root=Path(args.qwen_coarse_root) if args.qwen_coarse_root else None,
        canonical_root=Path(args.canonical_root) if args.canonical_root else None,
        angled_root=Path(args.angled_root) if args.angled_root else None,
        video_root=Path(args.video_root) if args.video_root else None,
    )
    _, val_samples = split_by_asset(samples, args.val_ratio, args.seed)
    if not val_samples:
        val_samples = samples
    print(f"[flow-eval-implicit] evaluating on {len(val_samples)} assets at {args.query_resolution}^3")

    config = ImplicitFlowRefinerConfig(
        coarse_resolution=args.coarse_resolution,
        visual_mode=args.visual_mode,
        thicken_fins=bool(args.thicken_fins),
        fin_thickness=float(args.fin_thickness),
        thin_threshold=float(args.thin_threshold),
    )
    use_visual = (not args.no_visual) and config.visual_mode != "none"

    model = build_implicit_model(
        coarse_resolution=args.coarse_resolution,
        patch=args.patch,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        feature_dim=args.feature_dim,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        decoder_num_freqs=args.decoder_num_freqs,
        use_visual=use_visual,
        dit_xt_channels=int(args.dit_xt_channels),
    ).to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=device)
    model.eval()

    ds = ImplicitFlowRefinerDataset(val_samples, config=config,
                                    coarse_mix=CoarseSourceMix(1.0, 0.0, 0.0), seed=args.seed)
    per_asset: list[dict[str, Any]] = []

    for i, sample in enumerate(val_samples):
        item = ds[i]
        coarse = item["coarse"].unsqueeze(0).to(device)
        visual = item["visual"].unsqueeze(0).to(device) if use_visual else None

        n_seeds = max(1, int(args.num_seeds))
        acc = None
        for s in range(n_seeds):
            gen = torch.Generator(device=device).manual_seed(int(args.seed) + 1000 * s)
            vol = sample_implicit_volume(
                model, coarse, visual,
                query_resolution=args.query_resolution,
                num_steps=args.num_steps,
                batch_query=args.batch_query,
                device=device,
                generator=gen,
            ).squeeze().cpu().numpy()
            acc = vol if acc is None else acc + vol
        sdf_pred = (acc / n_seeds).astype(np.float32)

        # Optional Gaussian smoothing of the SDF field to kill high-freq surface
        # fuzz before marching cubes (does not move a clean zero-crossing much).
        if args.smooth_sigma > 0.0:
            from scipy.ndimage import gaussian_filter
            sdf_pred = gaussian_filter(sdf_pred, sigma=float(args.smooth_sigma)).astype(np.float32)

        gt_voxel64 = np.load(sample.voxel_path).astype(bool)
        # Reduce prediction to the comparison bucket (64^3) for IoU vs GT.
        if args.query_resolution != 64:
            occ_pred_hi = sdf_pred < 0.0
            # average-pool to 64^3
            from simfishlib.experimental.flow_transformer_refiner.target import downsample_voxel as _ds
            occ64 = _ds(occ_pred_hi.astype(np.float32), 64) > 0.5
        else:
            occ64 = sdf_pred < 0.0
        gt64 = gt_voxel64
        iou_refined = iou(occ64, gt64)

        iou_raw = float("nan")
        if sample.qwen_voxel_path is not None:
            raw64 = np.load(sample.qwen_voxel_path).astype(bool)
            iou_raw = iou(raw64, gt64)

        asset_dir = output_dir / sample.asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        np.save(asset_dir / "sdf.npy", sdf_pred.astype(np.float32))
        np.save(asset_dir / "occupancy_hi.npy", (sdf_pred < 0.0))

        # 1) final mesh = the model's predicted SDF at query_resolution
        verts, faces = tsdf_to_mesh(sdf_pred)
        if args.keep_largest:
            verts, faces = _keep_largest_component(verts, faces)
        if len(verts):
            write_obj(asset_dir / "final_mesh.obj", verts, faces)
            write_obj(asset_dir / "mesh.obj", verts, faces)  # legacy alias

        # 2) qwen 64^3 coarse voxel marched
        if sample.qwen_voxel_path is not None:
            qwen_occ = np.load(sample.qwen_voxel_path).astype(bool)
            qv, qf = _voxel_to_mesh_obj(qwen_occ)
            if len(qv):
                write_obj(asset_dir / "qwen_coarse_voxel.obj", qv, qf)

        # 3) original input mesh: GT asset.obj, normalized to [0,1]^3 (same
        #    coordinate frame as the prediction, so the user can overlay them)
        #    + direct 3D mesh-to-mesh distance metrics vs the predicted mesh.
        mesh_metrics: dict = {}
        try:
            gt_mesh = load_normalized_mesh(Path(sample.mesh_path))
            gt_v = np.asarray(gt_mesh.vertices, dtype=np.float32)
            gt_f = np.asarray(gt_mesh.faces, dtype=np.int64)
            write_obj(asset_dir / "original_input.obj", gt_v, gt_f)
            mesh_metrics = _mesh_distance_metrics(
                verts, faces, gt_v, gt_f,
                num_samples=int(args.metric_samples),
                taus=tuple(args.fscore_taus),
            )
        except Exception as exc:
            print(f"[flow-eval-implicit] {sample.asset_id} original_input/metrics failed: {exc}")

        # 4) initial / GT SDF: the dense 64^3 GT TSDF the model was supervised
        #    against (same thicken flags as training so this matches the cache).
        try:
            tsdf_gt64 = mesh_to_signed_distance(
                Path(sample.mesh_path),
                resolution=args.coarse_resolution,
                truncation=4.0,
                thicken_fins=bool(args.thicken_fins),
                fin_thickness=float(args.fin_thickness),
                thin_threshold=float(args.thin_threshold),
            )
            iv, ifc = tsdf_to_mesh(tsdf_gt64.astype(np.float32))
            if len(iv):
                write_obj(asset_dir / "initial_sdf.obj", iv, ifc)
        except Exception as exc:
            print(f"[flow-eval-implicit] {sample.asset_id} initial_sdf failed: {exc}")

        # 5) visual input the model actually consumed
        if use_visual:
            try:
                _save_visual(item["visual"], asset_dir)
            except Exception as exc:
                print(f"[flow-eval-implicit] {sample.asset_id} visual save failed: {exc}")

        entry = {
            "asset_id": sample.asset_id,
            "query_resolution": args.query_resolution,
            "iou_refined_vs_gt64": iou_refined,
            "iou_raw_qwen_vs_gt64": iou_raw,
            "refined_minus_raw_qwen": (iou_refined - iou_raw) if not np.isnan(iou_raw) else float("nan"),
            "mesh_verts": int(len(verts)),
            "mesh_faces": int(len(faces)),
        }
        entry.update(mesh_metrics)  # chamfer_l1/l2, hausdorff, fscore@tau (mesh-vs-mesh, 3D)
        per_asset.append(entry)
        cd = mesh_metrics.get("chamfer_l1", float("nan"))
        f1 = mesh_metrics.get(f"fscore@{args.fscore_taus[0]}", float("nan"))
        print(f"[flow-eval-implicit] {sample.asset_id}  chamfer_l1={cd:.4f}  fscore@{args.fscore_taus[0]}={f1:.3f}  iou={iou_refined:.3f}  verts={len(verts)}")
        write_json(asset_dir / "summary.json", entry)

    iou_list = [e["iou_refined_vs_gt64"] for e in per_asset if not np.isnan(e["iou_refined_vs_gt64"])]
    raw_list = [e["iou_raw_qwen_vs_gt64"] for e in per_asset if not np.isnan(e["iou_raw_qwen_vs_gt64"])]
    delta_list = [e["refined_minus_raw_qwen"] for e in per_asset if not np.isnan(e["refined_minus_raw_qwen"])]
    # aggregate the 3D mesh-to-mesh metrics over assets
    metric_keys = ["chamfer_l1", "chamfer_l2", "hausdorff"] + [f"fscore@{t}" for t in args.fscore_taus]
    mesh_means = {}
    for k in metric_keys:
        vals = [e[k] for e in per_asset if k in e and not np.isnan(e[k])]
        mesh_means[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_assets": len(per_asset),
        "query_resolution": args.query_resolution,
        "num_steps": args.num_steps,
        # primary: 3D mesh-to-mesh distance (predicted mesh vs GT mesh surface)
        **mesh_means,
        # secondary: voxel IoU at 64^3 (coarse, kept for back-compat)
        "mean_iou_refined": float(np.mean(iou_list)) if iou_list else float("nan"),
        "mean_iou_raw_qwen": float(np.mean(raw_list)) if raw_list else float("nan"),
        "mean_delta_refined_minus_raw": float(np.mean(delta_list)) if delta_list else float("nan"),
        "mean_mesh_verts": float(np.mean([e["mesh_verts"] for e in per_asset])) if per_asset else float("nan"),
        "per_asset": per_asset,
    }
    write_json(output_dir / "summary.json", summary)
    t0 = args.fscore_taus[0]
    print(
        f"[flow-eval-implicit] mean chamfer_l1={mesh_means['mean_chamfer_l1']:.4f} "
        f"hausdorff={mesh_means['mean_hausdorff']:.4f} "
        f"fscore@{t0}={mesh_means[f'mean_fscore@{t0}']:.3f} | "
        f"iou={summary['mean_iou_refined']:.3f} avg-verts={summary['mean_mesh_verts']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
