#!/usr/bin/env python3
"""Stream SAM regions through the LangSplatV2 OpenCLIP teacher.

This is a bounded teacher-quality probe, not a LangSplatV2 reproduction. It
uses one automatic-mask scale and saves one frame at a time to avoid retaining
all full-resolution segmentation maps in RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import open_clip
from PIL import Image
import torch
import torch.nn.functional as functional
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


CLASS_NAMES = [
    "wall", "floor", "ceiling", "table", "chair", "door", "cabinet",
    "soccer ball", "storage box", "fire extinguisher", "sign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-reference-dir", type=Path)
    parser.add_argument("--frame-indices", type=int, nargs="*")
    parser.add_argument("--start-index", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--codebook-holdout-frames", type=int, default=1)
    parser.add_argument("--convex-oracle-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def discover_images(root: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        for path in (root / "images").glob(suffix):
            try:
                images[int(path.stem)] = path
            except ValueError:
                continue
    if not images:
        raise RuntimeError(f"No indexed images found under {root / 'images'}")
    return images


def square_masked_crop(image_rgb: np.ndarray, mask: dict) -> Image.Image:
    x, y, width, height = [int(round(value)) for value in mask["bbox"]]
    x = max(0, x)
    y = max(0, y)
    width = max(1, min(width, image_rgb.shape[1] - x))
    height = max(1, min(height, image_rgb.shape[0] - y))
    crop = image_rgb[y : y + height, x : x + width].copy()
    crop_mask = mask["segmentation"][y : y + height, x : x + width]
    crop[~crop_mask] = 0
    side = max(width, height)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    square[top : top + height, left : left + width] = crop
    return Image.fromarray(square)


@torch.inference_mode()
def encode_regions(
    image_rgb: np.ndarray,
    masks: list[dict],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for begin in range(0, len(masks), batch_size):
        batch_masks = masks[begin : begin + batch_size]
        batch = torch.stack([
            preprocess(square_masked_crop(image_rgb, mask))
            for mask in batch_masks
        ]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            features = model.encode_image(batch)
        outputs.append(functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs) if outputs else torch.empty((0, 512))


def make_segmentation_map(masks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    if len(masks) >= np.iinfo(np.int16).max:
        raise RuntimeError("Too many regions for int16 segmentation map")
    segmentation = np.full(shape, -1, dtype=np.int16)
    # Large regions first, small regions last: local objects overwrite broad
    # background surfaces instead of disappearing behind them.
    for index, mask in enumerate(masks):
        segmentation[mask["segmentation"]] = index
    return segmentation


@torch.inference_mode()
def encode_class_prompts(model, tokenizer, device: torch.device) -> torch.Tensor:
    prompts = [f"a photo of a {name}" for name in CLASS_NAMES]
    tokens = tokenizer(prompts).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        features = model.encode_text(tokens)
    return functional.normalize(features.float(), dim=-1).cpu()


def cosine_kmeans(features: torch.Tensor, count: int, iterations: int, seed: int) -> torch.Tensor:
    count = min(count, features.shape[0])
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randperm(features.shape[0], generator=generator)[:count]
    centers = features[initial].clone()
    for _ in range(iterations):
        assignment = (features @ centers.T).argmax(dim=1)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, assignment, features)
        counts = torch.bincount(assignment, minlength=count).float().unsqueeze(1)
        empty = counts.squeeze(1) == 0
        new_centers = new_centers / counts.clamp_min(1.0)
        if empty.any():
            replacement = torch.randperm(features.shape[0], generator=generator)[: int(empty.sum())]
            new_centers[empty] = features[replacement]
        centers = functional.normalize(new_centers, dim=1)
    return centers


def topk_unconstrained_reconstruction_cosine(
    features: torch.Tensor, centers: torch.Tensor, topk: int = 4
) -> torch.Tensor:
    k = min(topk, centers.shape[0])
    indices = (features @ centers.T).topk(k, dim=1).indices
    selected = centers[indices]
    gram = selected @ selected.transpose(1, 2)
    identity = torch.eye(k).unsqueeze(0)
    rhs = (selected @ features.unsqueeze(2)).squeeze(2)
    coefficients = torch.linalg.solve(gram + 1e-4 * identity, rhs)
    reconstruction = (selected * coefficients.unsqueeze(2)).sum(dim=1)
    reconstruction = functional.normalize(reconstruction, dim=1)
    return (features * reconstruction).sum(dim=1)


def topk_convex_oracle_cosine(
    features: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
    topk: int = 4,
    steps: int = 40,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Optimistic Top-K score under LangSplatV2-compatible coefficients.

    LangSplatV2 applies softmax to each Gaussian's codebook logits and
    renormalizes the selected Top-K values.  Here every teacher region gets
    independently optimized logits, so this remains an upper bound rather
    than a prediction of the eventual multi-view Gaussian-map accuracy.
    """
    k = min(topk, centers.shape[0])
    similarities = features @ centers.T
    all_scores: list[torch.Tensor] = []
    for begin in range(0, features.shape[0], batch_size):
        end = min(begin + batch_size, features.shape[0])
        target = features[begin:end].to(device)
        indices = similarities[begin:end].topk(k, dim=1).indices
        selected = centers[indices].to(device)
        logits = similarities[begin:end].topk(k, dim=1).values.to(device)
        logits = logits.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=0.2)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            weights = torch.softmax(logits, dim=1)
            reconstruction = (selected * weights.unsqueeze(2)).sum(dim=1)
            score = (
                functional.normalize(reconstruction, dim=1) * target
            ).sum(dim=1)
            (-score.mean()).backward()
            optimizer.step()
        with torch.no_grad():
            weights = torch.softmax(logits, dim=1)
            reconstruction = (selected * weights.unsqueeze(2)).sum(dim=1)
            score = (
                functional.normalize(reconstruction, dim=1) * target
            ).sum(dim=1)
        all_scores.append(score.cpu())
    return torch.cat(all_scores)


def codebook_metrics(
    features: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
    convex_oracle_steps: int,
) -> dict:
    top1 = (features * centers[(features @ centers.T).argmax(dim=1)]).sum(dim=1)
    top4_unconstrained = topk_unconstrained_reconstruction_cosine(
        features, centers, topk=4
    )
    top4_convex = topk_convex_oracle_cosine(
        features,
        centers,
        device=device,
        topk=4,
        steps=convex_oracle_steps,
    )
    return {
        "regions": int(features.shape[0]),
        "top1_cosine_mean": round(float(top1.mean()), 6),
        "top1_cosine_p05": round(float(torch.quantile(top1, 0.05)), 6),
        "top4_unconstrained_oracle_cosine_mean": round(
            float(top4_unconstrained.mean()), 6
        ),
        "top4_unconstrained_oracle_cosine_p05": round(
            float(torch.quantile(top4_unconstrained, 0.05)), 6
        ),
        "top4_convex_oracle_cosine_mean": round(float(top4_convex.mean()), 6),
        "top4_convex_oracle_cosine_p05": round(
            float(torch.quantile(top4_convex, 0.05)), 6
        ),
    }


def reference_metrics(
    frame_records: list[dict], text_features: torch.Tensor, reference_dir: Path | None
) -> dict:
    if reference_dir is None:
        return {}
    eligible = 0
    correct = 0
    weighted_correct = 0
    weighted_total = 0
    for record in frame_records:
        reference = cv2.imread(
            str(reference_dir / "labels" / f"{record['frame_index']:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if reference is None:
            continue
        archive = np.load(record["archive"])
        features = torch.from_numpy(archive["features"].astype(np.float32))
        segmentation = archive["segmentation"]
        clip_class = (features @ text_features.T).argmax(dim=1).numpy() + 1
        for region_index, predicted_class in enumerate(clip_class):
            pixels = reference[segmentation == region_index]
            labelled = pixels[(pixels > 0) & (pixels <= len(CLASS_NAMES))]
            if labelled.size == 0 or labelled.size < 0.5 * max(1, pixels.size):
                continue
            counts = np.bincount(labelled, minlength=len(CLASS_NAMES) + 1)
            target_class = int(counts[1:].argmax() + 1)
            eligible += 1
            correct += int(predicted_class == target_class)
            weighted_total += int(labelled.size)
            weighted_correct += int(labelled.size) * int(predicted_class == target_class)
    return {
        "eligible_regions": eligible,
        "region_top1_accuracy": round(correct / eligible, 6) if eligible else None,
        "pixel_weighted_region_top1_accuracy": round(
            weighted_correct / weighted_total, 6
        ) if weighted_total else None,
        "random_top1_accuracy": round(1.0 / len(CLASS_NAMES), 6),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the bounded SAM/CLIP probe")

    images = discover_images(args.dataset_root)
    if args.frame_indices:
        selected_indices = args.frame_indices
    else:
        selected_indices = sorted(
            index for index in images
            if index >= args.start_index and (index - args.start_index) % args.frame_step == 0
        )[: args.max_frames]
    missing = [index for index in selected_indices if index not in images]
    if missing:
        raise RuntimeError(f"Missing selected frames: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip_checkpoint), precision="fp16"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    text_features = encode_class_prompts(model, tokenizer, device)

    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    started = time.time()
    records: list[dict] = []
    all_features: list[torch.Tensor] = []
    for ordinal, frame_index in enumerate(selected_indices, start=1):
        image_bgr = cv2.imread(str(images[frame_index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read {images[frame_index]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = generator.generate(image_rgb)
        masks = sorted(masks, key=lambda item: int(item["area"]), reverse=True)
        features = encode_regions(
            image_rgb, masks, model, preprocess, device, args.batch_size
        )
        if features.shape != (len(masks), 512):
            raise RuntimeError("Unexpected teacher feature shape")
        if not torch.isfinite(features).all():
            raise RuntimeError("Teacher features contain non-finite values")
        segmentation = make_segmentation_map(masks, image_rgb.shape[:2])
        archive_path = args.output_dir / f"{frame_index:06d}.npz"
        np.savez_compressed(
            archive_path,
            features=features.numpy().astype(np.float16),
            segmentation=segmentation,
            boxes=np.asarray([mask["bbox"] for mask in masks], dtype=np.int32),
            areas=np.asarray([mask["area"] for mask in masks], dtype=np.int32),
            predicted_iou=np.asarray(
                [mask["predicted_iou"] for mask in masks], dtype=np.float16
            ),
            stability=np.asarray(
                [mask["stability_score"] for mask in masks], dtype=np.float16
            ),
        )
        coverage = float(np.mean(segmentation >= 0))
        records.append({
            "frame_index": frame_index,
            "image": str(images[frame_index]),
            "archive": str(archive_path),
            "regions": len(masks),
            "pixel_coverage": round(coverage, 6),
        })
        all_features.append(features)
        print(
            f"[LANG-TEACHER] {ordinal}/{len(selected_indices)} frame={frame_index} "
            f"regions={len(masks)} coverage={coverage:.3f}",
            flush=True,
        )

    if args.codebook_holdout_frames < 1 or args.codebook_holdout_frames >= len(all_features):
        raise RuntimeError(
            "codebook_holdout_frames must leave at least one fit and one held-out frame"
        )
    features = functional.normalize(torch.cat(all_features).float(), dim=1)
    fit_frame_count = len(all_features) - args.codebook_holdout_frames
    fit_features = functional.normalize(
        torch.cat(all_features[:fit_frame_count]).float(), dim=1
    )
    heldout_features = functional.normalize(
        torch.cat(all_features[fit_frame_count:]).float(), dim=1
    )
    centers = cosine_kmeans(
        fit_features, args.codebook_size, args.kmeans_iterations, args.seed
    )
    np.save(args.output_dir / "codebook64.npy", centers.numpy().astype(np.float16))
    report = {
        "schema": "single_scale_sam_openclip_teacher_probe",
        "warning": (
            "Optional reference metrics are evaluator-only diagnostics; they are "
            "not teacher supervision or dense ground-truth accuracy."
        ),
        "segmenter": {
            "family": "Segment Anything",
            "version": "SAM1",
            "architecture": "vit_h",
            "checkpoint": str(args.sam_checkpoint),
            "automatic_mask_generator": {
                "points_per_side": args.points_per_side,
                "pred_iou_threshold": 0.7,
                "box_nms_threshold": 0.7,
                "stability_score_threshold": 0.85,
                "crop_n_layers": 1,
                "minimum_mask_region_area": 100,
            },
        },
        "reference_diagnostics_enabled": args.semantic_reference_dir is not None,
        "clip_model": "ViT-B-16",
        "clip_pretrained": "laion2b_s34b_b88k",
        "clip_checkpoint": str(args.clip_checkpoint),
        "sam_checkpoint": str(args.sam_checkpoint),
        "frames": len(records),
        "total_regions": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_norm_min": round(float(features.norm(dim=1).min()), 6),
        "feature_norm_max": round(float(features.norm(dim=1).max()), 6),
        "feature_finite": bool(torch.isfinite(features).all()),
        "mean_pixel_coverage": round(
            float(np.mean([record["pixel_coverage"] for record in records])), 6
        ),
        "codebook_size": int(centers.shape[0]),
        "codebook_fit_frames": fit_frame_count,
        "codebook_holdout_frames": args.codebook_holdout_frames,
        "codebook_fit_metrics": codebook_metrics(
            fit_features, centers, device, args.convex_oracle_steps
        ),
        "codebook_holdout_metrics": codebook_metrics(
            heldout_features, centers, device, args.convex_oracle_steps
        ),
        "reference_metrics": reference_metrics(
            records, text_features, args.semantic_reference_dir
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "frame_records": records,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
