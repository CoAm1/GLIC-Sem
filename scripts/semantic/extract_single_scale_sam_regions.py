#!/usr/bin/env python3
"""Extract only the single flattened SAM region map used by the baseline.

This lightweight Gate-B utility deliberately omits OpenCLIP, PCA, and codebook
work.  Its mask-generation and overlap ordering match
``extract_langsplat_teacher.py`` so ``points_per_side`` can be changed without
also changing the semantic representation or consuming memory for CLIP.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    return parser.parse_args()


def discover_images(root: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        for path in (root / "images").glob(suffix):
            try:
                index = int(path.stem)
            except ValueError:
                continue
            if index in images:
                raise ValueError(f"duplicate image index {index}: {path}")
            images[index] = path
    if not images:
        raise RuntimeError(f"no indexed images under {root / 'images'}")
    return images


def make_segmentation_map(
    masks: Sequence[dict[str, Any]], shape: tuple[int, int]
) -> np.ndarray:
    segmentation = np.full(shape, -1, dtype=np.int32)
    # Match the existing baseline teacher: broad masks are written first so
    # later local masks overwrite them in overlap regions.
    for index, mask in enumerate(masks):
        segmentation[np.asarray(mask["segmentation"], dtype=np.bool_)] = index
    return segmentation


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len(set(args.frame_indices)) != len(args.frame_indices):
        raise ValueError("frame-indices must be unique")
    if args.points_per_side < 1 or args.points_per_batch < 1:
        raise ValueError("point sampling values must be positive")
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(args.sam_checkpoint)

    images = discover_images(args.dataset_root)
    frames = sorted(args.frame_indices)
    missing = [frame for frame in frames if frame not in images]
    if missing:
        raise FileNotFoundError(f"missing images: {missing}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_path = Path(inspect.getsourcefile(SamAutomaticMaskGenerator) or "")
    if not source_path.is_file():
        raise RuntimeError("cannot locate SamAutomaticMaskGenerator source")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        box_nms_thresh=args.box_nms_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )
    args.output_dir.mkdir(parents=True)

    started = time.time()
    frame_records = []
    for ordinal, frame in enumerate(frames, start=1):
        image_bgr = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"cannot read {images[frame]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        generated = generator.generate(image_rgb)
        if not isinstance(generated, list):
            raise RuntimeError(
                "single-scale baseline requires vanilla Segment Anything; "
                "the installed generator returned a multiscale tuple"
            )
        masks = sorted(generated, key=lambda item: int(item["area"]), reverse=True)
        segmentation = make_segmentation_map(masks, image_rgb.shape[:2])
        archive_path = args.output_dir / f"{frame:06d}.npz"
        np.savez_compressed(
            archive_path,
            segmentation=segmentation,
            boxes=np.asarray([item["bbox"] for item in masks], dtype=np.float32).reshape((-1, 4)),
            areas=np.asarray([item["area"] for item in masks], dtype=np.int32),
            predicted_iou=np.asarray(
                [item["predicted_iou"] for item in masks], dtype=np.float32
            ),
            stability=np.asarray(
                [item["stability_score"] for item in masks], dtype=np.float32
            ),
        )
        frame_records.append({
            "frame_index": frame,
            "image": str(images[frame]),
            "archive": str(archive_path),
            "regions": len(masks),
            "pixel_coverage": float(np.mean(segmentation >= 0)),
        })
        print(
            f"[SINGLE-SAM] {ordinal}/{len(frames)} frame={frame} "
            f"regions={len(masks)} coverage={frame_records[-1]['pixel_coverage']:.4f}",
            flush=True,
        )

    report = {
        "schema": "single_scale_sam_regions_v1",
        "guardrail": (
            "No semantic reference labels, OpenCLIP features, PCA basis, or 3D "
            "map outputs were inputs to mask extraction."
        ),
        "segmenter": {
            "family": "Segment Anything",
            "version": "SAM1",
            "architecture": "vit_h",
            "checkpoint": str(args.sam_checkpoint),
            "automatic_mask_generator_source": str(source_path),
            "automatic_mask_generator_source_sha256": source_sha256,
            "automatic_mask_generator": {
                "points_per_side": args.points_per_side,
                "points_per_batch": args.points_per_batch,
                "pred_iou_threshold": args.pred_iou_thresh,
                "box_nms_threshold": args.box_nms_thresh,
                "stability_score_threshold": args.stability_score_thresh,
                "crop_n_layers": args.crop_n_layers,
                "crop_n_points_downscale_factor": (
                    args.crop_n_points_downscale_factor
                ),
                "minimum_mask_region_area": args.min_mask_region_area,
            },
        },
        "dataset_root": str(args.dataset_root),
        "frames": frames,
        "reference_metrics": {},
        "elapsed_seconds": time.time() - started,
        "frame_records": frame_records,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_dir),
        "frames": len(frames),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
