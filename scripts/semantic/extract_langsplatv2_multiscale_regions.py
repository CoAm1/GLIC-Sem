#!/usr/bin/env python3
"""Extract the four SAM1 region levels used by LangSplatV2.

This is a Gate-B segmentation diagnostic.  It intentionally does not run
OpenCLIP, PCA, or Gaussian optimization.  It requires the exact
``segment-anything-langsplat`` fork pinned by the official LangSplatV2
repository and refuses to silently fall back to vanilla Segment Anything.

The upstream fork names the three SAM decoder candidates ``s/m/l`` according
to their fixed output indices 0/1/2.  Those names are retained for
reproducibility; this script does not assume that their physical mask areas are
strictly ordered.  Area statistics are recorded so that assumption can be
tested rather than asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch


UPSTREAM_REPOSITORY = (
    "https://github.com/ZhaoYujie2002/segment-anything-langsplat"
)
UPSTREAM_COMMIT = "e5dbe4b5616e24f02f15ce5a439a5edf228b3a75"
LEVELS = ("default", "s", "m", "l")
LEVEL_ALIASES = {
    "default": "all_decoder_candidates",
    "s": "decoder_candidate_0",
    "m": "decoder_candidate_1",
    "l": "decoder_candidate_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--langsplat-sam-root",
        type=Path,
        help=(
            "Checkout of ZhaoYujie2002/segment-anything-langsplat at the "
            f"pinned commit {UPSTREAM_COMMIT}. It is prepended to sys.path."
        ),
    )
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
    parser.add_argument("--outer-iou-thresh", type=float, default=0.8)
    parser.add_argument("--outer-score-thresh", type=float, default=0.7)
    parser.add_argument("--outer-inner-thresh", type=float, default=0.5)
    return parser.parse_args()


def configure_import_path(root: Path | None) -> None:
    if root is None:
        return
    root = root.resolve()
    expected = root / "segment_anything" / "automatic_mask_generator.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"--langsplat-sam-root does not contain {expected.relative_to(root)}"
        )
    sys.path.insert(0, str(root))


def load_upstream_api():
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    source_path = Path(inspect.getsourcefile(SamAutomaticMaskGenerator) or "")
    if not source_path.is_file():
        raise RuntimeError("cannot locate SamAutomaticMaskGenerator source")
    source = source_path.read_text(encoding="utf-8")
    required_markers = (
        "curr_anns_s",
        "curr_anns_m",
        "curr_anns_l",
        "masks[:,0,:,:]",
        "masks[:,1,:,:]",
        "masks[:,2,:,:]",
    )
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise RuntimeError(
            "installed segment_anything is not the LangSplat multiscale fork; "
            f"missing markers {missing}. Use --langsplat-sam-root with commit "
            f"{UPSTREAM_COMMIT}."
        )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return SamAutomaticMaskGenerator, sam_model_registry, source_path, digest


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


def official_mask_nms_indices(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    score_threshold: float,
    inner_threshold: float,
) -> torch.Tensor:
    """Reproduce LangSplatV2 preprocess.py mask-NMS selection.

    The implementation deliberately follows the upstream pairwise IoU and
    containment rules.  It returns original mask indices; callers preserve the
    original list order, matching the upstream ``filter`` helper.
    """
    if masks.ndim != 3 or scores.ndim != 1 or masks.shape[0] != scores.shape[0]:
        raise ValueError("mask/scores shapes do not agree")
    count = int(scores.shape[0])
    if count == 0:
        return torch.empty(0, dtype=torch.long)

    sorted_scores, order = scores.sort(descending=True)
    ordered_masks = masks[order].bool()
    areas = ordered_masks.sum(dim=(1, 2), dtype=torch.float32)
    iou_matrix = torch.zeros((count, count), dtype=torch.float32)
    inner_matrix = torch.zeros((count, count), dtype=torch.float32)

    for first in range(count):
        for second in range(first, count):
            intersection = torch.logical_and(
                ordered_masks[first], ordered_masks[second]
            ).sum(dtype=torch.float32)
            union = torch.logical_or(
                ordered_masks[first], ordered_masks[second]
            ).sum(dtype=torch.float32)
            iou_matrix[first, second] = intersection / union.clamp_min(1.0)
            first_fraction = intersection / areas[first].clamp_min(1.0)
            second_fraction = intersection / areas[second].clamp_min(1.0)
            containment_score = 1.0 - first_fraction * second_fraction
            if first_fraction < 0.5 and second_fraction >= 0.85:
                inner_matrix[first, second] = containment_score
            if first_fraction >= 0.85 and second_fraction < 0.5:
                inner_matrix[second, first] = containment_score

    iou_max = torch.triu(iou_matrix, diagonal=1).max(dim=0).values
    inner_upper_max = torch.triu(inner_matrix, diagonal=1).max(dim=0).values
    inner_lower_max = torch.tril(inner_matrix, diagonal=1).max(dim=0).values
    keep_iou = iou_max <= iou_threshold
    keep_confidence = sorted_scores > score_threshold
    keep_inner_upper = inner_upper_max <= 1.0 - inner_threshold
    keep_inner_lower = inner_lower_max <= 1.0 - inner_threshold

    # The upstream code attempts a top-3 fallback when an individual predicate
    # rejects everything. Make that behavior well-defined for 1-D tensors while
    # preserving the intersection with all other predicates.
    top_count = min(3, count)
    if not torch.any(keep_confidence):
        keep_confidence[:top_count] = True
    if not torch.any(keep_inner_upper):
        keep_inner_upper[:top_count] = True
    if not torch.any(keep_inner_lower):
        keep_inner_lower[:top_count] = True
    keep = keep_iou & keep_confidence & keep_inner_upper & keep_inner_lower
    return order[keep].sort().values.cpu()


def postprocess_masks(
    masks: Sequence[dict[str, Any]],
    iou_threshold: float,
    score_threshold: float,
    inner_threshold: float,
) -> list[dict[str, Any]]:
    if not masks:
        return []
    stacked = torch.from_numpy(
        np.stack([np.asarray(item["segmentation"], dtype=np.bool_) for item in masks])
    )
    scores = torch.from_numpy(np.asarray([
        float(item["predicted_iou"]) * float(item["stability_score"])
        for item in masks
    ], dtype=np.float32))
    selected = official_mask_nms_indices(
        stacked, scores, iou_threshold, score_threshold, inner_threshold
    ).tolist()
    return [masks[index] for index in selected]


def make_segmentation_map(
    masks: Sequence[dict[str, Any]], shape: tuple[int, int]
) -> np.ndarray:
    if len(masks) >= np.iinfo(np.int32).max:
        raise RuntimeError("too many regions for int32 segmentation")
    segmentation = np.full(shape, -1, dtype=np.int32)
    # Preserve upstream order. Overlap resolution is therefore identical to
    # LangSplatV2 preprocess.py's mask2segmap loop.
    for index, mask in enumerate(masks):
        segmentation[np.asarray(mask["segmentation"], dtype=np.bool_)] = index
    return segmentation


def area_statistics(areas: np.ndarray, image_pixels: int) -> dict[str, Any]:
    if areas.size == 0:
        return {
            "regions": 0,
            "area_pixels_min": None,
            "area_pixels_median": None,
            "area_pixels_p90": None,
            "area_pixels_max": None,
            "area_fraction_median": None,
        }
    return {
        "regions": int(areas.size),
        "area_pixels_min": int(areas.min()),
        "area_pixels_median": float(np.median(areas)),
        "area_pixels_p90": float(np.quantile(areas, 0.9)),
        "area_pixels_max": int(areas.max()),
        "area_fraction_median": float(np.median(areas) / image_pixels),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if len(set(args.frame_indices)) != len(args.frame_indices):
        raise ValueError("frame-indices must be unique")
    configure_import_path(args.langsplat_sam_root)
    (
        SamAutomaticMaskGenerator,
        sam_model_registry,
        generator_source,
        generator_sha256,
    ) = load_upstream_api()

    images = discover_images(args.dataset_root)
    frames = sorted(args.frame_indices)
    missing = [frame for frame in frames if frame not in images]
    if missing:
        raise FileNotFoundError(f"missing images: {missing}")
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(args.sam_checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

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
    for level in LEVELS:
        (args.output_dir / level).mkdir()

    started = time.time()
    frame_records: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frames, start=1):
        image_bgr = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"cannot read {images[frame]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        generated = generator.generate(image_rgb)
        if not isinstance(generated, tuple) or len(generated) != 4:
            raise RuntimeError(
                "LangSplat generator must return (default, s, m, l); "
                f"received {type(generated).__name__}"
            )

        level_record: dict[str, Any] = {}
        for level, raw_masks in zip(LEVELS, generated):
            masks = postprocess_masks(
                raw_masks,
                args.outer_iou_thresh,
                args.outer_score_thresh,
                args.outer_inner_thresh,
            )
            segmentation = make_segmentation_map(masks, image_rgb.shape[:2])
            areas = np.asarray([item["area"] for item in masks], dtype=np.int32)
            boxes = np.asarray([item["bbox"] for item in masks], dtype=np.float32)
            predicted_iou = np.asarray(
                [item["predicted_iou"] for item in masks], dtype=np.float32
            )
            stability = np.asarray(
                [item["stability_score"] for item in masks], dtype=np.float32
            )
            archive = args.output_dir / level / f"{frame:06d}.npz"
            np.savez_compressed(
                archive,
                segmentation=segmentation,
                boxes=boxes.reshape((-1, 4)),
                areas=areas,
                predicted_iou=predicted_iou,
                stability=stability,
            )
            stats = area_statistics(areas, segmentation.size)
            stats.update({
                "official_alias": LEVEL_ALIASES[level],
                "raw_regions_before_outer_nms": len(raw_masks),
                "pixel_coverage": float(np.mean(segmentation >= 0)),
                "archive": str(archive),
            })
            level_record[level] = stats

        frame_records.append({
            "frame_index": frame,
            "image": str(images[frame]),
            "levels": level_record,
        })
        print(
            f"[LANGSPLAT-MULTISCALE] {ordinal}/{len(frames)} frame={frame} "
            + " ".join(
                f"{level}={level_record[level]['regions']}"
                for level in LEVELS
            ),
            flush=True,
        )

    report = {
        "schema": "langsplatv2_multiscale_sam_regions_v1",
        "guardrail": (
            "No semantic reference labels, OpenCLIP features, PCA basis, or 3D "
            "map outputs were inputs to this extraction. The s/m/l names denote "
            "fixed decoder indices and are not assumed to be strict area scales."
        ),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "generator_source": str(generator_source),
            "generator_source_sha256": generator_sha256,
        },
        "segmenter": {
            "family": "Segment Anything",
            "version": "SAM1",
            "architecture": "vit_h",
            "checkpoint": str(args.sam_checkpoint),
            "levels": list(LEVELS),
            "level_aliases": LEVEL_ALIASES,
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
            "langsplat_outer_mask_nms": {
                "iou_threshold": args.outer_iou_thresh,
                "score_threshold": args.outer_score_thresh,
                "inner_threshold": args.outer_inner_thresh,
            },
        },
        "dataset_root": str(args.dataset_root),
        "frames": frames,
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
