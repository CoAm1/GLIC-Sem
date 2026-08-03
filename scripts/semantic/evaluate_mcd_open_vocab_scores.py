#!/usr/bin/env python3
"""Evaluate absolute open-vocabulary score PNGs on sparse MCD projections.

MCD labels are used only here. They are sparse LiDAR projections rather than
dense 2D ground truth. Prompt groups and unknown thresholds must be frozen
before held-out results are inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_grid(value: str) -> list[float]:
    result = sorted({float(item) for item in value.split(",") if item.strip()})
    if not result or result[0] < 0.0 or result[-1] > 1.0:
        raise argparse.ArgumentTypeError("grid values must be in [0, 1]")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--alpha-root", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--keyframe-period", type=int, default=5)
    parser.add_argument("--keyframe-offset", type=int, default=4)
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Expected PNG height; the non-default option is intended for tests.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Expected PNG width; the non-default option is intended for tests.",
    )
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--score-kind", default="raw_uint16")
    parser.add_argument("--calibrate-unknown-on-train", action="store_true")
    parser.add_argument(
        "--threshold-grid",
        type=parse_grid,
        default=parse_grid("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"),
    )
    parser.add_argument(
        "--margin-grid",
        type=parse_grid,
        default=parse_grid("0,0.01,0.02,0.05,0.1,0.2"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpp_safe_file_label(label: str) -> str:
    """Match safeFileLabel() byte-for-byte under the C locale."""
    encoded = label.encode("utf-8")
    safe = bytes(
        byte
        if (
            ord("0") <= byte <= ord("9")
            or ord("A") <= byte <= ord("Z")
            or ord("a") <= byte <= ord("z")
            or byte in (ord("-"), ord("_"))
        )
        else ord("_")
        for byte in encoded
    ).decode("ascii")
    return safe or "query"


@dataclass(frozen=True)
class PromptClass:
    class_id: int
    name: str
    queries: tuple[str, ...]


def load_prompt_groups(path: Path) -> tuple[list[PromptClass], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") not in {
        "mcd_open_vocab_prompt_groups_v1",
        "mcd_open_vocab_prompt_groups_v2",
    }:
        raise ValueError("unexpected prompt-group schema")
    if document.get("frozen_before_heldout_evaluation") is not True:
        raise ValueError("prompt groups must be marked frozen before held-out evaluation")
    primary = [int(value) for value in document["primary_macro_class_ids"]]
    classes: list[PromptClass] = []
    safe_to_query: dict[str, str] = {}
    for class_id in primary:
        record = document["classes"].get(str(class_id))
        if not record:
            raise ValueError(f"missing primary class {class_id}")
        queries = tuple(str(query) for query in record["queries"])
        if not queries:
            raise ValueError(f"class {class_id} has no queries")
        for query in queries:
            safe = cpp_safe_file_label(query)
            previous = safe_to_query.setdefault(safe, query)
            if previous != query:
                raise ValueError(
                    f"safe-label collision: {previous!r} and {query!r} -> {safe!r}"
                )
        classes.append(PromptClass(class_id, str(record["name"]), queries))
    if len({item.class_id for item in classes}) != len(classes):
        raise ValueError("duplicate primary class id")
    return classes, document


def load_alignment(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected = {"frame_index", "label_scan_index"}
        if not expected.issubset(reader.fieldnames or ()):
            raise ValueError("alignment CSV lacks frame_index/label_scan_index")
        for row in reader:
            result[int(row["frame_index"])] = int(row["label_scan_index"])
    return result


def load_png(path: Path, expected_dtype: np.dtype, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype != expected_dtype:
        raise TypeError(f"{path}: expected {expected_dtype}, got {image.dtype}")
    if image.shape != shape:
        raise ValueError(f"{path}: expected shape {shape}, got {image.shape}")
    return image


def aggregate_scores(
    score_root: Path,
    frame_name: str,
    classes: list[PromptClass],
    aggregation: str,
    shape: tuple[int, int],
) -> np.ndarray:
    class_scores = []
    for item in classes:
        query_scores = []
        for query in item.queries:
            path = score_root / cpp_safe_file_label(query) / frame_name
            score_u16 = load_png(path, np.dtype(np.uint16), shape)
            query_scores.append(score_u16.astype(np.float32) / 65535.0)
        stacked = np.stack(query_scores)
        if aggregation == "max":
            class_scores.append(stacked.max(axis=0))
        else:
            class_scores.append(stacked.mean(axis=0))
    return np.stack(class_scores, axis=-1)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    cumulative_true = np.cumsum(sorted_labels)
    cumulative_false = np.cumsum(1 - sorted_labels)
    # Group tied scores so AP is independent of input order.
    group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_at_threshold = cumulative_true[group_end]
    false_at_threshold = cumulative_false[group_end]
    precision = true_at_threshold / np.maximum(
        true_at_threshold + false_at_threshold, 1
    )
    recall = true_at_threshold / positives
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def metric_report(
    reference: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    classes: list[PromptClass],
    reference_known_pixels_before_alpha: int,
    frames_with_pixels: int,
) -> dict:
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    index_for_id = {class_id: index for index, class_id in enumerate(class_ids)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true_id, predicted_id in zip(reference, prediction):
        if int(predicted_id) not in index_for_id:
            continue
        confusion[index_for_id[int(true_id)], index_for_id[int(predicted_id)]] += 1

    per_class = {}
    ious = []
    recalls = []
    aps = []
    weighted_iou_sum = 0.0
    total_support = int(len(reference))
    for index, item in enumerate(classes):
        true_positive = int(np.sum((reference == item.class_id) & (prediction == item.class_id)))
        false_positive = int(np.sum((reference != item.class_id) & (prediction == item.class_id)))
        false_negative = int(np.sum((reference == item.class_id) & (prediction != item.class_id)))
        support = true_positive + false_negative
        predicted_count = true_positive + false_positive
        union = true_positive + false_positive + false_negative
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        iou = true_positive / union if union else None
        ap = average_precision(reference == item.class_id, scores[:, index])
        if iou is not None:
            ious.append(iou)
            weighted_iou_sum += support * iou
        if recall is not None:
            recalls.append(recall)
        if ap is not None:
            aps.append(ap)
        per_class[str(item.class_id)] = {
            "name": item.name,
            "reference_pixels": support,
            "predicted_pixels": predicted_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "ap": ap,
            "ap_positive_pixels": support,
            "ap_negative_pixels": total_support - support,
        }

    correct = int(np.sum(reference == prediction))
    return {
        "reference_pixels": total_support,
        "frames_with_pixels": frames_with_pixels,
        "reference_known_pixels_before_alpha": reference_known_pixels_before_alpha,
        "alpha_valid_coverage": (
            total_support / reference_known_pixels_before_alpha
            if reference_known_pixels_before_alpha
            else None
        ),
        "pixel_accuracy": correct / total_support if total_support else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_iou": float(np.mean(ious)) if ious else None,
        "frequency_weighted_iou": (
            weighted_iou_sum / total_support if total_support else None
        ),
        "macro_ap": float(np.mean(aps)) if aps else None,
        "confusion_class_ids": class_ids.tolist(),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def concatenate(parts: list[np.ndarray], columns: int | None = None) -> np.ndarray:
    if parts:
        return np.concatenate(parts, axis=0)
    shape = (0,) if columns is None else (0, columns)
    return np.empty(shape, dtype=np.float32)


def predictions_from_scores(
    scores: np.ndarray,
    class_ids: np.ndarray,
    score_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> np.ndarray:
    if len(scores) == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(scores, axis=1)
    top_index = order[:, -1]
    top_score = np.take_along_axis(scores, top_index[:, None], axis=1)[:, 0]
    runner_up = np.take_along_axis(scores, order[:, -2:-1], axis=1)[:, 0]
    prediction = class_ids[top_index].copy()
    prediction[(top_score < score_threshold) | (
        top_score - runner_up < margin_threshold
    )] = 0
    return prediction


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.start_frame > args.end_frame:
        raise ValueError("start-frame must be <= end-frame")
    if args.keyframe_period < 1:
        raise ValueError("keyframe-period must be positive")
    if not 0 <= args.keyframe_offset < args.keyframe_period:
        raise ValueError("keyframe-offset must be in [0, keyframe-period)")
    if not 0.0 <= args.min_reference_confidence <= 1.0:
        raise ValueError("min-reference-confidence must be in [0, 1]")
    if args.image_height < 1 or args.image_width < 1:
        raise ValueError("image dimensions must be positive")

    shape = (args.image_height, args.image_width)
    classes, prompt_document = load_prompt_groups(args.prompt_groups)
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    alignment = load_alignment(args.alignment_csv)
    frames = list(range(args.start_frame, args.end_frame + 1))
    missing_alignment = [frame for frame in frames if frame not in alignment]
    if missing_alignment:
        raise ValueError(f"missing alignment rows: {missing_alignment[:8]}")
    training_frames = {
        frame
        for frame in frames
        if frame % args.keyframe_period == args.keyframe_offset
    }
    train_scans = {
        alignment[frame] for frame in training_frames if alignment[frame] >= 0
    }

    split_reference: dict[str, list[np.ndarray]] = {"train": [], "heldout": []}
    split_scores: dict[str, list[np.ndarray]] = {"train": [], "heldout": []}
    split_known_before_alpha = {"train": 0, "heldout": 0}
    split_frames_with_pixels = {"train": 0, "heldout": 0}
    excluded_shared_scan_frames: list[int] = []
    unmatched_frames: list[int] = []
    per_frame = []
    excluded_reference_counts = {str(value): 0 for value in (0, 11)}

    for frame in frames:
        frame_name = f"{frame:06d}.png"
        reference = load_png(
            args.reference_dir / "labels" / frame_name,
            np.dtype(np.uint8),
            shape,
        )
        confidence_u8 = load_png(
            args.reference_dir / "confidence" / frame_name,
            np.dtype(np.uint8),
            shape,
        )
        alpha = load_png(
            args.alpha_root / frame_name,
            np.dtype(np.uint8),
            shape,
        )
        class_scores = aggregate_scores(
            args.score_root, frame_name, classes, args.aggregation, shape
        )
        confidence = confidence_u8.astype(np.float32) / 255.0
        primary_reference = np.isin(reference, class_ids)
        confident_reference = confidence >= args.min_reference_confidence
        known_before_alpha = primary_reference & confident_reference
        valid = known_before_alpha & (alpha > 0)

        for excluded_id in (0, 11):
            excluded_reference_counts[str(excluded_id)] += int(
                np.sum((reference == excluded_id) & confident_reference)
            )

        scan = alignment[frame]
        if frame in training_frames:
            split = "train"
        elif scan < 0:
            split = "unmatched"
            unmatched_frames.append(frame)
        elif scan in train_scans:
            split = "shared_scan_excluded"
            excluded_shared_scan_frames.append(frame)
        else:
            split = "heldout"

        valid_count = int(valid.sum())
        frame_accuracy = None
        if split in split_reference:
            split_known_before_alpha[split] += int(known_before_alpha.sum())
            if valid_count:
                reference_values = reference[valid].astype(np.int64)
                score_values = class_scores[valid].astype(np.float32)
                prediction_values = predictions_from_scores(
                    score_values, class_ids
                )
                split_reference[split].append(reference_values)
                split_scores[split].append(score_values)
                split_frames_with_pixels[split] += 1
                frame_accuracy = float(
                    np.mean(reference_values == prediction_values)
                )

        per_frame.append({
            "frame": frame,
            "label_scan_index": scan,
            "split": split,
            "reference_known_pixels_before_alpha": int(known_before_alpha.sum()),
            "evaluated_pixels": valid_count if split in split_reference else 0,
            "alpha_valid_coverage": (
                valid_count / int(known_before_alpha.sum())
                if known_before_alpha.any()
                else None
            ),
            "top1_accuracy": frame_accuracy,
        })

    arrays = {}
    threshold_free = {}
    for split in ("train", "heldout"):
        reference = concatenate(split_reference[split]).astype(np.int64)
        scores = concatenate(split_scores[split], len(classes)).astype(np.float32)
        prediction = predictions_from_scores(scores, class_ids)
        arrays[split] = (reference, scores)
        threshold_free[split] = metric_report(
            reference,
            prediction,
            scores,
            classes,
            split_known_before_alpha[split],
            split_frames_with_pixels[split],
        )

    calibration = {
        "enabled": bool(args.calibrate_unknown_on_train),
        "limitation": (
            "MCD reference label 0 is ignore, not validated unknown ground truth."
        ),
    }
    thresholded = None
    if args.calibrate_unknown_on_train:
        train_reference, train_scores = arrays["train"]
        grid_records = []
        best_key = None
        best_thresholds = None
        for score_threshold in args.threshold_grid:
            for margin_threshold in args.margin_grid:
                prediction = predictions_from_scores(
                    train_scores,
                    class_ids,
                    score_threshold,
                    margin_threshold,
                )
                report = metric_report(
                    train_reference,
                    prediction,
                    train_scores,
                    classes,
                    split_known_before_alpha["train"],
                    split_frames_with_pixels["train"],
                )
                macro_iou = report["macro_iou"]
                record = {
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                    "train_macro_iou": macro_iou,
                    "train_pixel_accuracy": report["pixel_accuracy"],
                    "known_prediction_coverage": (
                        float(np.mean(prediction != 0)) if len(prediction) else None
                    ),
                }
                grid_records.append(record)
                key = (
                    -np.inf if macro_iou is None else macro_iou,
                    -score_threshold,
                    -margin_threshold,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = (score_threshold, margin_threshold)
        assert best_thresholds is not None
        calibration.update({
            "objective": "train_primary_macro_iou",
            "score_threshold": best_thresholds[0],
            "margin_threshold": best_thresholds[1],
            "grid": grid_records,
        })
        thresholded = {}
        for split in ("train", "heldout"):
            reference, scores = arrays[split]
            prediction = predictions_from_scores(
                scores, class_ids, *best_thresholds
            )
            thresholded[split] = metric_report(
                reference,
                prediction,
                scores,
                classes,
                split_known_before_alpha[split],
                split_frames_with_pixels[split],
            )
            thresholded[split]["known_prediction_coverage"] = (
                float(np.mean(prediction != 0)) if len(prediction) else None
            )

    output = {
        "schema": "mcd_sparse_open_vocab_evaluation_v1",
        "score_kind": args.score_kind,
        "score_quantization": 65535,
        "aggregation": args.aggregation,
        "prompt_group_sha256": sha256(args.prompt_groups),
        "prompt_groups": prompt_document,
        "reference_source": str(args.reference_dir),
        "warning": "sparse projected LiDAR labels, not dense 2D ground truth",
        "frame_range": [args.start_frame, args.end_frame],
        "min_reference_confidence": args.min_reference_confidence,
        "split_rule": {
            "keyframe_period": args.keyframe_period,
            "keyframe_offset": args.keyframe_offset,
            "training_frames": sorted(training_frames),
            "training_label_scans": sorted(train_scans),
            "heldout_requires_disjoint_scan": True,
            "shared_scan_excluded_frames": excluded_shared_scan_frames,
            "unmatched_frames": unmatched_frames,
        },
        "excluded_reference_pixel_counts": excluded_reference_counts,
        "calibration": calibration,
        "train": threshold_free["train"],
        "heldout_disjoint_scan": threshold_free["heldout"],
        "threshold_free": threshold_free,
        "thresholded": thresholded,
        "per_frame": per_frame,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "train_pixels": threshold_free["train"]["reference_pixels"],
        "heldout_pixels": threshold_free["heldout"]["reference_pixels"],
        "heldout_frames": threshold_free["heldout"]["frames_with_pixels"],
        "heldout_accuracy": threshold_free["heldout"]["pixel_accuracy"],
        "heldout_macro_iou": threshold_free["heldout"]["macro_iou"],
        "heldout_macro_ap": threshold_free["heldout"]["macro_ap"],
        "shared_scan_excluded_frames": len(excluded_shared_scan_frames),
    }, indent=2))


if __name__ == "__main__":
    main()
