#!/usr/bin/env python3
"""Decompose open-vocabulary errors from SAM regions to a 3D Gaussian map.

The primary evaluation domain contains every confident sparse MCD reference
pixel. Missing SAM coverage or invalid 3D alpha is predicted as unknown and is
therefore penalized. A common-valid intersection is reported only as a
secondary diagnostic.

MCD labels are read exclusively by this evaluator. They must not be inputs to
the teacher, PCA fit, or mapper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PromptClass:
    class_id: int
    name: str
    queries: tuple[str, ...]


@dataclass
class StagePixels:
    prediction: np.ndarray
    scores: np.ndarray | None
    valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage3-score-root", type=Path)
    parser.add_argument("--stage3-alpha-root", type=Path)
    parser.add_argument("--start-frame", type=int, default=1800)
    parser.add_argument("--end-frame", type=int, default=1859)
    parser.add_argument("--keyframe-period", type=int, default=5)
    parser.add_argument("--keyframe-offset", type=int, default=4)
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-size", type=int, default=3)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: Iterable[Path], root: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        label = str(path.relative_to(root) if root is not None else path.name)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def cpp_safe_file_label(label: str) -> str:
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


def normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(
        np.linalg.norm(features, axis=-1, keepdims=True), 1e-12
    )


def read_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        header = source.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated matrix header: {path}")
        rows, columns = struct.unpack("<II", header)
        values = np.frombuffer(source.read(), dtype="<f4")
    if values.size != rows * columns:
        raise ValueError(f"invalid matrix payload: {path}")
    return values.reshape(rows, columns).copy()


def load_prompt_groups(path: Path) -> tuple[list[PromptClass], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") not in {
        "mcd_open_vocab_prompt_groups_v1",
        "mcd_open_vocab_prompt_groups_v2",
    }:
        raise ValueError("unexpected prompt-group schema")
    if document.get("frozen_before_heldout_evaluation") is not True:
        raise ValueError("prompt groups are not marked frozen")
    primary_ids = [int(value) for value in document["primary_macro_class_ids"]]
    classes: list[PromptClass] = []
    safe_labels: dict[str, str] = {}
    for class_id in primary_ids:
        record = document["classes"].get(str(class_id))
        if record is None:
            raise ValueError(f"missing prompt class {class_id}")
        queries = tuple(str(value) for value in record["queries"])
        if not queries:
            raise ValueError(f"prompt class {class_id} has no queries")
        for query in queries:
            safe = cpp_safe_file_label(query)
            previous = safe_labels.setdefault(safe, query)
            if previous != query:
                raise ValueError(
                    f"safe-label collision: {previous!r} and {query!r} -> {safe!r}"
                )
        classes.append(PromptClass(class_id, str(record["name"]), queries))
    if len({item.class_id for item in classes}) != len(classes):
        raise ValueError("duplicate primary class IDs")
    return classes, document


def load_query_features(
    path: Path, classes: Sequence[PromptClass]
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "pca_text_queries_v2":
        raise ValueError("query features must use pca_text_queries_v2")
    query_features: dict[str, np.ndarray] = {}
    for record in document["queries"]:
        label = str(record["label"])
        if label in query_features:
            raise ValueError(f"duplicate query feature: {label}")
        query_features[label] = normalize(
            np.asarray(record["feature"], dtype=np.float32)[None]
        )[0]
    required = {query for item in classes for query in item.queries}
    missing = sorted(required - set(query_features))
    if missing:
        raise ValueError(f"query feature JSON is missing prompts: {missing}")
    negatives = normalize(np.asarray(
        [record["feature"] for record in document["negatives"]],
        dtype=np.float32,
    ))
    if negatives.ndim != 2 or negatives.shape[0] == 0:
        raise ValueError("at least one negative text feature is required")
    dimensions = {len(value) for value in query_features.values()}
    dimensions.add(negatives.shape[1])
    if len(dimensions) != 1:
        raise ValueError("text feature dimensions do not agree")
    return query_features, negatives, document


def load_alignment(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"frame_index", "label_scan_index"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("alignment CSV lacks required columns")
        for row in reader:
            frame = int(row["frame_index"])
            if frame in result:
                raise ValueError(f"duplicate alignment frame: {frame}")
            result[frame] = int(row["label_scan_index"])
    return result


def make_split(
    frames: Sequence[int],
    alignment: dict[int, int],
    keyframe_period: int,
    keyframe_offset: int,
) -> tuple[dict[int, str], dict]:
    missing = [frame for frame in frames if frame not in alignment]
    if missing:
        raise ValueError(f"missing alignment rows: {missing[:8]}")
    training = {
        frame for frame in frames
        if frame % keyframe_period == keyframe_offset
    }
    train_scans = {alignment[frame] for frame in training if alignment[frame] >= 0}
    split: dict[int, str] = {}
    shared = []
    unmatched = []
    for frame in frames:
        scan = alignment[frame]
        if frame in training:
            split[frame] = "train"
        elif scan < 0:
            split[frame] = "unmatched"
            unmatched.append(frame)
        elif scan in train_scans:
            split[frame] = "shared_scan_excluded"
            shared.append(frame)
        else:
            split[frame] = "heldout"
    return split, {
        "keyframe_period": keyframe_period,
        "keyframe_offset": keyframe_offset,
        "training_frames": sorted(training),
        "training_label_scans": sorted(train_scans),
        "shared_scan_excluded_frames": shared,
        "unmatched_frames": unmatched,
        "heldout_requires_disjoint_scan": True,
    }


def load_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != shape:
        raise ValueError(f"{path}: expected shape {shape}, got {image.shape}")
    return image


def relevancy(
    features: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> np.ndarray:
    """Match LangSplat's minimum positive probability over negatives."""
    positive_score = features @ positive.T
    negative_score = features @ negative.T
    delta = positive_score[:, :, None] - negative_score[:, None, :]
    probability = 1.0 / (1.0 + np.exp(np.clip(-10.0 * delta, -80.0, 80.0)))
    return probability.min(axis=2)


def class_region_scores(
    features: np.ndarray,
    classes: Sequence[PromptClass],
    query_features: dict[str, np.ndarray],
    negatives: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    ordered_queries = [query for item in classes for query in item.queries]
    positive = np.stack([query_features[query] for query in ordered_queries])
    query_scores = relevancy(features, positive, negatives)
    class_scores = []
    offset = 0
    for item in classes:
        selected = query_scores[:, offset : offset + len(item.queries)]
        class_scores.append(
            selected.max(axis=1) if aggregation == "max" else selected.mean(axis=1)
        )
        offset += len(item.queries)
    return np.stack(class_scores, axis=1).astype(np.float32)


def region_scores_to_pixels(
    region_scores: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if segmentation.ndim != 2:
        raise ValueError("segmentation must be two-dimensional")
    valid = segmentation >= 0
    if valid.any() and int(segmentation[valid].max()) >= len(region_scores):
        raise ValueError("segmentation references a missing region feature")
    scores = np.zeros((*segmentation.shape, region_scores.shape[1]), dtype=np.float32)
    scores[valid] = region_scores[segmentation[valid]]
    return scores, valid


def prediction_from_scores(
    scores: np.ndarray,
    class_ids: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    if scores.shape[:2] != valid.shape or scores.shape[2] != len(class_ids):
        raise ValueError("score/valid/class shapes do not agree")
    prediction = np.zeros(valid.shape, dtype=np.int16)
    prediction[valid] = class_ids[np.argmax(scores[valid], axis=1)]
    return prediction


def sam_oracle(
    segmentation: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
    class_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    prediction = np.zeros(segmentation.shape, dtype=np.int16)
    sam_valid = segmentation >= 0
    purities = []
    labelled_pixels = 0
    assigned_regions = 0
    assigned_class_counts = {str(int(class_id)): 0 for class_id in class_ids}
    region_ids = np.unique(segmentation[sam_valid])
    for region_id in region_ids:
        pixels = (segmentation == region_id) & reference_valid
        values = reference[pixels]
        if values.size == 0:
            continue
        counts = np.asarray([np.count_nonzero(values == value) for value in class_ids])
        target = int(class_ids[int(np.argmax(counts))])
        prediction[segmentation == region_id] = target
        maximum = int(counts.max())
        purities.append(maximum / int(values.size))
        labelled_pixels += int(values.size)
        assigned_regions += 1
        assigned_class_counts[str(target)] += 1
    weighted_correct = int(np.count_nonzero(
        (prediction == reference) & reference_valid & sam_valid
    ))
    per_class = {}
    for class_id in class_ids:
        class_id_int = int(class_id)
        class_reference = reference_valid & (reference == class_id_int)
        reference_pixels = int(np.count_nonzero(class_reference))
        covered_reference_pixels = int(np.count_nonzero(class_reference & sam_valid))
        oracle_correct_pixels = int(np.count_nonzero(
            class_reference & sam_valid & (prediction == class_id_int)
        ))
        per_class[str(class_id_int)] = {
            "reference_pixels": reference_pixels,
            "sam_covered_reference_pixels": covered_reference_pixels,
            "sam_reference_coverage": (
                covered_reference_pixels / reference_pixels
                if reference_pixels else None
            ),
            "oracle_correct_reference_pixels": oracle_correct_pixels,
            "oracle_recall_on_all_reference": (
                oracle_correct_pixels / reference_pixels
                if reference_pixels else None
            ),
            "oracle_accuracy_on_sam_covered_reference": (
                oracle_correct_pixels / covered_reference_pixels
                if covered_reference_pixels else None
            ),
        }
    diagnostic = {
        "sam_pixel_coverage_on_reference": (
            float(np.mean(sam_valid[reference_valid]))
            if reference_valid.any() else None
        ),
        "assigned_regions": assigned_regions,
        "mean_region_purity": float(np.mean(purities)) if purities else None,
        "pixel_weighted_region_purity": (
            weighted_correct / labelled_pixels if labelled_pixels else None
        ),
        "assigned_region_count_by_class": assigned_class_counts,
        "per_class": per_class,
    }
    return prediction, sam_valid, diagnostic


def aggregate_sam_oracle_diagnostics(
    diagnostics: Sequence[dict], classes: Sequence[PromptClass]
) -> dict:
    grouped: dict[str, list[dict]] = {"all": list(diagnostics)}
    for diagnostic in diagnostics:
        split = str(diagnostic.get("split", "unspecified"))
        grouped.setdefault(split, []).append(diagnostic)

    report = {}
    for split, items in grouped.items():
        per_class = {}
        for item in classes:
            class_key = str(item.class_id)
            reference_pixels = sum(
                int(frame["per_class"][class_key]["reference_pixels"])
                for frame in items
            )
            covered_pixels = sum(
                int(frame["per_class"][class_key]["sam_covered_reference_pixels"])
                for frame in items
            )
            correct_pixels = sum(
                int(frame["per_class"][class_key]["oracle_correct_reference_pixels"])
                for frame in items
            )
            assigned_regions = sum(
                int(frame["assigned_region_count_by_class"][class_key])
                for frame in items
            )
            uncovered_pixels = reference_pixels - covered_pixels
            covered_but_not_correct = covered_pixels - correct_pixels
            per_class[class_key] = {
                "name": item.name,
                "reference_pixels": reference_pixels,
                "sam_covered_reference_pixels": covered_pixels,
                "oracle_correct_reference_pixels": correct_pixels,
                "uncovered_reference_pixels": uncovered_pixels,
                "covered_but_not_oracle_correct_pixels": covered_but_not_correct,
                "sam_reference_coverage": (
                    covered_pixels / reference_pixels if reference_pixels else None
                ),
                "oracle_accuracy_on_sam_covered_reference": (
                    correct_pixels / covered_pixels if covered_pixels else None
                ),
                "oracle_recall_on_all_reference": (
                    correct_pixels / reference_pixels if reference_pixels else None
                ),
                "uncovered_fraction_of_reference": (
                    uncovered_pixels / reference_pixels if reference_pixels else None
                ),
                "covered_but_not_oracle_correct_fraction_of_reference": (
                    covered_but_not_correct / reference_pixels
                    if reference_pixels else None
                ),
                "assigned_regions": assigned_regions,
            }
        report[split] = {
            "frames": len(items),
            "per_class": per_class,
            "interpretation_guardrail": (
                "Covered-but-not-oracle-correct is a partition/merge candidate, "
                "not proof of a SAM error; sparse-label projection or timing errors "
                "can produce the same symptom."
            ),
        }
    return report


def load_stage3_scores(
    score_root: Path,
    alpha_root: Path,
    frame: int,
    shape: tuple[int, int],
    classes: Sequence[PromptClass],
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame_name = f"{frame:06d}.png"
    class_scores = []
    for item in classes:
        query_scores = []
        for query in item.queries:
            image = load_png(
                score_root / cpp_safe_file_label(query) / frame_name, shape
            )
            if image.dtype != np.uint16:
                raise TypeError(f"3D score must be uint16: {query}/{frame_name}")
            query_scores.append(image.astype(np.float32) / 65535.0)
        stacked = np.stack(query_scores)
        class_scores.append(
            stacked.max(axis=0) if aggregation == "max" else stacked.mean(axis=0)
        )
    alpha = load_png(alpha_root / frame_name, shape)
    if alpha.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"3D alpha must be uint8/uint16: {frame_name}")
    return np.stack(class_scores, axis=-1), alpha > 0


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    true_cumulative = np.cumsum(sorted_labels)
    false_cumulative = np.cumsum(1 - sorted_labels)
    group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_at_threshold = true_cumulative[group_end]
    false_at_threshold = false_cumulative[group_end]
    precision = true_at_threshold / np.maximum(
        true_at_threshold + false_at_threshold, 1
    )
    recall = true_at_threshold / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion_from_predictions(
    reference: np.ndarray,
    prediction: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    row_for_id = {int(value): index for index, value in enumerate(class_ids)}
    column_ids = np.r_[0, class_ids]
    column_for_id = {int(value): index for index, value in enumerate(column_ids)}
    confusion = np.zeros((len(class_ids), len(class_ids) + 1), dtype=np.int64)
    for true_value, predicted_value in zip(reference, prediction):
        row = row_for_id[int(true_value)]
        column = column_for_id.get(int(predicted_value), 0)
        confusion[row, column] += 1
    return confusion


def scalar_metrics_from_confusion(
    confusion: np.ndarray,
    valid_pixels: int,
    reference_pixels: int,
) -> dict:
    if confusion.shape[1] != confusion.shape[0] + 1:
        raise ValueError("confusion must include an unknown prediction column")
    ious = []
    recalls = []
    correct = 0
    for index in range(confusion.shape[0]):
        true_positive = int(confusion[index, index + 1])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index + 1].sum())
        union = support + predicted - true_positive
        # A class with no reference positives in this evaluation split has no
        # defined IoU.  False-positive predictions still affect pixel
        # accuracy and the confusion matrix, but must not manufacture a zero
        # entry in the macro-IoU average.
        if support:
            ious.append(true_positive / union)
        if support:
            recalls.append(true_positive / support)
        correct += true_positive
    total = int(confusion.sum())
    return {
        "pixel_accuracy": correct / total if total else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_iou": float(np.mean(ious)) if ious else None,
        "prediction_coverage": (
            1.0 - int(confusion[:, 0].sum()) / total if total else None
        ),
        "stage_valid_coverage": (
            valid_pixels / reference_pixels if reference_pixels else None
        ),
    }


def metric_report(
    reference: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray | None,
    classes: Sequence[PromptClass],
    valid_pixels: int,
    reference_pixels: int,
    frames_with_reference: int,
) -> dict:
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    confusion = confusion_from_predictions(reference, prediction, class_ids)
    scalar = scalar_metrics_from_confusion(confusion, valid_pixels, reference_pixels)
    per_class = {}
    ious = []
    aps = []
    for index, item in enumerate(classes):
        true_positive = int(confusion[index, index + 1])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index + 1].sum())
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        union = true_positive + false_positive + false_negative
        iou = true_positive / union if support else None
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        ap = (
            average_precision(reference == item.class_id, scores[:, index])
            if scores is not None else None
        )
        if iou is not None:
            ious.append(iou)
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
        }
    return {
        "reference_pixels": int(reference_pixels),
        "evaluated_pixels": int(len(reference)),
        "frames_with_reference": frames_with_reference,
        **scalar,
        "macro_ap": float(np.mean(aps)) if aps else None,
        "macro_ap_class_count": len(aps),
        "macro_iou_class_count": len(ious),
        "confusion_row_class_ids": class_ids.tolist(),
        "confusion_column_class_ids": np.r_[0, class_ids].tolist(),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def concatenate(parts: list[np.ndarray], columns: int | None = None) -> np.ndarray:
    if parts:
        return np.concatenate(parts, axis=0)
    if columns is None:
        return np.empty((0,), dtype=np.float32)
    return np.empty((0, columns), dtype=np.float32)


def quantile_interval(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "bootstrap_mean": float(array.mean()),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def temporal_block_bootstrap(
    frame_confusions: dict[int, dict[str, np.ndarray]],
    frame_coverages: dict[int, dict[str, tuple[int, int]]],
    stage_order: Sequence[str],
    replicates: int,
    block_size: int,
    seed: int,
    frame_groups: dict[int, int] | None = None,
) -> dict:
    frames = sorted(frame_confusions)
    if not frames or replicates <= 0:
        return {"enabled": False, "reason": "no frames or zero replicates"}
    if frame_groups is None:
        blocks = [
            frames[index : index + block_size]
            for index in range(0, len(frames), block_size)
        ]
        resampling_unit = "contiguous_frame_block"
        group_ids = None
    else:
        missing_groups = [frame for frame in frames if frame not in frame_groups]
        if missing_groups:
            raise ValueError(f"missing bootstrap groups: {missing_groups[:8]}")
        grouped_frames: dict[int, list[int]] = {}
        for frame in frames:
            grouped_frames.setdefault(int(frame_groups[frame]), []).append(frame)
        group_ids = sorted(grouped_frames)
        blocks = [grouped_frames[group_id] for group_id in group_ids]
        resampling_unit = "label_scan_index"
    generator = np.random.default_rng(seed)
    samples = {
        stage: {metric: [] for metric in (
            "macro_iou", "balanced_accuracy", "pixel_accuracy", "stage_valid_coverage"
        )}
        for stage in stage_order
    }
    drops = {"delta_clip": [], "delta_pca": []}
    if "3d_headonly" in stage_order:
        drops["delta_3d"] = []
    for _ in range(replicates):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        replicate_metrics = {}
        for stage in stage_order:
            confusion = None
            valid = 0
            reference = 0
            for block_index in selected:
                for frame in blocks[int(block_index)]:
                    current = frame_confusions[frame][stage]
                    confusion = current.copy() if confusion is None else confusion + current
                    stage_valid, stage_reference = frame_coverages[frame][stage]
                    valid += stage_valid
                    reference += stage_reference
            assert confusion is not None
            scalar = scalar_metrics_from_confusion(confusion, valid, reference)
            replicate_metrics[stage] = scalar
            for metric in samples[stage]:
                value = scalar[metric]
                if value is not None:
                    samples[stage][metric].append(value)
        drops["delta_clip"].append(
            replicate_metrics["sam_oracle"]["macro_iou"]
            - replicate_metrics["full_teacher"]["macro_iou"]
        )
        drops["delta_pca"].append(
            replicate_metrics["full_teacher"]["macro_iou"]
            - replicate_metrics["pca_teacher"]["macro_iou"]
        )
        if "3d_headonly" in stage_order:
            drops["delta_3d"].append(
                replicate_metrics["pca_teacher"]["macro_iou"]
                - replicate_metrics["3d_headonly"]["macro_iou"]
            )
    return {
        "enabled": True,
        "replicates": replicates,
        "resampling_unit": resampling_unit,
        "block_size_frames": block_size if frame_groups is None else None,
        "block_count": len(blocks),
        "group_ids": group_ids,
        "frames_per_group": [len(block) for block in blocks],
        "seed": seed,
        "stages": {
            stage: {
                metric: quantile_interval(values)
                for metric, values in metrics.items()
            }
            for stage, metrics in samples.items()
        },
        "paired_stage_drops": {
            name: quantile_interval(values) for name, values in drops.items()
        },
        "guardrail": (
            "LiDAR label-scan groups, not pixels or projected camera frames, "
            "are the resampling unit."
            if frame_groups is not None else
            "Contiguous temporal frame blocks, not pixels, are the resampling unit."
        ),
    }


def preflight(
    args: argparse.Namespace,
    frames: Sequence[int],
    teacher_report: dict,
) -> dict:
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    archives = [args.teacher_dir / f"{frame:06d}.npz" for frame in frames]
    missing_archives = [str(path) for path in archives if not path.exists()]
    record(
        "teacher_archives_complete",
        not missing_archives,
        f"expected={len(archives)} missing={len(missing_archives)}",
    )
    report_frames = {
        int(item["frame_index"]) for item in teacher_report.get("frame_records", [])
    }
    record(
        "teacher_report_frame_set_exact",
        report_frames == set(frames),
        f"report={len(report_frames)} expected={len(frames)}",
    )
    reference_metrics = teacher_report.get("reference_metrics", {})
    record(
        "teacher_has_no_reference_diagnostics",
        not bool(reference_metrics),
        "reference_metrics must be empty for the formal no-leakage run",
    )
    segmenter = teacher_report.get("segmenter")
    checkpoint = str(teacher_report.get("sam_checkpoint", ""))
    sam1_identified = (
        isinstance(segmenter, dict) and segmenter.get("version") == "SAM1"
    ) or "sam_vit_h" in checkpoint
    record(
        "segmenter_identified_as_sam1",
        sam1_identified,
        "legacy report accepted from checkpoint name; new reports must use structured metadata",
    )
    legacy_warning = "SAM3" in str(teacher_report.get("warning", ""))
    record(
        "teacher_warning_metadata_correct",
        not legacy_warning,
        "legacy SAM1 report incorrectly mentions SAM3" if legacy_warning else "ok",
    )
    basis_shape = read_matrix(args.basis_dir / "basis.f32").shape
    mean_shape = read_matrix(args.basis_dir / "mean.f32").shape
    record(
        "pca_shape_512x128",
        basis_shape == (512, 128) and mean_shape == (1, 512),
        f"basis={basis_shape} mean={mean_shape}",
    )
    if args.stage3_score_root is None:
        record("stage3_inputs_paired", args.stage3_alpha_root is None, "2D-only run")
    else:
        record(
            "stage3_inputs_paired",
            args.stage3_alpha_root is not None,
            "score and alpha roots must be provided together",
        )
    hard_fail_names = {
        "teacher_archives_complete",
        "teacher_report_frame_set_exact",
        "teacher_has_no_reference_diagnostics",
        "segmenter_identified_as_sam1",
        "pca_shape_512x128",
        "stage3_inputs_paired",
    }
    passed = all(
        item["passed"] for item in checks if item["name"] in hard_fail_names
    )
    return {
        "passed": passed,
        "checks": checks,
        "note": "Legacy warning text is recorded but is not a hard failure for existing artifacts.",
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.start_frame > args.end_frame:
        raise ValueError("start-frame must be <= end-frame")
    if args.keyframe_period < 1 or not 0 <= args.keyframe_offset < args.keyframe_period:
        raise ValueError("invalid keyframe rule")
    if not 0.0 <= args.min_reference_confidence <= 1.0:
        raise ValueError("min-reference-confidence must be in [0, 1]")
    if args.bootstrap_replicates < 0 or args.bootstrap_block_size < 1:
        raise ValueError("invalid bootstrap settings")
    if (args.stage3_score_root is None) != (args.stage3_alpha_root is None):
        raise ValueError("stage3 score and alpha roots must be supplied together")

    shape = (args.image_height, args.image_width)
    frames = list(range(args.start_frame, args.end_frame + 1))
    classes, prompt_document = load_prompt_groups(args.prompt_groups)
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    query_features, negatives, query_document = load_query_features(
        args.query_features, classes
    )
    mean = read_matrix(args.basis_dir / "mean.f32")
    basis = read_matrix(args.basis_dir / "basis.f32")
    if mean.shape != (1, 512) or basis.shape[0] != 512:
        raise ValueError("PCA teacher dimension must be 512")
    teacher_report_path = args.teacher_dir / "report.json"
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    alignment = load_alignment(args.alignment_csv)
    split_for_frame, split_report = make_split(
        frames, alignment, args.keyframe_period, args.keyframe_offset
    )
    gate_a = preflight(args, frames, teacher_report)

    args.output_dir.mkdir(parents=True)
    provenance = {
        "teacher_report_sha256": sha256(teacher_report_path),
        "teacher_archives_combined_sha256": combined_sha256(
            [args.teacher_dir / f"{frame:06d}.npz" for frame in frames],
            args.teacher_dir,
        ) if gate_a["checks"][0]["passed"] else None,
        "pca_mean_sha256": sha256(args.basis_dir / "mean.f32"),
        "pca_basis_sha256": sha256(args.basis_dir / "basis.f32"),
        "query_features_sha256": sha256(args.query_features),
        "prompt_groups_sha256": sha256(args.prompt_groups),
        "alignment_sha256": sha256(args.alignment_csv),
        "reference_metadata_sha256": (
            sha256(args.reference_dir / "metadata.json")
            if (args.reference_dir / "metadata.json").exists() else None
        ),
    }
    preflight_report = {
        "schema": "semantic_error_decomposition_preflight_v1",
        "frame_range": [args.start_frame, args.end_frame],
        "gate_a": gate_a,
        "split": split_report,
        "provenance": provenance,
    }
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not gate_a["passed"]:
        raise RuntimeError(f"Gate A hard failure; inspect {args.output_dir / 'preflight.json'}")
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
        return

    stage_order = ["sam_oracle", "full_teacher", "pca_teacher"]
    if args.stage3_score_root is not None:
        stage_order.append("3d_headonly")
    domains = ("full_reference", "common_valid")
    accumulators = {
        split: {
            domain: {
                stage: {
                    "reference": [],
                    "prediction": [],
                    "scores": [],
                    "valid_pixels": 0,
                    "reference_pixels": 0,
                    "frames": 0,
                }
                for stage in stage_order
            }
            for domain in domains
        }
        for split in ("train", "heldout")
    }
    oracle_diagnostics = []
    per_frame_rows = []
    heldout_confusions: dict[int, dict[str, np.ndarray]] = {}
    heldout_coverages: dict[int, dict[str, tuple[int, int]]] = {}

    for frame in frames:
        split = split_for_frame[frame]
        if split not in ("train", "heldout"):
            continue
        frame_name = f"{frame:06d}.png"
        reference = load_png(args.reference_dir / "labels" / frame_name, shape)
        confidence = load_png(
            args.reference_dir / "confidence" / frame_name, shape
        ).astype(np.float32) / 255.0
        if reference.dtype != np.uint8:
            raise TypeError(f"reference label must be uint8: {frame_name}")
        reference_valid = np.isin(reference, class_ids) & (
            confidence >= args.min_reference_confidence
        )

        archive = np.load(args.teacher_dir / f"{frame:06d}.npz")
        full_features = normalize(archive["features"].astype(np.float32))
        segmentation = archive["segmentation"].astype(np.int32)
        if segmentation.shape != shape:
            raise ValueError(f"teacher segmentation shape mismatch: {frame_name}")
        full_region_scores = class_region_scores(
            full_features, classes, query_features, negatives, args.aggregation
        )
        full_scores, sam_valid = region_scores_to_pixels(
            full_region_scores, segmentation
        )
        full_prediction = prediction_from_scores(full_scores, class_ids, sam_valid)

        reconstructed = normalize(mean + ((full_features - mean) @ basis) @ basis.T)
        pca_region_scores = class_region_scores(
            reconstructed, classes, query_features, negatives, args.aggregation
        )
        pca_scores, pca_valid = region_scores_to_pixels(
            pca_region_scores, segmentation
        )
        if not np.array_equal(sam_valid, pca_valid):
            raise RuntimeError("full and PCA teacher validity differ")
        pca_prediction = prediction_from_scores(pca_scores, class_ids, pca_valid)

        oracle_prediction, oracle_valid, oracle_diag = sam_oracle(
            segmentation, reference, reference_valid, class_ids
        )
        oracle_diag.update({"frame": frame, "split": split})
        oracle_diagnostics.append(oracle_diag)

        stages = {
            "sam_oracle": StagePixels(oracle_prediction, None, oracle_valid),
            "full_teacher": StagePixels(full_prediction, full_scores, sam_valid),
            "pca_teacher": StagePixels(pca_prediction, pca_scores, pca_valid),
        }
        if args.stage3_score_root is not None and args.stage3_alpha_root is not None:
            stage3_scores, stage3_valid = load_stage3_scores(
                args.stage3_score_root,
                args.stage3_alpha_root,
                frame,
                shape,
                classes,
                args.aggregation,
            )
            stages["3d_headonly"] = StagePixels(
                prediction_from_scores(stage3_scores, class_ids, stage3_valid),
                stage3_scores,
                stage3_valid,
            )

        common_valid = reference_valid.copy()
        for stage in stages.values():
            common_valid &= stage.valid
        domain_masks = {
            "full_reference": reference_valid,
            "common_valid": common_valid,
        }
        if split == "heldout":
            heldout_confusions[frame] = {}
            heldout_coverages[frame] = {}

        for domain, domain_mask in domain_masks.items():
            domain_reference = reference[domain_mask].astype(np.int64)
            for stage_name, stage in stages.items():
                accumulator = accumulators[split][domain][stage_name]
                domain_prediction = stage.prediction[domain_mask].astype(np.int64)
                accumulator["reference"].append(domain_reference)
                accumulator["prediction"].append(domain_prediction)
                if stage.scores is not None:
                    accumulator["scores"].append(
                        stage.scores[domain_mask].astype(np.float32)
                    )
                accumulator["valid_pixels"] += int(
                    np.count_nonzero(stage.valid & domain_mask)
                )
                accumulator["reference_pixels"] += int(domain_mask.sum())
                accumulator["frames"] += int(domain_mask.any())

                if domain == "full_reference":
                    confusion = confusion_from_predictions(
                        domain_reference, domain_prediction, class_ids
                    )
                    if split == "heldout":
                        heldout_confusions[frame][stage_name] = confusion
                        heldout_coverages[frame][stage_name] = (
                            int(np.count_nonzero(stage.valid & domain_mask)),
                            int(domain_mask.sum()),
                        )
                    scalar = scalar_metrics_from_confusion(
                        confusion,
                        int(np.count_nonzero(stage.valid & domain_mask)),
                        int(domain_mask.sum()),
                    )
                    per_frame_rows.append({
                        "frame": frame,
                        "label_scan_index": alignment[frame],
                        "split": split,
                        "stage": stage_name,
                        "reference_pixels": int(domain_mask.sum()),
                        **scalar,
                    })

    reports = {split: {} for split in ("train", "heldout")}
    for split in reports:
        for domain in domains:
            reports[split][domain] = {}
            for stage_name in stage_order:
                accumulator = accumulators[split][domain][stage_name]
                reference = concatenate(accumulator["reference"]).astype(np.int64)
                prediction = concatenate(accumulator["prediction"]).astype(np.int64)
                score_parts = accumulator["scores"]
                scores = (
                    concatenate(score_parts, len(classes)).astype(np.float32)
                    if score_parts else None
                )
                reports[split][domain][stage_name] = metric_report(
                    reference,
                    prediction,
                    scores,
                    classes,
                    accumulator["valid_pixels"],
                    accumulator["reference_pixels"],
                    accumulator["frames"],
                )

    heldout_main = reports["heldout"]["full_reference"]
    stage_drops = {
        "delta_clip": (
            heldout_main["sam_oracle"]["macro_iou"]
            - heldout_main["full_teacher"]["macro_iou"]
        ),
        "delta_pca": (
            heldout_main["full_teacher"]["macro_iou"]
            - heldout_main["pca_teacher"]["macro_iou"]
        ),
    }
    if "3d_headonly" in stage_order:
        stage_drops["delta_3d"] = (
            heldout_main["pca_teacher"]["macro_iou"]
            - heldout_main["3d_headonly"]["macro_iou"]
        )

    bootstrap = temporal_block_bootstrap(
        heldout_confusions,
        heldout_coverages,
        stage_order,
        args.bootstrap_replicates,
        args.bootstrap_block_size,
        args.bootstrap_seed,
        frame_groups={frame: alignment[frame] for frame in heldout_confusions},
    )
    output = {
        "schema": "open_vocab_semantic_error_decomposition_v1",
        "warning": "MCD projected LiDAR labels are sparse reference, not dense 2D ground truth.",
        "primary_domain": (
            "All confident reference pixels; missing SAM coverage or 3D alpha is unknown and penalized."
        ),
        "secondary_domain": "Common-valid intersection; diagnostic only.",
        "frame_range": [args.start_frame, args.end_frame],
        "minimum_reference_confidence": args.min_reference_confidence,
        "aggregation": args.aggregation,
        "stage_order": stage_order,
        "split": split_report,
        "gate_a": gate_a,
        "provenance": provenance,
        "prompt_groups": prompt_document,
        "query_metadata": {
            "schema": query_document.get("schema"),
            "model": query_document.get("model"),
            "pretrained": query_document.get("pretrained"),
            "prompt_template": query_document.get("prompt_template"),
            "negative_prompts": [item.get("prompt") for item in query_document["negatives"]],
        },
        "reports": reports,
        "heldout_full_reference_stage_drops": stage_drops,
        "heldout_temporal_block_bootstrap": bootstrap,
        "sam_oracle_per_frame": oracle_diagnostics,
        "sam_oracle_by_split": aggregate_sam_oracle_diagnostics(
            oracle_diagnostics, classes
        ),
        "guardrails": [
            "SAM Oracle uses evaluation labels and is not a deployable method result.",
            "MCD labels were read only inside this evaluator.",
            "No per-frame min-max normalization or spatial smoothing was applied.",
            "Unknown/invalid pixels are errors in the primary domain.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "frame", "label_scan_index", "split", "stage", "reference_pixels",
            "pixel_accuracy", "balanced_accuracy", "macro_iou",
            "prediction_coverage", "stage_valid_coverage",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_rows)
    print(json.dumps({
        "output": str(args.output_dir / "summary.json"),
        "stages": stage_order,
        "heldout_frames": len(heldout_confusions),
        "heldout_macro_iou": {
            stage: heldout_main[stage]["macro_iou"] for stage in stage_order
        },
        "stage_drops": stage_drops,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
