#!/usr/bin/env python3
"""Evaluate LangSplatV2 SAM1 levels on a label-isolated selection subset.

Each level receives the same per-region majority-label oracle used by the
single-level Gate-B diagnostic.  A joint-partition oracle is also reported: a
pixel belongs to a cell identified by its tuple of region IDs across all four
levels, and each cell receives one majority label.  This is an evaluator-only
upper bound for information present across levels, not a deployable fusion
rule and not evidence that a learned 3D map will reach the same accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluate_semantic_error_decomposition import (
    aggregate_sam_oracle_diagnostics,
    combined_sha256,
    load_png,
    load_prompt_groups,
    metric_report,
    sam_oracle,
    sha256,
)


LEVELS = ("default", "s", "m", "l")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    return parser.parse_args()


def build_joint_partition(segmentations: Sequence[np.ndarray]) -> np.ndarray:
    """Create a deterministic region map from cross-level region-ID tuples."""
    if not segmentations:
        raise ValueError("at least one segmentation is required")
    shape = segmentations[0].shape
    if any(item.shape != shape for item in segmentations):
        raise ValueError("segmentation shapes do not agree")
    stacked = np.stack(
        [np.asarray(item, dtype=np.int64) for item in segmentations], axis=-1
    )
    covered = np.any(stacked >= 0, axis=-1)
    joint = np.full(shape, -1, dtype=np.int32)
    if covered.any():
        _, inverse = np.unique(stacked[covered], axis=0, return_inverse=True)
        joint[covered] = inverse.astype(np.int32)
    return joint


def any_level_recoverability(
    reference: np.ndarray,
    reference_valid: np.ndarray,
    predictions: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    class_ids: np.ndarray,
) -> dict:
    if not predictions or len(predictions) != len(valid_masks):
        raise ValueError("prediction/valid level counts do not agree")
    any_covered = np.logical_or.reduce(valid_masks)
    correct_by_level = [
        valid & (prediction == reference)
        for prediction, valid in zip(predictions, valid_masks)
    ]
    any_correct = np.logical_or.reduce(correct_by_level) & reference_valid
    per_class = {}
    for class_id in class_ids:
        class_reference = reference_valid & (reference == int(class_id))
        reference_pixels = int(np.count_nonzero(class_reference))
        covered_pixels = int(np.count_nonzero(class_reference & any_covered))
        recoverable_pixels = int(np.count_nonzero(class_reference & any_correct))
        per_class[str(int(class_id))] = {
            "reference_pixels": reference_pixels,
            "covered_by_any_level": covered_pixels,
            "oracle_correct_in_at_least_one_level": recoverable_pixels,
        }
    return {
        "reference_pixels": int(np.count_nonzero(reference_valid)),
        "covered_by_any_level": int(np.count_nonzero(reference_valid & any_covered)),
        "oracle_correct_in_at_least_one_level": int(np.count_nonzero(any_correct)),
        "per_class": per_class,
    }


def aggregate_any_level(records: Sequence[dict], class_names: dict[str, str]) -> dict:
    reference_pixels = sum(int(item["reference_pixels"]) for item in records)
    covered_pixels = sum(int(item["covered_by_any_level"]) for item in records)
    recoverable_pixels = sum(
        int(item["oracle_correct_in_at_least_one_level"]) for item in records
    )
    per_class = {}
    for class_key, class_name in class_names.items():
        reference = sum(
            int(item["per_class"][class_key]["reference_pixels"])
            for item in records
        )
        covered = sum(
            int(item["per_class"][class_key]["covered_by_any_level"])
            for item in records
        )
        recoverable = sum(
            int(
                item["per_class"][class_key][
                    "oracle_correct_in_at_least_one_level"
                ]
            )
            for item in records
        )
        per_class[class_key] = {
            "name": class_name,
            "reference_pixels": reference,
            "covered_by_any_level": covered,
            "oracle_correct_in_at_least_one_level": recoverable,
            "any_level_coverage": covered / reference if reference else None,
            "any_level_oracle_recall": (
                recoverable / reference if reference else None
            ),
        }
    return {
        "reference_pixels": reference_pixels,
        "covered_by_any_level": covered_pixels,
        "oracle_correct_in_at_least_one_level": recoverable_pixels,
        "any_level_coverage": (
            covered_pixels / reference_pixels if reference_pixels else None
        ),
        "any_level_oracle_recall": (
            recoverable_pixels / reference_pixels if reference_pixels else None
        ),
        "per_class": per_class,
        "interpretation_guardrail": (
            "This statistic uses reference labels to ask whether any level is "
            "correct for each pixel. It is not a coherent prediction and must "
            "not be reported as deployable accuracy or mIoU."
        ),
    }


def summarize_area_order(teacher_report: dict, frames: Sequence[int]) -> dict:
    by_frame = {
        int(item["frame_index"]): item
        for item in teacher_report.get("frame_records", [])
    }
    medians = {level: [] for level in ("s", "m", "l")}
    strict_order = []
    for frame in frames:
        record = by_frame.get(frame)
        if record is None:
            raise ValueError(f"teacher report lacks frame {frame}")
        values = []
        for level in ("s", "m", "l"):
            value = record["levels"][level]["area_pixels_median"]
            medians[level].append(value)
            values.append(value)
        strict_order.append(
            None not in values and values[0] <= values[1] <= values[2]
        )
    return {
        "median_area_pixels_across_frames": {
            level: float(np.median(values)) if values else None
            for level, values in medians.items()
        },
        "frames_with_s_m_l_nondecreasing_median_area": int(sum(strict_order)),
        "frames_tested": len(strict_order),
        "fraction_with_s_m_l_nondecreasing_median_area": (
            float(np.mean(strict_order)) if strict_order else None
        ),
        "interpretation": (
            "A low ordering fraction falsifies a literal small/medium/large area "
            "interpretation; the names should then be treated only as decoder "
            "candidate indices."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if len(set(args.frame_indices)) != len(args.frame_indices):
        raise ValueError("frame-indices must be unique")
    if not 0.0 <= args.min_reference_confidence <= 1.0:
        raise ValueError("min-reference-confidence must be in [0, 1]")

    frames = sorted(args.frame_indices)
    shape = (args.image_height, args.image_width)
    classes, prompt_document = load_prompt_groups(args.prompt_groups)
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    class_names = {str(item.class_id): item.name for item in classes}

    teacher_report_path = args.teacher_root / "report.json"
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    if teacher_report.get("schema") != "langsplatv2_multiscale_sam_regions_v1":
        raise ValueError("unexpected multiscale teacher schema")
    if teacher_report.get("frames") != frames:
        raise ValueError("teacher frames differ from requested selection frames")

    archives = [
        args.teacher_root / level / f"{frame:06d}.npz"
        for level in LEVELS
        for frame in frames
    ]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing teacher archives: {missing[:8]}")

    stage_names = (*LEVELS, "joint_partition")
    references = {stage: [] for stage in stage_names}
    predictions = {stage: [] for stage in stage_names}
    valid_pixels = {stage: 0 for stage in stage_names}
    reference_pixels = 0
    diagnostics = {stage: [] for stage in stage_names}
    any_level_records = []

    for frame in frames:
        frame_name = f"{frame:06d}.png"
        reference = load_png(args.reference_dir / "labels" / frame_name, shape)
        confidence = load_png(
            args.reference_dir / "confidence" / frame_name, shape
        ).astype(np.float32) / 255.0
        reference_valid = np.isin(reference, class_ids) & (
            confidence >= args.min_reference_confidence
        )
        reference_count = int(np.count_nonzero(reference_valid))
        reference_pixels += reference_count

        segmentations = []
        frame_predictions = []
        frame_valid_masks = []
        for level in LEVELS:
            with np.load(
                args.teacher_root / level / f"{frame:06d}.npz"
            ) as archive:
                segmentation = archive["segmentation"].astype(np.int32)
            if segmentation.shape != shape:
                raise ValueError(f"{level}/{frame_name}: shape mismatch")
            prediction, stage_valid, diagnostic = sam_oracle(
                segmentation, reference, reference_valid, class_ids
            )
            diagnostic.update({"frame": frame, "split": "selection"})
            segmentations.append(segmentation)
            frame_predictions.append(prediction)
            frame_valid_masks.append(stage_valid)
            references[level].append(reference[reference_valid].astype(np.int64))
            predictions[level].append(prediction[reference_valid].astype(np.int64))
            valid_pixels[level] += int(
                np.count_nonzero(reference_valid & stage_valid)
            )
            diagnostics[level].append(diagnostic)

        joint = build_joint_partition(segmentations)
        joint_prediction, joint_valid, joint_diagnostic = sam_oracle(
            joint, reference, reference_valid, class_ids
        )
        joint_diagnostic.update({"frame": frame, "split": "selection"})
        references["joint_partition"].append(
            reference[reference_valid].astype(np.int64)
        )
        predictions["joint_partition"].append(
            joint_prediction[reference_valid].astype(np.int64)
        )
        valid_pixels["joint_partition"] += int(
            np.count_nonzero(reference_valid & joint_valid)
        )
        diagnostics["joint_partition"].append(joint_diagnostic)

        any_level_records.append(any_level_recoverability(
            reference,
            reference_valid,
            frame_predictions,
            frame_valid_masks,
            class_ids,
        ))

    metrics = {}
    aggregate_diagnostics = {}
    for stage in stage_names:
        metrics[stage] = metric_report(
            np.concatenate(references[stage]),
            np.concatenate(predictions[stage]),
            None,
            classes,
            valid_pixels=valid_pixels[stage],
            reference_pixels=reference_pixels,
            frames_with_reference=len(frames),
        )
        aggregate_diagnostics[stage] = aggregate_sam_oracle_diagnostics(
            diagnostics[stage], classes
        )["selection"]

    output = {
        "schema": "langsplatv2_multiscale_sam_oracle_subset_v1",
        "guardrail": (
            "Reference labels are evaluator-only. Per-level and joint-partition "
            "oracles leak labels and are diagnostic upper bounds, not method "
            "predictions. Hyperparameters may be selected only on this declared "
            "selection subset before one untouched held-out evaluation."
        ),
        "frames": frames,
        "minimum_reference_confidence": args.min_reference_confidence,
        "teacher_report_sha256": sha256(teacher_report_path),
        "teacher_archives_combined_sha256": combined_sha256(
            archives, args.teacher_root
        ),
        "prompt_groups_sha256": sha256(args.prompt_groups),
        "prompt_groups": prompt_document,
        "teacher_segmenter": teacher_report.get("segmenter"),
        "decoder_area_order_audit": summarize_area_order(teacher_report, frames),
        "metrics": metrics,
        "sam_oracle_aggregate": aggregate_diagnostics,
        "any_level_label_leaking_recoverability": aggregate_any_level(
            any_level_records, class_names
        ),
        "per_frame": {
            stage: diagnostics[stage] for stage in stage_names
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_json),
        "frames": len(frames),
        "macro_iou": {
            stage: metrics[stage]["macro_iou"] for stage in stage_names
        },
        "any_level_oracle_recall": output[
            "any_level_label_leaking_recoverability"
        ]["any_level_oracle_recall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
