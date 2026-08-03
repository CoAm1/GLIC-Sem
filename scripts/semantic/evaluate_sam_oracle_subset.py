#!/usr/bin/env python3
"""Evaluate a SAM teacher subset without using labels to generate its masks.

This utility is for training-split hyperparameter selection.  It separates
reference pixels that SAM leaves uncovered from pixels that are covered but
cannot be recovered by a per-region majority-label oracle.  The latter is a
partition/merge diagnostic, not automatically proof that SAM is wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    return parser.parse_args()


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
    archives = [args.teacher_dir / f"{frame:06d}.npz" for frame in frames]
    missing = [str(path) for path in archives if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing teacher archives: {missing}")

    teacher_report_path = args.teacher_dir / "report.json"
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    reported_frames = {
        int(item["frame_index"])
        for item in teacher_report.get("frame_records", [])
    }
    if not set(frames).issubset(reported_frames):
        raise ValueError("requested frames are absent from teacher report")
    if teacher_report.get("reference_metrics"):
        raise ValueError("teacher report contains reference diagnostics")

    references = []
    predictions = []
    diagnostics = []
    valid_pixels = 0
    reference_pixels = 0
    for frame in frames:
        frame_name = f"{frame:06d}.png"
        reference = load_png(args.reference_dir / "labels" / frame_name, shape)
        confidence = load_png(
            args.reference_dir / "confidence" / frame_name, shape
        ).astype(np.float32) / 255.0
        reference_valid = np.isin(reference, class_ids) & (
            confidence >= args.min_reference_confidence
        )
        archive = np.load(args.teacher_dir / f"{frame:06d}.npz")
        segmentation = archive["segmentation"].astype(np.int32)
        if segmentation.shape != shape:
            raise ValueError(f"teacher segmentation shape mismatch: {frame_name}")
        prediction, sam_valid, diagnostic = sam_oracle(
            segmentation, reference, reference_valid, class_ids
        )
        diagnostic.update({"frame": frame, "split": "selection"})
        diagnostics.append(diagnostic)
        references.append(reference[reference_valid].astype(np.int64))
        predictions.append(prediction[reference_valid].astype(np.int64))
        valid_pixels += int(np.count_nonzero(reference_valid & sam_valid))
        reference_pixels += int(np.count_nonzero(reference_valid))

    output = {
        "schema": "sam_oracle_subset_diagnostic_v1",
        "guardrail": (
            "Reference labels are evaluator-only and were not inputs to SAM mask "
            "generation, OpenCLIP features, PCA fitting, or 3D optimization."
        ),
        "frames": frames,
        "minimum_reference_confidence": args.min_reference_confidence,
        "teacher_segmenter": teacher_report.get("segmenter"),
        "teacher_report_sha256": sha256(teacher_report_path),
        "teacher_archives_combined_sha256": combined_sha256(
            archives, args.teacher_dir
        ),
        "prompt_groups_sha256": sha256(args.prompt_groups),
        "prompt_groups": prompt_document,
        "metrics": metric_report(
            np.concatenate(references),
            np.concatenate(predictions),
            None,
            classes,
            valid_pixels=valid_pixels,
            reference_pixels=reference_pixels,
            frames_with_reference=len(frames),
        ),
        "sam_oracle_aggregate": aggregate_sam_oracle_diagnostics(
            diagnostics, classes
        )["selection"],
        "sam_oracle_per_frame": diagnostics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_json),
        "frames": len(frames),
        "macro_iou": output["metrics"]["macro_iou"],
        "coverage": output["metrics"]["stage_valid_coverage"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
