#!/usr/bin/env python3
"""Compare S0/S1/S2 on an explicit model-selection frame subset.

This evaluator is intended for training-side single-variable comparisons.  It
does not create a train/held-out split and must not be presented as final test
performance.  Reference labels remain evaluator-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_semantic_error_decomposition import (
    aggregate_sam_oracle_diagnostics,
    class_region_scores,
    combined_sha256,
    load_png,
    load_prompt_groups,
    load_query_features,
    metric_report,
    normalize,
    prediction_from_scores,
    read_matrix,
    region_scores_to_pixels,
    sam_oracle,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
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
    query_features, negatives, query_document = load_query_features(
        args.query_features, classes
    )
    mean = read_matrix(args.basis_dir / "mean.f32")
    basis = read_matrix(args.basis_dir / "basis.f32")
    if mean.shape != (1, 512) or basis.shape[0] != 512:
        raise ValueError("expected a 512-dimensional PCA teacher")

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

    stage_names = ("sam_oracle", "full_teacher", "pca_teacher")
    reference_parts = {name: [] for name in stage_names}
    prediction_parts = {name: [] for name in stage_names}
    score_parts: dict[str, list[np.ndarray]] = {
        "full_teacher": [],
        "pca_teacher": [],
    }
    valid_pixels = {name: 0 for name in stage_names}
    reference_pixels = 0
    oracle_diagnostics = []

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
        full_features = normalize(archive["features"].astype(np.float32))
        segmentation = archive["segmentation"].astype(np.int32)
        if segmentation.shape != shape:
            raise ValueError(f"teacher segmentation shape mismatch: {frame_name}")

        full_region_scores = class_region_scores(
            full_features, classes, query_features, negatives, args.aggregation
        )
        full_scores, full_valid = region_scores_to_pixels(
            full_region_scores, segmentation
        )
        full_prediction = prediction_from_scores(
            full_scores, class_ids, full_valid
        )

        reconstructed = normalize(mean + ((full_features - mean) @ basis) @ basis.T)
        pca_region_scores = class_region_scores(
            reconstructed, classes, query_features, negatives, args.aggregation
        )
        pca_scores, pca_valid = region_scores_to_pixels(
            pca_region_scores, segmentation
        )
        if not np.array_equal(full_valid, pca_valid):
            raise RuntimeError("full and PCA validity differ")
        pca_prediction = prediction_from_scores(pca_scores, class_ids, pca_valid)

        oracle_prediction, oracle_valid, oracle_diagnostic = sam_oracle(
            segmentation, reference, reference_valid, class_ids
        )
        oracle_diagnostic.update({"frame": frame, "split": "selection"})
        oracle_diagnostics.append(oracle_diagnostic)

        stage_values = {
            "sam_oracle": (oracle_prediction, None, oracle_valid),
            "full_teacher": (full_prediction, full_scores, full_valid),
            "pca_teacher": (pca_prediction, pca_scores, pca_valid),
        }
        reference_pixels += int(np.count_nonzero(reference_valid))
        for name, (prediction, scores, valid) in stage_values.items():
            reference_parts[name].append(reference[reference_valid].astype(np.int64))
            prediction_parts[name].append(prediction[reference_valid].astype(np.int64))
            valid_pixels[name] += int(np.count_nonzero(reference_valid & valid))
            if scores is not None:
                score_parts[name].append(scores[reference_valid].astype(np.float32))

    reports = {}
    for name in stage_names:
        reports[name] = metric_report(
            np.concatenate(reference_parts[name]),
            np.concatenate(prediction_parts[name]),
            (
                np.concatenate(score_parts[name], axis=0)
                if name in score_parts else None
            ),
            classes,
            valid_pixels=valid_pixels[name],
            reference_pixels=reference_pixels,
            frames_with_reference=len(frames),
        )

    output = {
        "schema": "open_vocab_2d_teacher_subset_v1",
        "status": "training-side model selection; not final held-out evidence",
        "guardrail": (
            "Reference labels were read only by this evaluator and were not used "
            "to generate SAM regions, OpenCLIP features, or the PCA basis."
        ),
        "frames": frames,
        "minimum_reference_confidence": args.min_reference_confidence,
        "aggregation": args.aggregation,
        "teacher_report_sha256": sha256(teacher_report_path),
        "teacher_archives_combined_sha256": combined_sha256(
            archives, args.teacher_dir
        ),
        "pca_basis_sha256": sha256(args.basis_dir / "basis.f32"),
        "query_features_sha256": sha256(args.query_features),
        "prompt_groups_sha256": sha256(args.prompt_groups),
        "prompt_groups": prompt_document,
        "query_metadata": {
            "schema": query_document.get("schema"),
            "model": query_document.get("model"),
            "prompt_template": query_document.get("prompt_template"),
            "prompt_groups_schema": query_document.get("prompt_groups_schema"),
            "export_device": query_document.get("export_device"),
        },
        "reports": reports,
        "stage_drops": {
            "delta_teacher": (
                reports["sam_oracle"]["macro_iou"]
                - reports["full_teacher"]["macro_iou"]
            ),
            "delta_pca": (
                reports["full_teacher"]["macro_iou"]
                - reports["pca_teacher"]["macro_iou"]
            ),
        },
        "sam_oracle_aggregate": aggregate_sam_oracle_diagnostics(
            oracle_diagnostics, classes
        )["selection"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_json),
        "frames": len(frames),
        "macro_iou": {
            name: reports[name]["macro_iou"] for name in stage_names
        },
        "stage_drops": output["stage_drops"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
