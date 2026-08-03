#!/usr/bin/env python3
"""Audit PCA preservation of the official LangSplat positive/negative relevancy."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        rows, columns = struct.unpack("<II", source.read(8))
        values = np.frombuffer(source.read(), dtype="<f4")
    if values.size != rows * columns:
        raise ValueError(f"invalid matrix: {path}")
    return values.reshape(rows, columns).copy()


def load_teacher(probe_dir: Path) -> tuple[np.ndarray, list[slice]]:
    report = json.loads((probe_dir / "report.json").read_text(encoding="utf-8"))
    features = []
    frame_slices = []
    offset = 0
    for frame in report["frame_records"]:
        archive = np.load(probe_dir / Path(frame["archive"]).name)
        feature = archive["features"].astype(np.float32)
        feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
        features.append(feature)
        frame_slices.append(slice(offset, offset + len(feature)))
        offset += len(feature)
    return np.concatenate(features), frame_slices


def relevancy(features: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Match OpenCLIPNetwork.get_relevancy: min positive probability over negatives."""
    positive_score = features @ positive.T
    negative_score = features @ negative.T
    delta = positive_score[:, :, None] - negative_score[:, None, :]
    probability = 1.0 / (1.0 + np.exp(np.clip(-10.0 * delta, -80.0, 80.0)))
    return probability.min(axis=2)


def correlation_per_query(reference: np.ndarray, estimate: np.ndarray) -> list[float | None]:
    result = []
    for index in range(reference.shape[1]):
        ref = reference[:, index]
        est = estimate[:, index]
        denominator = float(ref.std() * est.std())
        result.append(
            float(np.mean((ref - ref.mean()) * (est - est.mean())) / denominator)
            if denominator > 1e-12
            else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    teacher, frame_slices = load_teacher(args.probe_dir)
    mean = read_matrix(args.basis_dir / "mean.f32")
    basis = read_matrix(args.basis_dir / "basis.f32")
    reconstructed = mean + ((teacher - mean) @ basis) @ basis.T
    reconstructed /= np.maximum(
        np.linalg.norm(reconstructed, axis=1, keepdims=True), 1e-12
    )

    query_report = json.loads(args.queries.read_text(encoding="utf-8"))
    if query_report.get("schema") != "pca_text_queries_v2":
        raise ValueError("query JSON must include v2 negative prompts")
    positive = np.asarray(
        [record["feature"] for record in query_report["queries"]], dtype=np.float32
    )
    negative = np.asarray(
        [record["feature"] for record in query_report["negatives"]], dtype=np.float32
    )
    positive /= np.maximum(np.linalg.norm(positive, axis=1, keepdims=True), 1e-12)
    negative /= np.maximum(np.linalg.norm(negative, axis=1, keepdims=True), 1e-12)

    full = relevancy(teacher, positive, negative)
    compact = relevancy(reconstructed, positive, negative)
    absolute_error = np.abs(full - compact)
    labels = [record["label"] for record in query_report["queries"]]

    # Compare the top 10% response set within each frame. This is teacher
    # consistency, not semantic accuracy and deliberately uses no GT labels.
    overlaps = [[] for _ in labels]
    for frame_slice in frame_slices:
        for query_index in range(len(labels)):
            ref = full[frame_slice, query_index]
            est = compact[frame_slice, query_index]
            count = max(1, int(np.ceil(0.1 * len(ref))))
            ref_top = set(np.argpartition(ref, -count)[-count:].tolist())
            est_top = set(np.argpartition(est, -count)[-count:].tolist())
            overlaps[query_index].append(len(ref_top & est_top) / len(ref_top | est_top))

    correlations = correlation_per_query(full, compact)
    report = {
        "schema": "pca_langsplat_relevancy_audit_v1",
        "regions": int(len(teacher)),
        "frames": len(frame_slices),
        "dimension": int(basis.shape[1]),
        "prompt_template": query_report["prompt_template"],
        "negative_prompts": [record["prompt"] for record in query_report["negatives"]],
        "temperature": 10.0,
        "global_mae": float(absolute_error.mean()),
        "global_p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "global_max_absolute_error": float(absolute_error.max()),
        "per_query": {
            label: {
                "mae": float(absolute_error[:, index].mean()),
                "pearson": correlations[index],
                "frame_mean_top10pct_iou": float(np.mean(overlaps[index])),
                "teacher_relevancy_mean": float(full[:, index].mean()),
                "teacher_relevancy_p95": float(np.quantile(full[:, index], 0.95)),
            }
            for index, label in enumerate(labels)
        },
        "interpretation_guardrail": (
            "Measures PCA consistency with the full teacher only; it is not semantic "
            "accuracy, mIoU, or evidence that any query is correct."
        ),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
