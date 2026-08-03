#!/usr/bin/env python3
"""Measure whether PCA compression preserves full-CLIP text query decisions."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    args = parser.parse_args()

    teacher_report = json.loads(
        (args.probe_dir / "report.json").read_text(encoding="utf-8")
    )
    features = []
    for frame in teacher_report["frame_records"]:
        archive = np.load(args.probe_dir / Path(frame["archive"]).name)
        feature = archive["features"].astype(np.float32)
        feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
        features.append(feature)
    teacher = np.concatenate(features)

    mean = read_matrix(args.basis_dir / "mean.f32")
    basis = read_matrix(args.basis_dir / "basis.f32")
    latent = (teacher - mean) @ basis
    reconstructed = mean + latent @ basis.T
    reconstructed /= np.maximum(
        np.linalg.norm(reconstructed, axis=1, keepdims=True), 1e-12
    )
    query_report = json.loads(args.queries.read_text(encoding="utf-8"))
    query = np.asarray(
        [record["feature"] for record in query_report["queries"]],
        dtype=np.float32,
    )
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)

    full_scores = teacher @ query.T
    pca_scores = reconstructed @ query.T
    full_order = np.argsort(full_scores, axis=1)
    pca_order = np.argsort(pca_scores, axis=1)
    full_top = full_order[:, -1]
    pca_top = pca_order[:, -1]
    full_margin = np.take_along_axis(
        full_scores, full_order[:, -1:], axis=1
    )[:, 0] - np.take_along_axis(
        full_scores, full_order[:, -2:-1], axis=1
    )[:, 0]
    agreement = full_top == pca_top
    labels = [record["label"] for record in query_report["queries"]]
    print(json.dumps({
        "regions": int(len(teacher)),
        "queries": labels,
        "top1_agreement": float(agreement.mean()),
        "top1_agreement_full_margin_ge_0.01": float(
            agreement[full_margin >= 0.01].mean()
        ) if np.any(full_margin >= 0.01) else None,
        "top1_agreement_full_margin_ge_0.02": float(
            agreement[full_margin >= 0.02].mean()
        ) if np.any(full_margin >= 0.02) else None,
        "full_margin_mean": float(full_margin.mean()),
        "score_mae": float(np.abs(full_scores - pca_scores).mean()),
        "score_max_abs": float(np.abs(full_scores - pca_scores).max()),
        "full_top1_counts": {
            label: int((full_top == index).sum())
            for index, label in enumerate(labels)
        },
        "pca_top1_counts": {
            label: int((pca_top == index).sum())
            for index, label in enumerate(labels)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
