#!/usr/bin/env python3
"""Export a simple colored point cloud from PCA-language Gaussian queries."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from inspect_language_pca_ply import read_vertex_table


PALETTE = np.asarray(
    [
        [128, 64, 128],   # floor
        [180, 180, 180],  # wall
        [255, 170, 0],    # table
        [60, 180, 75],    # chair
        [150, 75, 0],     # box
        [255, 225, 25],   # ball
        [0, 130, 200],    # monitor
        [145, 30, 180],   # cabinet
    ],
    dtype=np.uint8,
)


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    class_ids: np.ndarray,
    maximum: np.ndarray,
    margin: np.ndarray,
) -> None:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("class_id", "u1"), ("max_similarity", "<f4"),
        ("similarity_margin", "<f4"),
    ])
    output = np.empty(len(xyz), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        output[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        output[name] = colors[:, index]
    output["class_id"] = class_ids
    output["max_similarity"] = maximum
    output["similarity_margin"] = margin
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(output)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar class_id\nproperty float max_similarity\n"
        "property float similarity_margin\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as target:
        target.write(header)
        output.tofile(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vertices = read_vertex_table(args.ply)
    report = json.loads(args.queries.read_text(encoding="utf-8"))
    query_records = report["queries"]
    if len(query_records) > len(PALETTE):
        raise ValueError(f"palette supports at most {len(PALETTE)} queries")
    pca_names = sorted(
        (name for name in vertices.dtype.names if name.startswith("language_pca_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    latent = np.column_stack([vertices[name] for name in pca_names]).astype(np.float32)
    basis_mean = np.asarray(report["basis_mean"], dtype=np.float32)
    query_basis = np.asarray(
        [record["basis_dot"] for record in query_records], dtype=np.float32
    )
    query_mean = np.asarray(
        [record["mean_dot"] for record in query_records], dtype=np.float32
    )
    denominator = np.sqrt(np.maximum(
        report["mean_norm_squared"] + 2.0 * (latent @ basis_mean) +
        np.square(latent).sum(axis=1),
        1e-12,
    ))
    similarity = (
        latent @ query_basis.T + query_mean[None, :]
    ) / denominator[:, None]
    order = np.argsort(similarity, axis=1)
    class_ids = order[:, -1].astype(np.uint8)
    maximum = np.take_along_axis(similarity, order[:, -1:], axis=1)[:, 0]
    runner_up = np.take_along_axis(similarity, order[:, -2:-1], axis=1)[:, 0]
    margin = maximum - runner_up
    colors = PALETTE[class_ids]
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    write_binary_ply(args.output, xyz, colors, class_ids, maximum, margin)
    counts = np.bincount(class_ids, minlength=len(query_records))
    print(json.dumps({
        "points": int(len(vertices)),
        "output": str(args.output),
        "similarity_mean": float(maximum.mean()),
        "margin_mean": float(margin.mean()),
        "classes": {
            record["label"]: int(count)
            for record, count in zip(query_records, counts)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
