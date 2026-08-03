#!/usr/bin/env python3
"""Export compact PCA language supervision for the C++ incremental mapper."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


MAGIC = b"LGSPCA\0\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument(
        "--basis-dir",
        type=Path,
        help="reuse an existing mean/basis instead of fitting on this probe set",
    )
    return parser.parse_args()


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype="<f4", order="C")
    with path.open("wb") as output:
        output.write(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
        output.write(matrix.tobytes(order="C"))


def read_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        rows, columns = struct.unpack("<II", source.read(8))
        matrix = np.frombuffer(source.read(), dtype="<f4")
    if matrix.size != rows * columns:
        raise ValueError(f"invalid matrix payload: {path}")
    return matrix.reshape(rows, columns).copy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target_dir = args.output_dir / "targets"
    segmentation_dir = args.output_dir / "segmentation"
    target_dir.mkdir()
    segmentation_dir.mkdir()
    report = json.loads((args.probe_dir / "report.json").read_text(encoding="utf-8"))
    archives = []
    features = []
    for frame in report["frame_records"]:
        archive = np.load(args.probe_dir / Path(frame["archive"]).name)
        archives.append({name: archive[name] for name in archive.files})
        features.append(functional.normalize(torch.from_numpy(
            archive["features"].astype(np.float32)), dim=1))
    all_features = torch.cat(features)
    if args.basis_dir is None:
        mean = all_features.mean(dim=0, keepdim=True)
        centered = all_features - mean
        covariance = centered.T @ centered / max(1, all_features.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[order]
        basis = eigenvectors[:, order[: args.dimension]].contiguous()
        explained_variance_fraction = float(
            eigenvalues[: args.dimension].clamp_min(0).sum() /
            eigenvalues.clamp_min(0).sum()
        )
    else:
        mean = torch.from_numpy(read_matrix(args.basis_dir / "mean.f32"))
        basis = torch.from_numpy(read_matrix(args.basis_dir / "basis.f32"))
        if mean.shape != (1, all_features.shape[1]):
            raise ValueError("reused PCA mean has the wrong teacher dimension")
        if basis.shape != (all_features.shape[1], args.dimension):
            raise ValueError("reused PCA basis has the wrong shape")
        source_constants = json.loads(
            (args.basis_dir / "constants.json").read_text(encoding="utf-8")
        )
        explained_variance_fraction = float(
            source_constants["explained_variance_fraction"]
        )
    basis_mean = (mean @ basis).squeeze(0)
    mean_norm_squared = float((mean * mean).sum())
    write_matrix(args.output_dir / "mean.f32", mean.numpy())
    write_matrix(args.output_dir / "basis.f32", basis.numpy())
    write_matrix(
        args.output_dir / "basis_mean.f32", basis_mean.unsqueeze(0).numpy()
    )
    constants = {
        "mean_norm_squared": mean_norm_squared,
        "explained_variance_fraction": explained_variance_fraction,
        "basis_reused": args.basis_dir is not None,
        "basis_source": str(args.basis_dir) if args.basis_dir is not None else None,
    }
    (args.output_dir / "constants.json").write_text(
        json.dumps(constants, indent=2), encoding="utf-8"
    )

    frames = []
    for frame, archive, teacher in zip(report["frame_records"], archives, features):
        frame_index = int(frame["frame_index"])
        basis_dot = (teacher @ basis).numpy().astype("<f2")
        mean_dot = (teacher @ mean.T).numpy().astype("<f2").reshape(-1)
        confidence = np.sqrt(np.clip(
            archive["predicted_iou"].astype(np.float32) *
            archive["stability"].astype(np.float32), 0.0, 1.0
        )).astype("<f2")
        target_path = target_dir / f"{frame_index:06d}.bin"
        with target_path.open("wb") as output:
            output.write(struct.pack(
                "<8sIII", MAGIC, 1, basis_dot.shape[0], basis_dot.shape[1]
            ))
            output.write(basis_dot.tobytes(order="C"))
            output.write(mean_dot.tobytes(order="C"))
            output.write(confidence.tobytes(order="C"))
        segmentation = archive["segmentation"].astype(np.int32) + 1
        segmentation[segmentation < 0] = 0
        segmentation_path = segmentation_dir / f"{frame_index:06d}.png"
        if not cv2.imwrite(str(segmentation_path), segmentation.astype(np.uint16)):
            raise RuntimeError(f"Failed to write {segmentation_path}")
        frames.append({
            "frame_index": frame_index,
            "regions": int(basis_dot.shape[0]),
            "target": str(target_path.relative_to(args.output_dir)),
            "segmentation": str(segmentation_path.relative_to(args.output_dir)),
        })
    manifest = {
        "schema": "pca_language_targets_v1",
        "offline_two_pass": True,
        "feature_dimension": args.dimension,
        "teacher_dimension": int(all_features.shape[1]),
        **constants,
        "frames": frames,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "frames": len(frames),
        "regions": int(all_features.shape[0]),
        "dimension": args.dimension,
        "explained_variance_fraction": constants["explained_variance_fraction"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
