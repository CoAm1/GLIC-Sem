#!/usr/bin/env python3
"""Fit a reusable CLIP PCA basis from scene regions and generic text anchors.

The output uses the matrix format consumed by the incremental C++ mapper.  A
basis produced here can be frozen and reused on a different scene, unlike the
original per-scene PCA export.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as functional
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES


PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a photo of the {}",
    "a close-up photo of a {}",
    "an indoor photo of a {}",
    "an outdoor photo of a {}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument(
        "--text-mass-ratio",
        type=float,
        default=0.5,
        help="Total covariance weight of text anchors relative to scene regions.",
    )
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype="<f4", order="C")
    with path.open("wb") as output:
        output.write(struct.pack("<II", matrix.shape[0], matrix.shape[1]))
        output.write(matrix.tobytes(order="C"))


def load_probe_features(paths: list[Path]) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for path in paths:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        for frame in report["frame_records"]:
            archive = np.load(path / Path(frame["archive"]).name)
            features.append(torch.from_numpy(
                archive["features"].astype(np.float32)
            ))
    return functional.normalize(torch.cat(features), dim=1)


@torch.inference_mode()
def encode_text_anchors(
    checkpoint: Path, batch_size: int, device: torch.device
) -> torch.Tensor:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(checkpoint), precision="fp16"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = [
        template.format(class_name)
        for class_name in IMAGENET_CLASSNAMES
        for template in PROMPT_TEMPLATES
    ]
    outputs: list[torch.Tensor] = []
    for begin in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[begin : begin + batch_size]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            encoded = model.encode_text(tokens)
        outputs.append(functional.normalize(encoded.float(), dim=1).cpu())
    return torch.cat(outputs)


def main() -> None:
    args = parse_args()
    if args.dimension < 1 or args.dimension > 512:
        raise ValueError("dimension must be in [1, 512]")
    if args.text_mass_ratio < 0:
        raise ValueError("text-mass-ratio must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)

    scene = load_probe_features(args.probe_dir).double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = encode_text_anchors(
        args.checkpoint, args.text_batch_size, device
    ).double()

    scene_weight = torch.ones(scene.shape[0], dtype=torch.float64)
    text_weight_per_item = (
        args.text_mass_ratio * scene_weight.sum() / max(1, text.shape[0])
    )
    text_weight = torch.full(
        (text.shape[0],), float(text_weight_per_item), dtype=torch.float64
    )
    values = torch.cat((scene, text), dim=0)
    weights = torch.cat((scene_weight, text_weight), dim=0)
    weight_sum = weights.sum()
    mean = (values * weights.unsqueeze(1)).sum(0, keepdim=True) / weight_sum
    centered = values - mean
    covariance = (
        centered.T @ (centered * weights.unsqueeze(1))
    ) / max(float(weight_sum - 1.0), 1.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order[: args.dimension]].contiguous()

    mean_f32 = mean.float()
    basis_f32 = basis.float()
    basis_mean = (mean_f32 @ basis_f32).squeeze(0)
    explained = float(
        eigenvalues[: args.dimension].clamp_min(0).sum()
        / eigenvalues.clamp_min(0).sum()
    )
    write_matrix(args.output_dir / "mean.f32", mean_f32.numpy())
    write_matrix(args.output_dir / "basis.f32", basis_f32.numpy())
    write_matrix(
        args.output_dir / "basis_mean.f32",
        basis_mean.unsqueeze(0).numpy(),
    )
    report = {
        "schema": "universal_language_pca_v1",
        "model": "ViT-B-16",
        "checkpoint": str(args.checkpoint),
        "dimension": args.dimension,
        "scene_probe_dirs": [str(path) for path in args.probe_dir],
        "scene_regions": int(scene.shape[0]),
        "text_anchors": int(text.shape[0]),
        "text_mass_ratio": args.text_mass_ratio,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "mean_norm_squared": float(mean_f32.square().sum()),
        "explained_variance_fraction": explained,
    }
    (args.output_dir / "constants.json").write_text(
        json.dumps({
            "mean_norm_squared": report["mean_norm_squared"],
            "explained_variance_fraction": explained,
            "basis_reused": False,
            "basis_source": "generic_text_anchors_plus_scene_probes",
        }, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
