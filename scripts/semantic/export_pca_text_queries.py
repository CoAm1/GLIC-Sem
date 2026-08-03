#!/usr/bin/env python3
"""Encode open-vocabulary text queries with the exact teacher OpenCLIP model."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as functional


def read_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        rows, columns = struct.unpack("<II", source.read(8))
        values = np.frombuffer(source.read(), dtype="<f4")
    if values.size != rows * columns:
        raise ValueError(f"invalid matrix: {path}")
    return values.reshape(rows, columns).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument(
        "--prompt-template",
        default="a photo of a {query}",
        help="Python format string. Use '{query}' to match LangSplatV2 evaluation.",
    )
    parser.add_argument(
        "--negative-prompts",
        nargs="+",
        default=["object", "things", "stuff", "texture"],
        help="Negative prompts used by LangSplat-style relevance queries.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.checkpoint), precision="fp16"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = [args.prompt_template.format(query=query) for query in args.queries]
    all_prompts = prompts + args.negative_prompts
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16
    ):
        all_text_features = model.encode_text(tokenizer(all_prompts).to(device))
    all_text_features = functional.normalize(
        all_text_features.float(), dim=1
    ).cpu().numpy()
    text_features = all_text_features[: len(prompts)]
    negative_features = all_text_features[len(prompts) :]

    mean = read_matrix(args.basis_dir / "mean.f32")
    basis = read_matrix(args.basis_dir / "basis.f32")
    basis_mean = read_matrix(args.basis_dir / "basis_mean.f32").reshape(-1)
    if mean.shape != (1, text_features.shape[1]):
        raise ValueError("teacher mean and text embedding dimensions differ")
    if basis.shape[0] != text_features.shape[1]:
        raise ValueError("PCA basis and text embedding dimensions differ")

    report = {
        "schema": "pca_text_queries_v2",
        "model": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "checkpoint": str(args.checkpoint),
        "prompt_template": args.prompt_template,
        "mean_norm_squared": float(np.square(mean).sum()),
        "basis_mean": basis_mean.tolist(),
        "queries": [
            {
                "label": label,
                "prompt": prompt,
                "feature": text.tolist(),
                "mean_dot": float(text @ mean.reshape(-1)),
                "basis_dot": (text @ basis).tolist(),
            }
            for label, prompt, text in zip(args.queries, prompts, text_features)
        ],
        "negatives": [
            {
                "prompt": prompt,
                "feature": feature.tolist(),
                "mean_dot": float(feature @ mean.reshape(-1)),
                "basis_dot": (feature @ basis).tolist(),
            }
            for prompt, feature in zip(args.negative_prompts, negative_features)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "queries": len(args.queries),
        "dimension": int(basis.shape[1]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
