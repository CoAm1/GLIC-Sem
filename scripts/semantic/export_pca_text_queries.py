#!/usr/bin/env python3
"""Encode open-vocabulary text queries with the exact teacher OpenCLIP model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--queries", nargs="+")
    source.add_argument(
        "--prompt-groups",
        type=Path,
        help="Load the exact query list, prompt template, and negatives from JSON.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CPU for text-only export when experiment GPUs are occupied.",
    )
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

    prompt_group_document = None
    if args.prompt_groups is not None:
        prompt_group_document = json.loads(
            args.prompt_groups.read_text(encoding="utf-8")
        )
        if prompt_group_document.get("schema") not in {
            "mcd_open_vocab_prompt_groups_v1",
            "mcd_open_vocab_prompt_groups_v2",
        }:
            raise ValueError("unexpected prompt-group schema")
        queries = []
        for class_id in prompt_group_document["primary_macro_class_ids"]:
            queries.extend(
                prompt_group_document["classes"][str(class_id)]["queries"]
            )
        if len(set(queries)) != len(queries):
            raise ValueError("prompt-group queries must be globally unique")
        prompt_template = str(prompt_group_document["prompt_template"])
        negative_prompts = [
            str(value) for value in prompt_group_document["negative_prompts"]
        ]
    else:
        queries = list(args.queries)
        prompt_template = args.prompt_template
        negative_prompts = list(args.negative_prompts)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    precision = "fp16" if device.type == "cuda" else "fp32"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.checkpoint), precision=precision
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompts = [prompt_template.format(query=query) for query in queries]
    all_prompts = prompts + negative_prompts
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
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
        "export_device": str(device),
        "prompt_template": prompt_template,
        "prompt_groups": str(args.prompt_groups) if args.prompt_groups else None,
        "prompt_groups_schema": (
            prompt_group_document.get("schema")
            if prompt_group_document is not None else None
        ),
        "prompt_groups_sha256": (
            hashlib.sha256(args.prompt_groups.read_bytes()).hexdigest()
            if args.prompt_groups is not None else None
        ),
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
            for label, prompt, text in zip(queries, prompts, text_features)
        ],
        "negatives": [
            {
                "prompt": prompt,
                "feature": feature.tolist(),
                "mean_dot": float(feature @ mean.reshape(-1)),
                "basis_dot": (feature @ basis).tolist(),
            }
            for prompt, feature in zip(negative_prompts, negative_features)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "queries": len(queries),
        "dimension": int(basis.shape[1]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
