#!/usr/bin/env python3
"""Render training-split SAM partition diagnostics for sparse references.

The resulting images are evaluator-only evidence.  Reference labels are used
to color a SAM oracle and must never be consumed by the teacher or mapper.
Sparse reference pixels are dilated only for display; metrics use raw pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluate_semantic_error_decomposition import (
    load_png,
    load_prompt_groups,
    sam_oracle,
)


PALETTE_BGR = np.asarray([
    [0, 0, 0],
    [70, 180, 70],
    [170, 120, 70],
    [50, 150, 30],
    [180, 140, 60],
    [0, 210, 255],
    [40, 40, 230],
    [220, 180, 40],
    [220, 80, 180],
    [180, 80, 80],
    [130, 130, 130],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prompt-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--min-reference-confidence", type=float, default=0.35)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--display-radius", type=int, default=2)
    parser.add_argument("--focus-class-ids", type=int, nargs="*", default=[7, 8, 9])
    parser.add_argument("--focus-min-side", type=int, default=180)
    return parser.parse_args()


def indexed_images(root: Path) -> dict[int, Path]:
    result = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in (root / "images").glob(pattern):
            try:
                result[int(path.stem)] = path
            except ValueError:
                continue
    return result


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def display_expand(values: np.ndarray, valid: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0:
        return values, valid
    expanded = np.zeros_like(values)
    expanded_valid = np.zeros_like(valid)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    for value in np.unique(values[valid]):
        mask = ((values == value) & valid).astype(np.uint8)
        dilated = cv2.dilate(mask, kernel) > 0
        expanded[dilated] = value
        expanded_valid |= dilated
    return expanded, expanded_valid


def color_overlay(
    image: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    palette: np.ndarray,
    opacity: float,
) -> np.ndarray:
    output = image.copy()
    colors = palette[np.clip(values, 0, len(palette) - 1)]
    output[valid] = np.rint(
        (1.0 - opacity) * image[valid] + opacity * colors[valid]
    ).astype(np.uint8)
    return output


def region_boundaries(segmentation: np.ndarray) -> np.ndarray:
    valid = segmentation >= 0
    boundary = np.zeros(segmentation.shape, dtype=np.bool_)
    horizontal = valid[:, 1:] & valid[:, :-1] & (
        segmentation[:, 1:] != segmentation[:, :-1]
    )
    vertical = valid[1:, :] & valid[:-1, :] & (
        segmentation[1:, :] != segmentation[:-1, :]
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def square_focus_bounds(
    mask: np.ndarray, minimum_side: int
) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    height, width = mask.shape
    center_x = 0.5 * (float(columns.min()) + float(columns.max()))
    center_y = 0.5 * (float(rows.min()) + float(rows.max()))
    span = max(
        minimum_side,
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )
    span = min(span, height, width)
    x0 = int(round(center_x - span / 2))
    y0 = int(round(center_y - span / 2))
    x0 = min(max(x0, 0), width - span)
    y0 = min(max(y0, 0), height - span)
    return x0, y0, x0 + span, y0 + span


def resize_focus(panel: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return cv2.resize(panel[y0:y1, x0:x1], (360, 360), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.display_radius < 0:
        raise ValueError("display-radius must be non-negative")
    if args.focus_min_side < 1:
        raise ValueError("focus-min-side must be positive")
    args.output_dir.mkdir(parents=True)

    shape = (args.image_height, args.image_width)
    classes, _ = load_prompt_groups(args.prompt_groups)
    class_ids = np.asarray([item.class_id for item in classes], dtype=np.int64)
    images = indexed_images(args.dataset_root)
    montages = []
    for frame in args.frame_indices:
        if frame not in images:
            raise FileNotFoundError(f"missing image for frame {frame}")
        image = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != shape:
            raise ValueError(f"invalid image for frame {frame}: {images[frame]}")
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
        prediction, sam_valid, diagnostic = sam_oracle(
            segmentation, reference, reference_valid, class_ids
        )

        expanded_reference, expanded_valid = display_expand(
            reference, reference_valid, args.display_radius
        )
        reference_panel = color_overlay(
            image, expanded_reference, expanded_valid, PALETTE_BGR, 0.85
        )

        boundary_panel = image.copy()
        boundaries = region_boundaries(segmentation)
        boundary_panel[boundaries] = (0, 255, 255)

        oracle_panel = color_overlay(
            image, prediction, sam_valid, PALETTE_BGR, 0.55
        )

        error_code = np.zeros(shape, dtype=np.uint8)
        error_code[reference_valid & ~sam_valid] = 1
        error_code[
            reference_valid & sam_valid & (prediction != reference)
        ] = 2
        error_code[
            reference_valid & sam_valid & (prediction == reference)
        ] = 3
        error_palette = np.asarray(
            [[0, 0, 0], [255, 80, 0], [0, 0, 255], [0, 210, 0]],
            dtype=np.uint8,
        )
        expanded_error, expanded_error_valid = display_expand(
            error_code, error_code > 0, args.display_radius
        )
        error_panel = color_overlay(
            image, expanded_error, expanded_error_valid, error_palette, 0.90
        )

        reference_count = int(np.count_nonzero(reference_valid))
        covered_count = int(np.count_nonzero(reference_valid & sam_valid))
        correct_count = int(np.count_nonzero(
            reference_valid & sam_valid & (prediction == reference)
        ))
        panels = [
            add_title(image, f"RGB frame {frame}"),
            add_title(
                reference_panel,
                f"Sparse reference (display radius {args.display_radius}px)",
            ),
            add_title(boundary_panel, "SAM exclusive-region boundaries"),
            add_title(oracle_panel, "SAM Oracle (EVALUATION ONLY)"),
            add_title(
                error_panel,
                "Error: green correct | red covered-wrong | blue uncovered",
            ),
        ]
        montage = cv2.hconcat(panels)
        footer = np.zeros((46, montage.shape[1], 3), dtype=np.uint8)
        coverage = covered_count / reference_count if reference_count else 0.0
        covered_accuracy = correct_count / covered_count if covered_count else 0.0
        cv2.putText(
            footer,
            (
                f"raw reference={reference_count} covered={covered_count} "
                f"coverage={coverage:.3f} correct={correct_count} "
                f"oracle_acc_on_covered={covered_accuracy:.3f} "
                f"regions={diagnostic['assigned_regions']}"
            ),
            (10, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        montage = cv2.vconcat([montage, footer])
        cv2.imwrite(str(args.output_dir / f"{frame:06d}_sam_oracle.png"), montage)
        montages.append(montage)

        class_name = {item.class_id: item.name for item in classes}
        for class_id in args.focus_class_ids:
            class_reference = reference_valid & (reference == class_id)
            bounds = square_focus_bounds(class_reference, args.focus_min_side)
            if bounds is None:
                continue
            focused_panels = [
                add_title(resize_focus(image, bounds), f"RGB | class {class_id}"),
                add_title(
                    resize_focus(reference_panel, bounds),
                    f"Reference | {class_name.get(class_id, 'unknown')}",
                ),
                add_title(
                    resize_focus(boundary_panel, bounds),
                    "SAM boundaries",
                ),
                add_title(
                    resize_focus(oracle_panel, bounds),
                    "Oracle regions (EVAL ONLY)",
                ),
                add_title(
                    resize_focus(error_panel, bounds),
                    "green correct | red wrong | blue missing",
                ),
            ]
            focused = cv2.hconcat(focused_panels)
            raw_count = int(np.count_nonzero(class_reference))
            class_covered = int(np.count_nonzero(class_reference & sam_valid))
            class_correct = int(np.count_nonzero(
                class_reference & sam_valid & (prediction == class_id)
            ))
            focus_footer = np.zeros((46, focused.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                focus_footer,
                (
                    f"frame={frame} class={class_id}:{class_name.get(class_id, 'unknown')} "
                    f"raw_ref={raw_count} covered={class_covered} "
                    f"oracle_correct={class_correct} crop={bounds}"
                ),
                (10, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            focused = cv2.vconcat([focused, focus_footer])
            cv2.imwrite(
                str(args.output_dir / f"{frame:06d}_class{class_id:02d}_focus.png"),
                focused,
            )

    if montages:
        cv2.imwrite(str(args.output_dir / "selected_frames_montage.png"), cv2.vconcat(montages))


if __name__ == "__main__":
    main()
