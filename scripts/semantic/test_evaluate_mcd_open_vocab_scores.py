#!/usr/bin/env python3
"""Deterministic unit tests for the sparse MCD open-vocabulary evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import evaluate_mcd_open_vocab_scores as evaluator


SHAPE = (4, 4)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def quantize(score: float) -> np.uint16:
    return np.uint16(round(score * 65535.0))


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.score_root = self.root / "scores"
        self.alpha_root = self.root / "alpha"
        self.reference_root = self.root / "reference"
        self.prompt_path = self.root / "prompts.json"
        self.alignment_path = self.root / "alignment.csv"
        self.output_path = self.root / "evaluation.json"
        self.prompts = {
            "schema": "mcd_open_vocab_prompt_groups_v1",
            "frozen_before_heldout_evaluation": True,
            "prompt_template": "a photo of a {query}",
            "negative_prompts": ["object"],
            "classes": {
                "1": {"name": "one", "queries": ["one", "first class"]},
                "2": {"name": "two", "queries": ["two"]},
            },
            "primary_macro_class_ids": [1, 2],
            "excluded_reference_class_ids": [0, 11],
        }
        self.prompt_path.write_text(json.dumps(self.prompts), encoding="utf-8")
        self.classes, _ = evaluator.load_prompt_groups(self.prompt_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_frame(
        self,
        frame: int,
        reference: np.ndarray,
        alpha: np.ndarray,
        one: np.ndarray,
        first_class: np.ndarray,
        two: np.ndarray,
    ) -> None:
        name = f"{frame:06d}.png"
        write_png(self.reference_root / "labels" / name, reference.astype(np.uint8))
        write_png(
            self.reference_root / "confidence" / name,
            np.full(SHAPE, 255, dtype=np.uint8),
        )
        write_png(self.alpha_root / name, alpha.astype(np.uint8))
        for query, scores in (
            ("one", one),
            ("first class", first_class),
            ("two", two),
        ):
            write_png(
                self.score_root / evaluator.cpp_safe_file_label(query) / name,
                scores.astype(np.uint16),
            )

    def test_safe_labels_and_collision_detection(self) -> None:
        self.assertEqual(evaluator.cpp_safe_file_label("traffic sign"), "traffic_sign")
        self.assertEqual(evaluator.cpp_safe_file_label("a/b"), "a_b")
        self.assertEqual(evaluator.cpp_safe_file_label("中文"), "______")
        collision = dict(self.prompts)
        collision["classes"] = {
            "1": {"name": "one", "queries": ["a b"]},
            "2": {"name": "two", "queries": ["a/b"]},
        }
        path = self.root / "collision.json"
        path.write_text(json.dumps(collision), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "safe-label collision"):
            evaluator.load_prompt_groups(path)

    def test_uint16_dequantization_and_max_mean_aggregation(self) -> None:
        name = "000004.png"
        zeros = np.zeros(SHAPE, dtype=np.uint16)
        one = zeros.copy()
        first = zeros.copy()
        two = zeros.copy()
        one[0, 0] = quantize(0.25)
        first[0, 0] = quantize(0.75)
        two[0, 0] = quantize(0.50)
        for query, scores in (
            ("one", one),
            ("first class", first),
            ("two", two),
        ):
            write_png(
                self.score_root / evaluator.cpp_safe_file_label(query) / name,
                scores,
            )
        maximum = evaluator.aggregate_scores(
            self.score_root, name, self.classes, "max", SHAPE
        )
        mean = evaluator.aggregate_scores(
            self.score_root, name, self.classes, "mean", SHAPE
        )
        self.assertAlmostEqual(maximum[0, 0, 0], float(first[0, 0]) / 65535.0)
        self.assertAlmostEqual(maximum[0, 0, 1], float(two[0, 0]) / 65535.0)
        self.assertAlmostEqual(
            mean[0, 0, 0],
            (float(one[0, 0]) + float(first[0, 0])) / (2.0 * 65535.0),
        )

    def test_end_to_end_split_ignore_alpha_confusion_ap_and_iou(self) -> None:
        # Frame 4 is a training keyframe. Frame 5 shares its scan and must be
        # excluded. Frame 6 is the only disjoint-scan held-out frame.
        self.alignment_path.write_text(
            "frame_index,label_scan_index\n4,100\n5,100\n6,101\n",
            encoding="utf-8",
        )
        base_reference = np.zeros(SHAPE, dtype=np.uint8)
        alpha = np.full(SHAPE, 255, dtype=np.uint8)
        low = np.full(SHAPE, quantize(0.1), dtype=np.uint16)

        train_ref = base_reference.copy()
        train_ref[0, :4] = [1, 2, 0, 11]
        train_one = low.copy()
        train_first = low.copy()
        train_two = low.copy()
        train_one[0, 0] = quantize(0.9)
        train_two[0, 1] = quantize(0.9)
        self.write_frame(4, train_ref, alpha, train_one, train_first, train_two)

        shared_ref = base_reference.copy()
        shared_ref[0, :2] = [1, 2]
        shared_one = low.copy()
        shared_first = low.copy()
        shared_two = low.copy()
        shared_one[0, 0] = quantize(0.9)
        shared_two[0, 1] = quantize(0.9)
        self.write_frame(
            5, shared_ref, alpha, shared_one, shared_first, shared_two
        )

        held_ref = base_reference.copy()
        held_ref[0, :4] = [1, 2, 2, 0]
        held_ref[1, 0] = 1
        held_alpha = alpha.copy()
        held_alpha[1, 0] = 0
        held_one = low.copy()
        held_first = low.copy()
        held_two = low.copy()
        held_one[0, :3] = [
            quantize(0.9),
            quantize(0.8),
            quantize(0.1),
        ]
        held_two[0, :3] = [
            quantize(0.1),
            quantize(0.2),
            quantize(0.9),
        ]
        self.write_frame(
            6, held_ref, held_alpha, held_one, held_first, held_two
        )

        argv = [
            "evaluate_mcd_open_vocab_scores.py",
            "--score-root",
            str(self.score_root),
            "--alpha-root",
            str(self.alpha_root),
            "--prompt-groups",
            str(self.prompt_path),
            "--reference-dir",
            str(self.reference_root),
            "--alignment-csv",
            str(self.alignment_path),
            "--output",
            str(self.output_path),
            "--start-frame",
            "4",
            "--end-frame",
            "6",
            "--image-height",
            "4",
            "--image-width",
            "4",
            "--calibrate-unknown-on-train",
        ]
        with mock.patch.object(sys, "argv", argv):
            evaluator.main()
        result = json.loads(self.output_path.read_text(encoding="utf-8"))
        heldout = result["heldout_disjoint_scan"]
        self.assertEqual(result["split_rule"]["shared_scan_excluded_frames"], [5])
        self.assertEqual(heldout["reference_known_pixels_before_alpha"], 4)
        self.assertEqual(heldout["reference_pixels"], 3)
        self.assertEqual(heldout["frames_with_pixels"], 1)
        self.assertAlmostEqual(heldout["alpha_valid_coverage"], 0.75)
        self.assertAlmostEqual(heldout["pixel_accuracy"], 2.0 / 3.0)
        self.assertEqual(heldout["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertAlmostEqual(heldout["per_class"]["1"]["iou"], 0.5)
        self.assertAlmostEqual(heldout["per_class"]["2"]["iou"], 0.5)
        self.assertAlmostEqual(heldout["macro_iou"], 0.5)
        self.assertAlmostEqual(heldout["macro_ap"], 1.0)
        self.assertEqual(heldout["reference_pixels"], sum(map(sum, heldout["confusion_matrix"])))

    def test_missing_query_wrong_dtype_and_wrong_shape_fail(self) -> None:
        name = "000004.png"
        valid = np.zeros(SHAPE, dtype=np.uint16)
        write_png(self.score_root / "one" / name, valid)
        with self.assertRaises(FileNotFoundError):
            evaluator.aggregate_scores(
                self.score_root, name, self.classes, "max", SHAPE
            )

        wrong_dtype = self.root / "wrong_dtype.png"
        write_png(wrong_dtype, np.zeros(SHAPE, dtype=np.uint8))
        with self.assertRaises(TypeError):
            evaluator.load_png(wrong_dtype, np.dtype(np.uint16), SHAPE)

        wrong_shape = self.root / "wrong_shape.png"
        write_png(wrong_shape, np.zeros((8, 8), dtype=np.uint16))
        with self.assertRaises(ValueError):
            evaluator.load_png(wrong_shape, np.dtype(np.uint16), SHAPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
