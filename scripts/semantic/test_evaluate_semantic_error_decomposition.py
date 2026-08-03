#!/usr/bin/env python3
"""Deterministic tests for the semantic error-decomposition evaluator."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import evaluate_semantic_error_decomposition as evaluator


class SemanticErrorDecompositionTests(unittest.TestCase):
    def test_prompt_groups_v2_is_accepted(self) -> None:
        path = (
            Path(__file__).parents[2]
            / "config"
            / "mcd_open_vocab_prompt_groups_v2.json"
        )
        classes, loaded = evaluator.load_prompt_groups(path)
        self.assertEqual(loaded["schema"], "mcd_open_vocab_prompt_groups_v2")
        self.assertIn("traffic cone", classes[8].queries)

    def test_disjoint_scan_split(self) -> None:
        frames = [4, 5, 6, 7, 8, 9]
        alignment = {4: 100, 5: 100, 6: 101, 7: -1, 8: 102, 9: 103}
        split, report = evaluator.make_split(frames, alignment, 5, 4)
        self.assertEqual(split[4], "train")
        self.assertEqual(split[5], "shared_scan_excluded")
        self.assertEqual(split[6], "heldout")
        self.assertEqual(split[7], "unmatched")
        self.assertEqual(split[8], "heldout")
        self.assertEqual(split[9], "train")
        self.assertEqual(report["training_frames"], [4, 9])
        self.assertEqual(report["shared_scan_excluded_frames"], [5])

    def test_sam_oracle_majority_and_uncovered_pixels(self) -> None:
        segmentation = np.asarray([[0, 0, 1], [-1, 1, 1]], dtype=np.int32)
        reference = np.asarray([[1, 1, 2], [2, 1, 2]], dtype=np.uint8)
        valid = np.ones(reference.shape, dtype=np.bool_)
        prediction, sam_valid, diagnostic = evaluator.sam_oracle(
            segmentation, reference, valid, np.asarray([1, 2], dtype=np.int64)
        )
        self.assertEqual(prediction[0, 0], 1)
        # Region 1 has two class-2 pixels and one class-1 pixel.
        self.assertEqual(prediction[0, 2], 2)
        self.assertEqual(prediction[1, 1], 2)
        self.assertEqual(prediction[1, 0], 0)
        self.assertFalse(sam_valid[1, 0])
        self.assertEqual(diagnostic["assigned_regions"], 2)
        self.assertAlmostEqual(diagnostic["sam_pixel_coverage_on_reference"], 5 / 6)
        self.assertEqual(
            diagnostic["per_class"]["1"]["reference_pixels"], 3
        )
        self.assertEqual(
            diagnostic["per_class"]["1"]["sam_covered_reference_pixels"], 3
        )
        self.assertEqual(
            diagnostic["per_class"]["1"]["oracle_correct_reference_pixels"], 2
        )
        self.assertAlmostEqual(
            diagnostic["per_class"]["1"]["sam_reference_coverage"], 1.0
        )
        self.assertAlmostEqual(
            diagnostic["per_class"]["1"]["oracle_accuracy_on_sam_covered_reference"],
            2 / 3,
        )
        self.assertEqual(
            diagnostic["per_class"]["2"]["reference_pixels"], 3
        )
        self.assertEqual(
            diagnostic["per_class"]["2"]["sam_covered_reference_pixels"], 2
        )
        self.assertAlmostEqual(
            diagnostic["per_class"]["2"]["oracle_recall_on_all_reference"], 2 / 3
        )
        diagnostic["split"] = "train"
        aggregate = evaluator.aggregate_sam_oracle_diagnostics(
            [diagnostic],
            [
                evaluator.PromptClass(1, "one", ("one",)),
                evaluator.PromptClass(2, "two", ("two",)),
            ],
        )
        class_one = aggregate["train"]["per_class"]["1"]
        self.assertEqual(class_one["uncovered_reference_pixels"], 0)
        self.assertEqual(
            class_one["covered_but_not_oracle_correct_pixels"], 1
        )
        class_two = aggregate["train"]["per_class"]["2"]
        self.assertEqual(class_two["uncovered_reference_pixels"], 1)
        self.assertEqual(
            class_two["covered_but_not_oracle_correct_pixels"], 0
        )
        self.assertAlmostEqual(
            class_two["uncovered_fraction_of_reference"], 1 / 3
        )

    def test_primary_domain_penalizes_missing_coverage(self) -> None:
        classes = [
            evaluator.PromptClass(1, "one", ("one",)),
            evaluator.PromptClass(2, "two", ("two",)),
        ]
        reference = np.asarray([1, 2], dtype=np.int64)
        prediction = np.asarray([1, 0], dtype=np.int64)
        scores = np.asarray([[0.9, 0.1], [0.0, 0.0]], dtype=np.float32)
        report = evaluator.metric_report(
            reference,
            prediction,
            scores,
            classes,
            valid_pixels=1,
            reference_pixels=2,
            frames_with_reference=1,
        )
        self.assertAlmostEqual(report["pixel_accuracy"], 0.5)
        self.assertAlmostEqual(report["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(report["macro_iou"], 0.5)
        self.assertAlmostEqual(report["prediction_coverage"], 0.5)
        self.assertAlmostEqual(report["stage_valid_coverage"], 0.5)
        self.assertEqual(report["confusion_matrix"], [[0, 1, 0], [1, 0, 0]])

    def test_region_score_mapping_keeps_invalid_unknown(self) -> None:
        region_scores = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        segmentation = np.asarray([[0, -1], [1, 0]], dtype=np.int32)
        scores, valid = evaluator.region_scores_to_pixels(
            region_scores, segmentation
        )
        prediction = evaluator.prediction_from_scores(
            scores, np.asarray([1, 2], dtype=np.int64), valid
        )
        self.assertEqual(prediction.tolist(), [[1, 0], [2, 1]])
        self.assertTrue(np.all(scores[0, 1] == 0))

    def test_absent_reference_class_is_excluded_from_macro_iou(self) -> None:
        classes = [
            evaluator.PromptClass(1, "present", ("present",)),
            evaluator.PromptClass(2, "absent", ("absent",)),
        ]
        reference = np.asarray([1, 1], dtype=np.int64)
        prediction = np.asarray([1, 2], dtype=np.int64)
        scores = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        report = evaluator.metric_report(
            reference,
            prediction,
            scores,
            classes,
            valid_pixels=2,
            reference_pixels=2,
            frames_with_reference=1,
        )
        self.assertAlmostEqual(report["macro_iou"], 0.5)
        self.assertEqual(report["macro_iou_class_count"], 1)
        self.assertIsNone(report["per_class"]["2"]["iou"])
        self.assertEqual(report["per_class"]["2"]["predicted_pixels"], 1)
        self.assertAlmostEqual(report["per_class"]["2"]["precision"], 0.0)

    def test_temporal_block_bootstrap_is_deterministic_and_paired(self) -> None:
        # Rows are reference classes [1, 2], columns are [unknown, 1, 2].
        oracle = np.asarray([[0, 5, 0], [0, 0, 5]], dtype=np.int64)
        full = np.asarray([[0, 4, 1], [0, 1, 4]], dtype=np.int64)
        pca = np.asarray([[1, 3, 1], [1, 1, 3]], dtype=np.int64)
        frame_confusions = {
            frame: {
                "sam_oracle": oracle,
                "full_teacher": full,
                "pca_teacher": pca,
            }
            for frame in range(6)
        }
        frame_coverages = {
            frame: {
                "sam_oracle": (10, 10),
                "full_teacher": (10, 10),
                "pca_teacher": (8, 10),
            }
            for frame in range(6)
        }
        first = evaluator.temporal_block_bootstrap(
            frame_confusions,
            frame_coverages,
            ["sam_oracle", "full_teacher", "pca_teacher"],
            replicates=100,
            block_size=2,
            seed=3407,
        )
        second = evaluator.temporal_block_bootstrap(
            frame_confusions,
            frame_coverages,
            ["sam_oracle", "full_teacher", "pca_teacher"],
            replicates=100,
            block_size=2,
            seed=3407,
        )
        self.assertEqual(first, second)
        self.assertGreater(
            first["paired_stage_drops"]["delta_clip"]["bootstrap_mean"], 0
        )
        self.assertGreater(
            first["paired_stage_drops"]["delta_pca"]["bootstrap_mean"], 0
        )

    def test_bootstrap_can_resample_label_scan_groups(self) -> None:
        confusion = np.asarray([[0, 3, 0], [0, 0, 3]], dtype=np.int64)
        frame_confusions = {
            frame: {
                "sam_oracle": confusion,
                "full_teacher": confusion,
                "pca_teacher": confusion,
            }
            for frame in range(4)
        }
        frame_coverages = {
            frame: {
                "sam_oracle": (6, 6),
                "full_teacher": (6, 6),
                "pca_teacher": (6, 6),
            }
            for frame in range(4)
        }
        result = evaluator.temporal_block_bootstrap(
            frame_confusions,
            frame_coverages,
            ["sam_oracle", "full_teacher", "pca_teacher"],
            replicates=20,
            block_size=3,
            seed=3407,
            frame_groups={0: 100, 1: 100, 2: 101, 3: 101},
        )
        self.assertEqual(result["resampling_unit"], "label_scan_index")
        self.assertEqual(result["block_count"], 2)
        self.assertEqual(result["group_ids"], [100, 101])
        self.assertEqual(result["frames_per_group"], [2, 2])
        self.assertIsNone(result["block_size_frames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
