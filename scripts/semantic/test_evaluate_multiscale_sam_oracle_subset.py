#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_multiscale_sam_oracle_subset import (
    aggregate_any_level,
    any_level_recoverability,
    build_joint_partition,
    summarize_area_order,
)


class MultiscaleSamOracleTest(unittest.TestCase):
    def test_joint_partition_uses_cross_level_intersections(self) -> None:
        horizontal = np.asarray([[0, 0], [1, 1]], dtype=np.int32)
        vertical = np.asarray([[0, 1], [0, 1]], dtype=np.int32)
        joint = build_joint_partition([horizontal, vertical])
        self.assertEqual(len(np.unique(joint)), 4)
        self.assertTrue(np.all(joint >= 0))

    def test_joint_partition_keeps_fully_uncovered_pixels_unknown(self) -> None:
        first = np.asarray([[-1, 0], [-1, 0]], dtype=np.int32)
        second = np.asarray([[-1, -1], [2, 2]], dtype=np.int32)
        joint = build_joint_partition([first, second])
        self.assertEqual(int(joint[0, 0]), -1)
        self.assertTrue(np.all(joint[[0, 1], [1, 0]] >= 0))

    def test_any_level_recoverability_is_explicitly_label_leaking(self) -> None:
        reference = np.asarray([[1, 2], [2, 1]], dtype=np.int16)
        reference_valid = np.ones((2, 2), dtype=np.bool_)
        first = np.asarray([[1, 1], [2, 1]], dtype=np.int16)
        second = np.asarray([[2, 2], [1, 1]], dtype=np.int16)
        valid = np.ones((2, 2), dtype=np.bool_)
        record = any_level_recoverability(
            reference,
            reference_valid,
            [first, second],
            [valid, valid],
            np.asarray([1, 2], dtype=np.int64),
        )
        self.assertEqual(record["reference_pixels"], 4)
        self.assertEqual(record["covered_by_any_level"], 4)
        self.assertEqual(record["oracle_correct_in_at_least_one_level"], 4)
        aggregate = aggregate_any_level([record], {"1": "one", "2": "two"})
        self.assertEqual(aggregate["any_level_oracle_recall"], 1.0)
        self.assertIn("not a coherent prediction", aggregate["interpretation_guardrail"])

    def test_area_order_audit_does_not_assume_aliases_are_literal(self) -> None:
        report = {
            "frame_records": [
                {
                    "frame_index": 4,
                    "levels": {
                        "s": {"area_pixels_median": 10.0},
                        "m": {"area_pixels_median": 20.0},
                        "l": {"area_pixels_median": 30.0},
                    },
                },
                {
                    "frame_index": 9,
                    "levels": {
                        "s": {"area_pixels_median": 40.0},
                        "m": {"area_pixels_median": 20.0},
                        "l": {"area_pixels_median": 30.0},
                    },
                },
            ]
        }
        audit = summarize_area_order(report, [4, 9])
        self.assertEqual(audit["frames_with_s_m_l_nondecreasing_median_area"], 1)
        self.assertEqual(
            audit["fraction_with_s_m_l_nondecreasing_median_area"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
