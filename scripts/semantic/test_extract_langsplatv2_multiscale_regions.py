#!/usr/bin/env python3

import os
from pathlib import Path
import unittest

import numpy as np
import torch

from extract_langsplatv2_multiscale_regions import (
    UPSTREAM_COMMIT,
    configure_import_path,
    load_upstream_api,
    make_segmentation_map,
    official_mask_nms_indices,
)


class LangSplatMultiscaleExtractionTest(unittest.TestCase):
    def test_nms_keeps_high_score_duplicate_and_disjoint_mask(self) -> None:
        masks = torch.zeros((3, 4, 4), dtype=torch.bool)
        masks[0, :2, :2] = True
        masks[1, :2, :2] = True
        masks[2, 2:, 2:] = True
        selected = official_mask_nms_indices(
            masks,
            torch.tensor([0.90, 0.80, 0.95]),
            iou_threshold=0.8,
            score_threshold=0.7,
            inner_threshold=0.5,
        )
        self.assertEqual(selected.tolist(), [0, 2])

    def test_segmentation_preserves_upstream_last_mask_overwrite(self) -> None:
        broad = np.ones((3, 3), dtype=np.bool_)
        local = np.zeros((3, 3), dtype=np.bool_)
        local[1, 1] = True
        segmentation = make_segmentation_map(
            [{"segmentation": broad}, {"segmentation": local}], (3, 3)
        )
        self.assertEqual(int(segmentation[0, 0]), 0)
        self.assertEqual(int(segmentation[1, 1]), 1)

    def test_pinned_upstream_fork_is_detected(self) -> None:
        raw_root = os.environ.get("LANGSPLAT_SAM_ROOT")
        if not raw_root:
            self.skipTest("LANGSPLAT_SAM_ROOT is not set")
        configure_import_path(Path(raw_root))
        _, _, source_path, digest = load_upstream_api()
        self.assertEqual(source_path.name, "automatic_mask_generator.py")
        self.assertEqual(len(digest), 64)
        self.assertIn(UPSTREAM_COMMIT[:8], raw_root)


if __name__ == "__main__":
    unittest.main()
