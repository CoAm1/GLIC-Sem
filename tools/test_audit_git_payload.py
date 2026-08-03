#!/usr/bin/env python3
"""Unit tests for the Git payload guard."""

from __future__ import annotations

import unittest
from pathlib import Path

import audit_git_payload as guard


class GitPayloadAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_source_and_document_are_allowed(self) -> None:
        self.assertEqual(
            guard.audit_path(self.root, "tools/audit_git_payload.py", 1024 * 1024),
            [],
        )
        self.assertEqual(
            guard.audit_path(
                self.root,
                "docs/OPEN_VOCAB_SEMANTIC_ERROR_DECOMPOSITION_PROTOCOL.md",
                1024 * 1024,
            ),
            [],
        )

    def test_model_dataset_result_and_secret_are_rejected(self) -> None:
        cases = ("model.pth", "scene.ply", "results/metrics.json", "yuhet")
        for relative in cases:
            with self.subTest(relative=relative):
                self.assertTrue(guard.audit_path(self.root, relative, 1024))

    def test_oversized_source_is_rejected(self) -> None:
        self.assertTrue(
            guard.audit_path(self.root, "tools/audit_git_payload.py", 1)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
