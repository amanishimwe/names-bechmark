#!/usr/bin/env python3
"""Loader, filenames, and which module holds the transformer blocks."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common as base


class RedactAndNamesTest(unittest.TestCase):
    def test_redact_hides_a_hub_token(self):
        text = "download failed for hf_AbC123xyz"
        cleaned = base.redact(text)
        self.assertNotIn("AbC123xyz", cleaned)
        self.assertIn("hf_[redacted]", cleaned)

    def test_slug_is_safe_as_a_filename(self):
        self.assertEqual(base.slug("Qwen/Qwen2.5-0.5B"), "Qwen__Qwen2.5-0.5B")


class LoadRowsTest(unittest.TestCase):
    def test_full_table_matches_the_config(self):
        rows = base.load_rows(0)
        self.assertEqual(len(rows), base.CONFIG["n_instances"])
        self.assertEqual(
            sum(row["split"] == "train" for row in rows), base.CONFIG["n_train"]
        )
        self.assertEqual(
            sum(row["split"] == "test" for row in rows), base.CONFIG["n_test"]
        )
        self.assertNotIn("places", rows[0])
        self.assertIsInstance(rows[0]["include_in_language_probe"], int)
        self.assertIsInstance(rows[0]["name_char_start"], int)

    def test_dual_listed_names_stay_in_the_file(self):
        rows = base.load_rows(0)
        dual = [
            row
            for row in rows
            if row["is_east_african"] == 1 and row["include_in_language_probe"] == 0
        ]
        self.assertEqual(
            sorted(row["name"] for row in dual),
            ["Mugisha", "Mugisha", "Rukundo", "Rukundo"],
        )

    def test_limit_keeps_each_split_and_set(self):
        full = base.load_rows(0)
        limited = base.load_rows(1)
        expected = {(row["split"], row["set"]) for row in full}
        counts = Counter((row["split"], row["set"]) for row in limited)
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(count == 1 for count in counts.values()))


class LayerLookupTest(unittest.TestCase):
    def test_gpt2_and_qwen_layouts(self):
        gpt2 = type("Model", (), {})()
        gpt2.transformer = type("Block", (), {"h": ["gpt-0", "gpt-1"]})()
        self.assertEqual(base.layer_modules(gpt2), ["gpt-0", "gpt-1"])

        neox = type("Model", (), {})()
        neox.gpt_neox = type("Block", (), {"layers": ["neox"]})()
        self.assertEqual(base.layer_modules(neox), ["neox"])

        qwen = type("Model", (), {})()
        qwen.model = type("Inner", (), {"layers": ["qwen"]})()
        self.assertEqual(base.layer_modules(qwen), ["qwen"])

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(RuntimeError):
            base.layer_modules(object())

    def test_offset_rows_accept_a_tensor(self):
        mapping = torch.tensor([[[0, 4], [4, 9]]])
        rows = base.as_offset_rows(mapping)
        self.assertEqual(rows, [[[0, 4], [4, 9]]])


if __name__ == "__main__":
    unittest.main()
