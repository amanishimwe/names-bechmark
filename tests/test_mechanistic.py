#!/usr/bin/env python3
"""Name-span hooks, the probe basis, and length-matched patching pairs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_language_mech as mech


class NameSpanTest(unittest.TestCase):
    def test_overlap_skips_empty_special_tokens(self):
        offsets = [[(0, 4), (4, 5), (5, 5), (5, 9)]]
        hits = mech.name_hits(offsets, [4], [9])
        self.assertEqual(hits, [[1, 3]])

    def test_a_token_that_straddles_the_name_counts(self):
        hits = mech.name_hits([[(0, 3), (3, 6), (6, 10)]], [4], [9])
        self.assertEqual(hits, [[1, 2]])

    def test_a_span_with_no_token_is_an_error(self):
        with self.assertRaises(RuntimeError):
            mech.name_hits([[(0, 3)]], [4], [9])


class HookTensorTest(unittest.TestCase):
    def test_attention_tuple_keeps_the_weights(self):
        hidden = torch.zeros(2, 3, 4)
        weights = torch.zeros(2, 3)
        found = mech.first_tensor((hidden, weights))
        self.assertTrue(torch.equal(found, hidden))
        updated = hidden + 1
        replaced = mech.replace_tensor((hidden, weights), updated)
        self.assertTrue(torch.equal(replaced[0], updated))
        self.assertTrue(torch.equal(replaced[1], weights))

    def test_a_bare_tensor_is_replaced_as_itself(self):
        hidden = torch.zeros(1, 2, 3)
        self.assertTrue(torch.equal(mech.first_tensor(hidden), hidden))
        updated = torch.ones(1, 2, 3)
        self.assertTrue(torch.equal(mech.replace_tensor(hidden, updated), updated))

    def test_component_modules_cover_gpt2_and_qwen_names(self):
        qwen = type("Layer", (), {"self_attn": "attn", "mlp": "mlp"})()
        gpt2 = type("Layer", (), {"attn": "attn", "mlp": "mlp"})()
        self.assertEqual(mech.component_modules(qwen), ("attn", "mlp"))
        self.assertEqual(mech.component_modules(gpt2), ("attn", "mlp"))

    def test_span_readouts_are_last_first_and_mean(self):
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
                [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]],
            ]
        )
        hits = [[0, 2], [1]]
        last = mech.gather_last(hidden, hits)
        self.assertTrue(torch.equal(last[0], hidden[0, 2]))
        self.assertTrue(torch.equal(last[1], hidden[1, 1]))
        last, first, mean = mech.gather_spans(hidden, hits)
        self.assertTrue(torch.equal(first[0], hidden[0, 0]))
        self.assertTrue(torch.allclose(mean[0], torch.tensor([3.0, 0.0])))


class ProbeBasisTest(unittest.TestCase):
    def test_tiny_singular_directions_are_dropped(self):
        raw = np.diag([1.0, 1.0, 1e-8])
        basis, keep = mech.row_basis(raw)
        self.assertEqual(keep, 2)
        self.assertEqual(basis.shape, (3, 2))
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(2), atol=1e-6))

    def test_probe_separates_two_clusters(self):
        train_x = np.vstack([np.zeros((20, 4)), np.ones((20, 4))])
        train_y = np.array([0] * 20 + [1] * 20)
        test_x = np.vstack([np.zeros((5, 4)), np.ones((5, 4))])
        test_y = np.array([0] * 5 + [1] * 5)
        probe = mech.Probe().fit(train_x, train_y)
        self.assertEqual(probe.accuracy(test_x, test_y), 1.0)
        self.assertEqual(probe.raw.shape[0], 1)

    def test_token_count_baseline_and_random_basis(self):
        score = mech.token_accuracy(
            np.array([1, 1, 2, 2]),
            np.array([0, 0, 1, 1]),
            np.array([1, 2]),
            np.array([0, 1]),
        )
        self.assertEqual(score, 1.0)
        basis = mech.random_basis(16, 3, draw=0)
        self.assertEqual(basis.shape, (16, 3))
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(3), atol=1e-6))
        again = mech.random_basis(16, 3, draw=0)
        self.assertTrue(np.allclose(basis, again))


class PatchPairTest(unittest.TestCase):
    def test_same_length_is_preferred(self):
        language = np.array([0, 0, 1, 1])
        n_tokens = np.array([2, 3, 2, 5])
        pool = np.array([0, 1, 2, 3])
        pairs, matched = mech.make_pairs(language, n_tokens, pool)
        by_dest = {dest: src for src, dest in pairs}
        self.assertEqual(matched, 2)
        self.assertEqual(by_dest[0], 2)
        self.assertEqual(by_dest[2], 0)
        self.assertNotEqual(language[by_dest[1]], language[1])
        self.assertNotEqual(language[by_dest[3]], language[3])


if __name__ == "__main__":
    unittest.main()
