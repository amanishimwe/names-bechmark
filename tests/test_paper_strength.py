#!/usr/bin/env python3
"""Train-chosen layer, the paired test against n-grams, and the letter shuffle."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_paper_strength as paper


class LayerChoiceTest(unittest.TestCase):
    def test_the_layer_is_chosen_on_train_accuracy(self):
        labels = np.array([0] * 30 + [1] * 30)
        train = np.zeros(len(labels), dtype=bool)
        train[:15] = True
        train[30:45] = True
        test = ~train
        train_winner = labels.astype(np.float64).reshape(-1, 1).copy()
        train_winner[test, 0] = 1 - labels[test]
        rng = np.random.default_rng(0)
        test_winner = labels.astype(np.float64).reshape(-1, 1) + rng.normal(
            scale=0.4, size=(len(labels), 1)
        )
        chosen, train_scores, test_scores = paper.layer_choice(
            [train_winner, test_winner], labels, train, test
        )
        self.assertEqual(chosen, 0)
        self.assertGreater(train_scores[0], train_scores[1])
        self.assertGreater(test_scores[1], test_scores[0])


class PairedTest(unittest.TestCase):
    def test_a_probe_that_always_wins(self):
        truth = np.array([0, 1, 0, 1, 0, 1])
        report = paper.paired_tests(truth, 1 - truth, truth)
        self.assertEqual(report["difference"], 1.0)
        self.assertEqual(report["mcnemar_model_only"], 6)
        self.assertEqual(report["mcnemar_ngram_only"], 0)
        self.assertLess(report["mcnemar_p"], 0.05)
        self.assertTrue(report["ci_excludes_zero"])

    def test_identical_predictions_have_no_disagreements(self):
        truth = np.array([0, 1, 0, 1])
        report = paper.paired_tests(truth, truth, truth)
        self.assertEqual(report["difference"], 0.0)
        self.assertEqual(report["mcnemar_p"], 1.0)
        self.assertFalse(report["ci_excludes_zero"])

    def test_probe_seeds_do_not_move_a_convex_fit(self):
        labels = np.array([0] * 20 + [1] * 20)
        features = labels.astype(np.float64).reshape(-1, 1)
        train = np.zeros(len(labels), dtype=bool)
        train[:10] = True
        train[20:30] = True
        report = paper.seed_accuracies(features, labels, train, ~train)
        self.assertEqual(report["n"], paper.N_SEEDS)
        self.assertEqual(report["n_unique"], 1)


class ShuffleTest(unittest.TestCase):
    def test_letters_move_and_the_language_label_stays(self):
        row = {
            "name": "Mugisha",
            "language_code": "kinyarwanda",
            "split": "train",
        }
        shuffled = paper.shuffled_copy([row])[0]
        self.assertEqual(shuffled["language_code"], "kinyarwanda")
        self.assertEqual(sorted(shuffled["name"]), sorted(row["name"]))
        self.assertEqual(
            shuffled["name_char_end"] - shuffled["name_char_start"],
            len(row["name"]),
        )
        self.assertIn(shuffled["name"], shuffled["prompt"])

    def test_character_ngrams_separate_two_spellings(self):
        names = ["aba"] * 12 + ["zyz"] * 12
        labels = np.array([0] * 12 + [1] * 12)
        train = np.zeros(len(labels), dtype=bool)
        train[:8] = True
        train[12:20] = True
        pred = paper.char_predict(names, labels, train, ~train, (2, 3), seed=0)
        self.assertEqual(float(np.mean(pred == labels[~train])), 1.0)


if __name__ == "__main__":
    unittest.main()
