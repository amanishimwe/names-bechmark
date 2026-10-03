#!/usr/bin/env python3
"""Checks that make the language result stronger.

Confusion matrices, a paired test of the residual probe against character
n-grams, and a within-name character shuffle. The layer is chosen on the
training split.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_language_mech as mech
import run_subspace as base

OUT = base.RESULTS / "paper_strength"
N_SEEDS = 10
N_BOOTSTRAP = 20
N_TEST_RESAMPLES = 10000
LARGER_MODEL = "Qwen/Qwen2.5-1.5B"


def fit_predict(train_x, train_y, test_x, seed: int):
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_x)
    clf = LogisticRegression(max_iter=500, tol=1e-3, random_state=seed)
    clf.fit(train_s, train_y)
    return clf.predict(scaler.transform(test_x))


def char_predict(names, labels, train, test, ngram_range, seed: int):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1)
    train_index = np.flatnonzero(train)
    test_index = np.flatnonzero(test)
    train_x = vectorizer.fit_transform([names[index] for index in train_index])
    test_x = vectorizer.transform([names[index] for index in test_index])
    clf = LogisticRegression(max_iter=500, tol=1e-3, random_state=seed)
    clf.fit(train_x, labels[train])
    return clf.predict(test_x)


def recalls(confusion, languages):
    scores = {}
    for index, name in enumerate(languages):
        total = int(confusion[index].sum())
        if total:
            scores[name] = float(confusion[index, index] / total)
    return scores


def paired_tests(model_pred, ngram_pred, truth):
    model_correct = model_pred == truth
    ngram_correct = ngram_pred == truth
    model_only = int(np.sum(model_correct & ~ngram_correct))
    ngram_only = int(np.sum(~model_correct & ngram_correct))
    compared = model_only + ngram_only
    if compared == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(model_only, compared, 0.5, alternative="two-sided").pvalue)
    difference = model_correct.astype(np.float64) - ngram_correct.astype(np.float64)
    rng = np.random.default_rng(base.SEED)
    draws = rng.integers(0, len(difference), size=(N_TEST_RESAMPLES, len(difference)))
    samples = difference[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "model_accuracy": float(np.mean(model_correct)),
        "ngram_accuracy": float(np.mean(ngram_correct)),
        "difference": float(np.mean(difference)),
        "bootstrap_ci95": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "mcnemar_model_only": model_only,
        "mcnemar_ngram_only": ngram_only,
        "mcnemar_p": p_value,
    }


def layer_choice(bank, labels, train, test):
    train_scores = []
    test_scores = []
    for features in bank:
        scaler = StandardScaler()
        train_s = scaler.fit_transform(features[train])
        clf = LogisticRegression(max_iter=500, tol=1e-3, random_state=base.SEED)
        clf.fit(train_s, labels[train])
        pred_train = clf.predict(train_s)
        pred_test = clf.predict(scaler.transform(features[test]))
        train_scores.append(float(np.mean(pred_train == labels[train])))
        test_scores.append(float(np.mean(pred_test == labels[test])))
    return int(np.argmax(train_scores)), train_scores, test_scores


def seed_accuracies(features, labels, train, test):
    scores = []
    for seed in range(N_SEEDS):
        pred = fit_predict(features[train], labels[train], features[test], seed)
        scores.append(float(np.mean(pred == labels[test])))
    return {
        "n": N_SEEDS,
        "accuracies": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_unique": len(set(np.round(scores, 6))),
    }


def train_bootstrap(features, labels, train, test):
    rng = np.random.default_rng(base.SEED)
    train_index = np.flatnonzero(train)
    scores = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.choice(train_index, size=len(train_index), replace=True)
        if len(np.unique(labels[drawn])) < 2:
            continue
        pred = fit_predict(features[drawn], labels[drawn], features[test], base.SEED)
        scores.append(float(np.mean(pred == labels[test])))
    return {
        "n": len(scores),
        "accuracies": scores,
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
    }


def site_report(features, labels, train, test, languages, names, ngram_pred):
    pred = fit_predict(features[train], labels[train], features[test], base.SEED)
    truth = labels[test]
    matrix = confusion_matrix(truth, pred, labels=list(range(len(languages))))
    return {
        "accuracy": float(np.mean(pred == truth)),
        "confusion": matrix.astype(int).tolist(),
        "recall": recalls(matrix, languages),
        "seeds": seed_accuracies(features, labels, train, test),
        "train_bootstrap": train_bootstrap(features, labels, train, test),
        "vs_char_ngram": paired_tests(pred, ngram_pred, truth),
    }


def shuffled_copy(rows):
    template = base.CONFIG["prompt_template"]
    start = template.index("{name}")
    copied = []
    for index, row in enumerate(rows):
        rng = random.Random(base.SEED + index)
        chars = list(row["name"])
        rng.shuffle(chars)
        name = "".join(chars)
        item = dict(row)
        item["name"] = name
        item["prompt"] = template.format(name=name)
        item["name_char_start"] = start
        item["name_char_end"] = start + len(name)
        copied.append(item)
    return copied


def load_model(model_name, token, device, dtype):
    hub = {"token": token} if token else {}
    tokenizer = base.AutoTokenizer.from_pretrained(model_name, **hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    try:
        model = base.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, **hub)
    except TypeError:
        model = base.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, **hub)
    model.to(device)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return model, tokenizer


def run_model(model_name, rows, device, batch_size, token, dtype) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    languages = sorted({row["language_code"] for row in rows})
    language_id = {code: index for index, code in enumerate(languages)}
    labels = np.asarray([language_id[row["language_code"]] for row in rows])
    train = np.asarray([row["split"] == "train" for row in rows])
    test = ~train
    names = [row["name"] for row in rows]
    ngram_pred = char_predict(names, labels, train, test, (2, 3), base.SEED)
    unigram_pred = char_predict(names, labels, train, test, (1, 1), base.SEED)
    print(
        f"  char 2-3gram {np.mean(ngram_pred == labels[test]):.3f} "
        f"unigram {np.mean(unigram_pred == labels[test]):.3f}",
        flush=True,
    )

    model, tokenizer = load_model(model_name, token, device, dtype)
    print(" collecting clean activations", flush=True)
    clean = mech.collect_clean(model, tokenizer, rows, device, batch_size)

    def analyze(bank, title):
        chosen, train_scores, test_scores = layer_choice(bank, labels, train, test)
        print(
            f"  {title} train-chosen layer {chosen} "
            f"test {test_scores[chosen]:.3f}",
            flush=True,
        )
        report = site_report(bank[chosen], labels, train, test, languages, names, ngram_pred)
        report["layer"] = chosen
        report["train_accuracy_at_layer"] = train_scores[chosen]
        report["by_layer_test"] = test_scores
        report["test_best_layer"] = int(np.argmax(test_scores))
        report["test_best_accuracy"] = test_scores[report["test_best_layer"]]
        return report

    last = analyze(clean["last"], "last piece")
    mean = analyze(clean["mean"], "mean of pieces")

    print(" collecting shuffled names", flush=True)
    shuffled_rows = shuffled_copy(rows)
    shuffled = mech.collect_clean(model, tokenizer, shuffled_rows, device, batch_size)
    shuffled_names = [row["name"] for row in shuffled_rows]
    shuffled_ngram = char_predict(shuffled_names, labels, train, test, (2, 3), base.SEED)
    shuffled_unigram = char_predict(shuffled_names, labels, train, test, (1, 1), base.SEED)

    def shuffled_site(bank, layer):
        _, train_scores, test_scores = layer_choice(bank, labels, train, test)
        chosen = int(np.argmax(train_scores))
        pred = fit_predict(bank[layer][train], labels[train], bank[layer][test], base.SEED)
        truth = labels[test]
        matrix = confusion_matrix(truth, pred, labels=list(range(len(languages))))
        return {
            "at_clean_layer": layer,
            "accuracy_at_clean_layer": float(np.mean(pred == truth)),
            "confusion_at_clean_layer": matrix.astype(int).tolist(),
            "recall_at_clean_layer": recalls(matrix, languages),
            "train_best_layer": chosen,
            "accuracy_at_train_best_layer": test_scores[chosen],
        }

    shuffled_last = shuffled_site(shuffled["last"], last["layer"])
    shuffled_mean = shuffled_site(shuffled["mean"], mean["layer"])
    print(
        "  shuffled last "
        f"{shuffled_last['accuracy_at_clean_layer']:.3f} mean "
        f"{shuffled_mean['accuracy_at_clean_layer']:.3f} "
        f"char {np.mean(shuffled_ngram == labels[test]):.3f}",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "model": model_name,
        "dtype": dtype,
        "n_language_train": int(train.sum()),
        "n_language_test": int(test.sum()),
        "languages": languages,
        "chance": 1 / len(languages),
        "orthography": {
            "char_2_3gram": float(np.mean(ngram_pred == labels[test])),
            "char_unigram": float(np.mean(unigram_pred == labels[test])),
        },
        "last": last,
        "mean": mean,
        "shuffled": {
            "examples": [
                {"original": rows[index]["name"], "shuffled": shuffled_rows[index]["name"]}
                for index in range(8)
            ],
            "char_2_3gram": float(np.mean(shuffled_ngram == labels[test])),
            "char_unigram": float(np.mean(shuffled_unigram == labels[test])),
            "last": shuffled_last,
            "mean": shuffled_mean,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()
    token = base.load_project_env()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in base.load_rows(0) if row["include_in_language_probe"] == 1]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device} language rows {len(rows)} dtype {args.dtype}", flush=True)
    for model_name in args.models:
        path = OUT / f"{base.slug(model_name)}.json"
        try:
            summary = run_model(model_name, rows, device, args.batch_size, token, args.dtype)
        except Exception as exc:
            message = base.redact(str(exc))
            print(f"FAILED {model_name}: {message}", flush=True)
            summary = {"model": model_name, "error": message}
        path.write_text(json.dumps(summary, indent=2, default=mech.json_ready) + "\n")
        print(f" wrote {path}", flush=True)


if __name__ == "__main__":
    main()
