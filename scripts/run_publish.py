#!/usr/bin/env python3
"""Checks that decide whether the language subspace is spelling.

Three measurements, each on the last name piece and on the mean of the name's
pieces:

- Character shuffle. Letters are reordered inside each name, including the
  English controls. The six-language subspace is rebuilt from the shuffled names.
- Spelling residual. Character n-grams fit on the training names are projected
  out of the residual, and the English offset is measured again.
- A second sentence. The name sits in "The person {name} is called".

The English offset is the distance from the English training centroid to the
span of the six training language centroids, divided by the average gap between
those centroids. A bootstrap redraws names within each language. Control
labels are -1. The layer is the training-accuracy choice from
`run_paper_strength.layer_choice`.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_english_subspace as english
import run_language_mech as mech
import run_paper_strength as paper
import common as base

OUT = base.RESULTS / "publish"
ALT_TEMPLATE = "The person {name} is called"
N_SHUFFLES = 3
N_BOOTSTRAP = 400
RIDGE = 1.0
SMALL_MODELS = [
    "distilgpt2",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
]


def collect_sites(model, tokenizer, rows, device, batch_size):
    """Block outputs at the last name piece and at the mean of the name's pieces.

    Attention and MLP writes are not stored. This pass is the one the paper
    reruns, and it only needs the two readouts.
    """
    layers = base.layer_modules(model)
    n_layers = len(layers)
    last_parts = [[] for _ in range(n_layers)]
    mean_parts = [[] for _ in range(n_layers)]
    handles = []
    stash = {"hits": None}

    def make_hook(layer_index):
        def hook(module, inputs, output):
            tensor = mech.first_tensor(output)
            last, _, mean = mech.gather_spans(tensor, stash["hits"])
            last_parts[layer_index].append(last.detach().to("cpu", torch.float32))
            mean_parts[layer_index].append(mean.detach().to("cpu", torch.float32))

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded, hits = mech.prepare_batch(tokenizer, batch, device)
            stash["hits"] = hits
            with torch.no_grad():
                model(**encoded, use_cache=False)
            print(
                f"  collected {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    packed = {
        "last": mech.stack_layers(last_parts),
        "mean": mech.stack_layers(mean_parts),
    }
    if packed["last"][0].shape[0] != len(rows):
        raise RuntimeError(
            f"activation rows {packed['last'][0].shape[0]} do not match prompts {len(rows)}"
        )
    return packed


def retarget(rows, template):
    start = template.index("{name}")
    copied = []
    for row in rows:
        item = dict(row)
        item["prompt"] = template.format(name=row["name"])
        item["name_char_start"] = start
        item["name_char_end"] = start + len(row["name"])
        copied.append(item)
    return copied


def shuffled_rows(rows, seed):
    template = base.CONFIG["prompt_template"]
    start = template.index("{name}")
    copied = []
    for index, row in enumerate(rows):
        rng = random.Random(seed + index)
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


def prepared(raw):
    """Language-probe rows plus the controls. Double-listed names are excluded."""
    rows = [
        row
        for row in raw
        if row["include_in_language_probe"] == 1 or row["is_east_african"] == 0
    ]
    languages = sorted(
        {row["language_code"] for row in rows if row["include_in_language_probe"] == 1}
    )
    language_id = {code: index for index, code in enumerate(languages)}
    labels = np.asarray(
        [
            (
                language_id[row["language_code"]]
                if row["include_in_language_probe"] == 1
                else -1
            )
            for row in rows
        ]
    )
    train = np.asarray([row["split"] == "train" for row in rows])
    names = [row["name"] for row in rows]
    return rows, languages, labels, train, names


def compact(report, layer):
    kept = {
        "layer": layer,
        "dimension": report["dimension"],
        "english_orthogonal_over_language_gap": report[
            "english_orthogonal_over_language_gap"
        ],
        "english_inplane_over_language_gap": report[
            "english_inplane_over_language_gap"
        ],
        "english_test_orthogonal_over_language_test": report[
            "english_test_orthogonal_over_language_test"
        ],
        "nearest_language_to_english_centroid": report[
            "nearest_language_to_english_centroid"
        ],
    }
    if "binary_direction_in_language_subspace" in report:
        kept["binary_direction_in_language_subspace"] = report[
            "binary_direction_in_language_subspace"
        ]
    return kept


def curve_of(bank, labels, train, test, languages):
    curve = []
    for index, features in enumerate(bank):
        report = english.geometry(
            features, labels, train, test, languages, binary_probe=False
        )
        curve.append(compact(report, index))
    return curve


def choose_layer(bank, labels, train, test):
    language = labels >= 0
    sliced = [features[language] for features in bank]
    return paper.layer_choice(sliced, labels[language], train[language], test[language])


def six_way(features, labels, train, test, languages, names, ngram_range):
    language = labels >= 0
    lang_features = features[language]
    lang_labels = labels[language]
    lang_train = train[language]
    lang_test = test[language]
    lang_names = [name for name, flag in zip(names, language) if flag]
    pred = paper.fit_predict(
        lang_features[lang_train],
        lang_labels[lang_train],
        lang_features[lang_test],
        base.SEED,
    )
    ngram = paper.char_predict(
        lang_names, lang_labels, lang_train, lang_test, ngram_range, base.SEED
    )
    truth = lang_labels[lang_test]
    matrix = confusion_matrix(truth, pred, labels=list(range(len(languages))))
    compared = paper.paired_tests(pred, ngram, truth)
    compared["confusion"] = matrix.astype(int).tolist()
    return compared


def gram_matrix(names, train, ngram_range):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=ngram_range, min_df=1)
    vectorizer.fit([names[index] for index in np.flatnonzero(train)])
    raw = vectorizer.transform(names).astype(np.float64).toarray()
    scaler = StandardScaler()
    scaler.fit(raw[train])
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0
    return (raw - scaler.mean_) / scale


def project_out(features, gram, train):
    """Remove a ridge fit of character n-grams from the residual.

    The design matrix is names by n-gram features, which is wider than it is
    tall, so the fit is solved in the dual: n-gram similarities among the
    training names, plus `RIDGE` on the diagonal. The same weights are then
    applied to every row, including the test names and the controls.
    """
    gram_train = gram[train]
    system = gram_train @ gram_train.T
    system.flat[:: system.shape[0] + 1] += RIDGE
    solved = np.linalg.solve(system, features[train].astype(np.float64))
    predicted = gram @ (gram_train.T @ solved)
    return features.astype(np.float64) - predicted


def class_means(features, labels, train, n_classes):
    rows = []
    for index in range(n_classes):
        chosen = train & (labels == index)
        rows.append(features[chosen].mean(axis=0))
    return np.stack(rows)


def pairwise(means):
    distances = []
    pairs = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            distances.append(float(np.linalg.norm(means[i] - means[j])))
            pairs.append((i, j))
    return distances, pairs


def correlation(left, right):
    """Spearman and Pearson of the 15 pairwise language-centroid gaps.

    `left` is distances in the residual. `right` is distances in a spelling
    vector (letter counts, or character 2–3 grams). Fifteen pairs is the
    whole comparison; nothing is selected after the fact.
    """
    spearman = spearmanr(left, right)
    pearson = pearsonr(left, right)
    return {
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n_pairs": len(left),
    }


def ci(draws, point):
    """95% interval for `point`, recentered on the full-sample estimate.

    Resampling class means rotates the subspace and inflates the language
    gaps, so the raw percentile interval sits below the observed offset and
    can miss it. Subtracting the bootstrap mean and adding `point` puts the
    observed offset back in the middle. `percentile_ci95` keeps the raw one.
    """
    draws = np.asarray(draws, dtype=np.float64)
    raw_low, raw_high = np.percentile(draws, [2.5, 97.5])
    centered = draws - np.mean(draws) + point
    low, high = np.percentile(centered, [2.5, 97.5])
    return {
        "n": int(len(draws)),
        "point": float(point),
        "bootstrap_mean": float(np.mean(draws)),
        "percentile_ci95": [float(raw_low), float(raw_high)],
        "ci95": [float(low), float(high)],
        "ci_above_one": bool(low > 1),
        "ci_below_one": bool(high < 1),
    }


def ratio_from_means(means, english_mean):
    basis, _ = english.centroid_basis(means)
    origin = means.mean(axis=0)
    offset = english_mean - origin
    inplane = (offset @ basis) @ basis.T
    gap = np.mean(pairwise(means)[0])
    return float(np.linalg.norm(offset - inplane) / gap), basis, origin


def bootstrap_residual_ratio(features, gram, labels, train, test, languages):
    """Redraw training names and refit the spelling projection before measuring."""
    rng = np.random.default_rng(base.SEED + 2)
    pools = [
        np.flatnonzero(train & (labels == index)) for index in range(len(languages))
    ]
    english_pool = np.flatnonzero(train & (labels < 0))
    language_test = np.flatnonzero(test & (labels >= 0))
    english_test = np.flatnonzero(test & (labels < 0))
    centroid_draws = []
    cloud_draws = []
    for _ in range(N_BOOTSTRAP):
        drawn = [rng.choice(pool, size=len(pool), replace=True) for pool in pools]
        english_drawn = rng.choice(english_pool, size=len(english_pool), replace=True)
        train_index = np.concatenate(drawn + [english_drawn])
        gram_train = gram[train_index]
        system = gram_train @ gram_train.T
        system.flat[:: system.shape[0] + 1] += RIDGE
        solved = np.linalg.solve(system, features[train_index].astype(np.float64))
        weights = gram_train.T @ solved

        def residual_rows(index):
            return features[index].astype(np.float64) - gram[index] @ weights

        means = np.stack([residual_rows(index).mean(axis=0) for index in drawn])
        english_mean = residual_rows(english_drawn).mean(axis=0)
        ratio, basis, origin = ratio_from_means(means, english_mean)
        centroid_draws.append(ratio)

        def ortho_mean(pool):
            rows = residual_rows(rng.choice(pool, size=len(pool), replace=True))
            centered = rows - origin
            leftover = centered - (centered @ basis) @ basis.T
            return np.linalg.norm(leftover, axis=1).mean()

        cloud_draws.append(float(ortho_mean(english_test) / ortho_mean(language_test)))
    full = project_out(features, gram, train)
    centroid_point, cloud_point = observed_ratios(full, labels, train, test, languages)
    return {
        "centroid_ratio": ci(centroid_draws, centroid_point),
        "cloud_ratio": ci(cloud_draws, cloud_point),
    }


def bootstrap_ratio(features, labels, train, test, languages):
    rng = np.random.default_rng(base.SEED)
    pools = [
        np.flatnonzero(train & (labels == index)) for index in range(len(languages))
    ]
    english_pool = np.flatnonzero(train & (labels < 0))
    language_test = np.flatnonzero(test & (labels >= 0))
    english_test = np.flatnonzero(test & (labels < 0))
    centroid_draws = []
    cloud_draws = []
    for _ in range(N_BOOTSTRAP):
        means = np.stack(
            [
                features[rng.choice(pool, size=len(pool), replace=True)].mean(axis=0)
                for pool in pools
            ]
        )
        english_mean = features[
            rng.choice(english_pool, size=len(english_pool), replace=True)
        ].mean(axis=0)
        basis, _ = english.centroid_basis(means)
        origin = means.mean(axis=0)
        offset = english_mean - origin
        inplane = (offset @ basis) @ basis.T
        gap = np.mean(pairwise(means)[0])
        centroid_draws.append(float(np.linalg.norm(offset - inplane) / gap))

        def ortho_mean(pool):
            drawn = features[rng.choice(pool, size=len(pool), replace=True)]
            centered = drawn - origin
            residual = centered - (centered @ basis) @ basis.T
            return np.linalg.norm(residual, axis=1).mean()

        cloud_draws.append(float(ortho_mean(english_test) / ortho_mean(language_test)))
    centroid_point, cloud_point = observed_ratios(
        features, labels, train, test, languages
    )
    return {
        "centroid_ratio": ci(centroid_draws, centroid_point),
        "cloud_ratio": ci(cloud_draws, cloud_point),
    }


def bootstrap_minimum(bank, labels, train, languages):
    """Smallest English centroid ratio across layers, redrawing training names."""
    rng = np.random.default_rng(base.SEED + 1)
    pools = [
        np.flatnonzero(train & (labels == index)) for index in range(len(languages))
    ]
    english_pool = np.flatnonzero(train & (labels < 0))
    draws = []
    layers = []
    for _ in range(N_BOOTSTRAP):
        drawn = [rng.choice(pool, size=len(pool), replace=True) for pool in pools]
        english_drawn = rng.choice(english_pool, size=len(english_pool), replace=True)
        best_ratio = None
        best_layer = 0
        for layer, features in enumerate(bank):
            means = np.stack([features[index].mean(axis=0) for index in drawn])
            english_mean = features[english_drawn].mean(axis=0)
            basis, _ = english.centroid_basis(means)
            origin = means.mean(axis=0)
            offset = english_mean - origin
            inplane = (offset @ basis) @ basis.T
            gap = np.mean(pairwise(means)[0])
            ratio = float(np.linalg.norm(offset - inplane) / gap)
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_layer = layer
        draws.append(best_ratio)
        layers.append(best_layer)
    observed = []
    for features in bank:
        means = np.stack(
            [
                features[train & (labels == index)].mean(axis=0)
                for index in range(len(languages))
            ]
        )
        english_mean = features[train & (labels < 0)].mean(axis=0)
        point, _, _ = ratio_from_means(means, english_mean)
        observed.append(point)
    summary = ci(draws, min(observed))
    summary["modal_layer"] = int(np.bincount(layers).argmax())
    return summary


def observed_ratios(features, labels, train, test, languages):
    means = np.stack(
        [
            features[train & (labels == index)].mean(axis=0)
            for index in range(len(languages))
        ]
    )
    english_mean = features[train & (labels < 0)].mean(axis=0)
    centroid, basis, origin = ratio_from_means(means, english_mean)

    def ortho(mask):
        centered = features[mask] - origin
        leftover = centered - (centered @ basis) @ basis.T
        return np.linalg.norm(leftover, axis=1).mean()

    cloud = float(ortho(test & (labels < 0)) / ortho(test & (labels >= 0)))
    return centroid, cloud


def spelling_match(features, gram, labels, train, languages):
    language = labels >= 0
    activation = class_means(
        features[language], labels[language], train[language], len(languages)
    )
    # class_means indexes labels == index, but features were already sliced to language rows,
    # whose labels are 0..5. train must be sliced the same way.
    spelling = class_means(
        gram[language], labels[language], train[language], len(languages)
    )
    activation_d, pairs = pairwise(activation)
    spelling_d, _ = pairwise(spelling)
    compared = correlation(activation_d, spelling_d)
    compared["pairs"] = [
        {
            "a": languages[i],
            "b": languages[j],
            "activation": activation_d[index],
            "spelling": spelling_d[index],
        }
        for index, (i, j) in enumerate(pairs)
    ]
    return compared


def band(curves):
    stacked = {
        key: np.vstack([[item[key] for item in curve] for curve in curves])
        for key in (
            "english_orthogonal_over_language_gap",
            "english_test_orthogonal_over_language_test",
        )
    }
    layers = [item["layer"] for item in curves[0]]
    return {
        "layer": layers,
        "centroid_mean": stacked["english_orthogonal_over_language_gap"]
        .mean(axis=0)
        .tolist(),
        "centroid_min": stacked["english_orthogonal_over_language_gap"]
        .min(axis=0)
        .tolist(),
        "centroid_max": stacked["english_orthogonal_over_language_gap"]
        .max(axis=0)
        .tolist(),
        "cloud_mean": stacked["english_test_orthogonal_over_language_test"]
        .mean(axis=0)
        .tolist(),
        "cloud_min": stacked["english_test_orthogonal_over_language_test"]
        .min(axis=0)
        .tolist(),
        "cloud_max": stacked["english_test_orthogonal_over_language_test"]
        .max(axis=0)
        .tolist(),
    }


def site_bundle(bank, labels, train, test, languages, names, clean_layer=None):
    chosen, train_scores, test_scores = choose_layer(bank, labels, train, test)
    layer = clean_layer if clean_layer is not None else chosen
    print(f"    layer {layer} (train-chosen {chosen})", flush=True)
    detailed = english.geometry(
        bank[layer], labels, train, test, languages, binary_probe=True
    )
    summary = compact(detailed, layer)
    summary["train_chosen_layer"] = chosen
    summary["train_accuracy"] = train_scores[chosen]
    summary["test_accuracy_at_train_chosen"] = test_scores[chosen]
    summary["test_accuracy_at_reported_layer"] = test_scores[layer]
    summary["six_way_vs_char_2_3gram"] = six_way(
        bank[layer], labels, train, test, languages, names, (2, 3)
    )
    summary["bootstrap"] = bootstrap_ratio(bank[layer], labels, train, test, languages)
    return summary


def residual_curve(bank, gram, labels, train, test, languages):
    curve = []
    for index, features in enumerate(bank):
        residual = project_out(features, gram, train)
        report = english.geometry(
            residual, labels, train, test, languages, binary_probe=False
        )
        curve.append(compact(report, index))
    return curve


def run_model(model_name, raw_rows, device, batch_size, token, dtype) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    rows, languages, labels, train, names = prepared(raw_rows)
    test = ~train
    print(
        f"  rows {len(rows)} languages {len(languages)} "
        f"english {(labels < 0).sum()} dtype {dtype}",
        flush=True,
    )
    grams = {
        "char_2_3gram": gram_matrix(names, train, (2, 3)),
        "char_unigram": gram_matrix(names, train, (1, 1)),
    }
    model, tokenizer = paper.load_model(model_name, token, device, dtype)

    print(" collecting clean", flush=True)
    clean = collect_sites(model, tokenizer, rows, device, batch_size)
    result = {
        "model": model_name,
        "dtype": dtype,
        "languages": languages,
        "alt_template": ALT_TEMPLATE,
        "n_shuffles": N_SHUFFLES,
        "n_bootstrap": N_BOOTSTRAP,
        "n_english_test": int(((labels < 0) & test).sum()),
        "n_language_test": int(((labels >= 0) & test).sum()),
        "sites": {},
    }
    for site in ("last", "mean"):
        print(f"  clean {site}", flush=True)
        bank = clean[site]
        chosen, _, _ = choose_layer(bank, labels, train, test)
        summary = site_bundle(bank, labels, train, test, languages, names)
        summary["by_layer"] = curve_of(bank, labels, train, test, languages)
        summary["minimum_across_layers"] = bootstrap_minimum(
            bank, labels, train, languages
        )
        observed_min = min(
            summary["by_layer"],
            key=lambda item: item["english_orthogonal_over_language_gap"],
        )
        summary["observed_minimum_layer"] = observed_min["layer"]
        summary["observed_minimum_ratio"] = observed_min[
            "english_orthogonal_over_language_gap"
        ]
        summary["spelling_distance"] = {
            name: spelling_match(bank[chosen], gram, labels, train, languages)
            for name, gram in grams.items()
        }
        summary["residual"] = {}
        for name, gram in grams.items():
            print(f"    residual {name}", flush=True)
            residual_features = project_out(bank[chosen], gram, train)
            detailed = english.geometry(
                residual_features, labels, train, test, languages, binary_probe=True
            )
            residual_summary = compact(detailed, chosen)
            before = english.geometry(
                bank[chosen], labels, train, test, languages, binary_probe=False
            )
            after_norm = detailed["english_orthogonal_distance"]
            before_norm = before["english_orthogonal_distance"]
            residual_summary["orthogonal_distance_remaining"] = (
                after_norm / before_norm if before_norm else None
            )
            residual_summary["by_layer"] = residual_curve(
                bank, gram, labels, train, test, languages
            )
            residual_summary["bootstrap"] = bootstrap_residual_ratio(
                bank[chosen], gram, labels, train, test, languages
            )
            summary["residual"][name] = residual_summary
        result["sites"][site] = {"clean_layer": chosen, "clean": summary}
        ratio = summary["english_orthogonal_over_language_gap"]
        low, high = summary["bootstrap"]["centroid_ratio"]["ci95"]
        print(f"    clean ratio {ratio:.3f} CI [{low:.3f}, {high:.3f}]", flush=True)

    del clean
    gc.collect()

    shuffle_curves = {site: [] for site in ("last", "mean")}
    shuffle_at_clean = {site: [] for site in ("last", "mean")}
    for repetition in range(N_SHUFFLES):
        seed = base.SEED if repetition == 0 else base.SEED + 10007 * repetition
        print(f" collecting shuffle {repetition + 1}/{N_SHUFFLES}", flush=True)
        shuffled = shuffled_rows(rows, seed)
        bank_all = collect_sites(model, tokenizer, shuffled, device, batch_size)
        shuffled_names = [row["name"] for row in shuffled]
        for site in ("last", "mean"):
            layer = result["sites"][site]["clean_layer"]
            print(f"  shuffle {repetition + 1} {site}", flush=True)
            curve = curve_of(bank_all[site], labels, train, test, languages)
            shuffle_curves[site].append(curve)
            detailed = english.geometry(
                bank_all[site][layer], labels, train, test, languages, binary_probe=True
            )
            item = compact(detailed, layer)
            item["bootstrap"] = bootstrap_ratio(
                bank_all[site][layer], labels, train, test, languages
            )
            item["six_way_vs_char_2_3gram"] = six_way(
                bank_all[site][layer],
                labels,
                train,
                test,
                languages,
                shuffled_names,
                (2, 3),
            )
            if repetition == 0:
                item["examples"] = [
                    {
                        "original": rows[index]["name"],
                        "shuffled": shuffled[index]["name"],
                    }
                    for index in range(6)
                ]
            shuffle_at_clean[site].append(item)
            print(
                f"    ratio {item['english_orthogonal_over_language_gap']:.3f}",
                flush=True,
            )
        del bank_all
        gc.collect()

    for site in ("last", "mean"):
        result["sites"][site]["shuffled"] = {
            "band": band(shuffle_curves[site]),
            "at_clean_layer": shuffle_at_clean[site],
        }

    print(" collecting second prompt", flush=True)
    alt_rows = retarget(rows, ALT_TEMPLATE)
    alternate = collect_sites(model, tokenizer, alt_rows, device, batch_size)
    for site in ("last", "mean"):
        print(f"  second prompt {site}", flush=True)
        summary = site_bundle(alternate[site], labels, train, test, languages, names)
        summary["by_layer"] = curve_of(alternate[site], labels, train, test, languages)
        same_layer = result["sites"][site]["clean_layer"]
        if summary["layer"] != same_layer:
            detailed = english.geometry(
                alternate[site][same_layer],
                labels,
                train,
                test,
                languages,
                binary_probe=True,
            )
            summary["at_clean_layer"] = compact(detailed, same_layer)
            summary["at_clean_layer"]["bootstrap"] = bootstrap_ratio(
                alternate[site][same_layer], labels, train, test, languages
            )
        result["sites"][site]["second_prompt"] = summary
        print(
            f"    ratio {summary['english_orthogonal_over_language_gap']:.3f} "
            f"six-way {summary['six_way_vs_char_2_3gram']['model_accuracy']:.3f}",
            flush=True,
        )

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--models", nargs="*", default=SMALL_MODELS)
    args = parser.parse_args()
    token = base.load_project_env()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows = base.load_rows(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device} rows {len(raw_rows)} dtype {args.dtype}", flush=True)
    for model_name in args.models:
        path = OUT / f"{base.slug(model_name)}.json"
        try:
            summary = run_model(
                model_name, raw_rows, device, args.batch_size, token, args.dtype
            )
        except Exception as exc:
            message = base.redact(str(exc))
            print(f"FAILED {model_name}: {message}", flush=True)
            summary = {"model": model_name, "error": message}
        path.write_text(json.dumps(summary, indent=2, default=mech.json_ready) + "\n")
        print(f" wrote {path}", flush=True)


if __name__ == "__main__":
    main()
