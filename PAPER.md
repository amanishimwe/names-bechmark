# The language subspace of these personal names is spelling

## Abstract

A linear probe can tell Gisu, Kinyarwanda, Luganda, Lusoga, Runyankore, and Swahili apart from the residual stream at a personal name, at well above the chance rate of one in six. Character 2–3 grams fit on the name string match or beat that probe. The 600 Latin-script control names sit 1.4 to 2.0 language-gaps off the plane spanned by the six language centroids, in every model from distilgpt2 to Qwen2.5-1.5B, and the same offset appears in a second sentence. Projecting character 2–3 grams out of the residual removes that offset: about 0.1% of the orthogonal distance remains. Shuffling the letters inside each name pulls the last-piece offset from about 1.5 gaps down to about 0.7. Distances among the six languages track letter counts, not letter order. The geometry is spelling.

## 1. Data

The table has 1,200 prompts and a fixed split, seed `20261002`, with 606 training rows and 594 test rows. Six hundred names are given names in the six languages, 100 each. Six hundred are Latin-script controls matched to those names on character length: 299 frequent European given names and 301 uncommon ones. Mugisha and Rukundo are listed under both Kinyarwanda and Runyankore. Those four rows stay in the training split and are excluded from the six-way probe, which leaves 298 language names and 296 control names in the test set.

The probe sentence is `The name {name} comes from`. A second sentence, `The person {name} is called`, is used only as a replication. Activations are the residual block output on the tokenizer pieces that fall inside the name span. Two readouts are reported: the last piece, and the mean of the pieces. The final layer norm is not part of either readout.

## 2. Method

Five causal language models are used: distilgpt2, SmolLM2-360M, Qwen3-0.6B, Qwen2.5-0.5B, and Qwen2.5-1.5B. The 1.5B model is run in float16. The others are float32.

The six-way probe is logistic regression on standardized residual vectors, fit on the training names and scored on the test names. The layer is the one with the highest training accuracy. Chance is 1/6. The spelling baseline is a logistic regression on character 2–3 grams of the name string, with the same split. The comparison is paired on the test names: an exact McNemar test, and a 10,000-draw bootstrap interval for the difference in accuracy.

The language subspace is the span of the six training language centroids. The control offset is the distance from the control-name centroid to that subspace, divided by the average gap between the language centroids. A value of 1 means the controls are as close to the subspace as the languages are to each other. The interval is a 400-draw bootstrap that resamples training names within each language and recenters the percentile interval on the full-sample estimate. The same ratio is recomputed at every layer, and the bootstrap is repeated for the smallest ratio across layers.

Three checks ask whether that offset is spelling.

- The letters inside each name are shuffled, three times, and the subspace is rebuilt. The first shuffle uses the dataset seed.
- Character n-grams fit on the training names are projected out of the residual with a ridge penalty of 1, and the offset is measured in what remains. Letter order is character 2–3 grams. Letter counts are character unigrams.
- The six language centroids are compared with the centroids of those same n-gram vectors. The statistic is the Spearman correlation over the 15 pairs of languages.

## 3. The probe does not beat spelling

Character 2–3 grams score 0.544 on the 298 test names. The last-piece residual probe is below that line on every model under 1B, and the paired test rejects equal accuracy. At Qwen2.5-1.5B the last-piece probe is 0.527 against the same 0.544, and the test no longer separates them (p = 0.67). The mean of the name's pieces is closer to the spelling line throughout. At 1.5B it is 0.594 (p = 0.11). The second sentence at 1.5B reaches 0.604 (p = 0.05). Averaging the pieces does not establish a win over spelling.

| Model | Last piece | p | Mean of pieces | p |
| --- | ---: | ---: | ---: | ---: |
| distilgpt2 | 0.433 | 0.0009 | 0.497 | 0.18 |
| SmolLM2-360M | 0.470 | 0.032 | 0.503 | 0.25 |
| Qwen3-0.6B | 0.413 | 6.5e-5 | 0.577 | 0.29 |
| Qwen2.5-0.5B | 0.399 | 1.2e-5 | 0.527 | 0.64 |
| Qwen2.5-1.5B | 0.527 | 0.67 | 0.594 | 0.11 |

Shuffling letters inside each name drops the last-piece probe to between 0.18 and 0.32. Chance is 0.167, and the shuffled character 2–3 gram baseline on those same names is 0.25 to 0.29. The mean readout falls to between 0.28 and 0.40. Letter order carries the six-way signal that the probe was reading.

## 4. Control names sit off the language plane, and spelling puts them there

On the original sentence, the control centroid is 1.43 to 1.62 language-gaps off the subspace at the last piece, and 1.73 to 1.97 gaps at the mean of the pieces. Every 95% interval is above 1, including the interval for the smallest ratio across layers. The second sentence reproduces the offset to within 0.03 gaps on every model.

![Control offset after the spelling checks. The dotted line is one language-gap.](paper/figures/english_spelling_checks.png)

Shuffling the letters cuts the last-piece offset to between 0.65 and 0.80. The mean-of-pieces offset falls to about 1. Projecting character 2–3 grams out of the residual leaves an offset of 0.14 to 0.20 gaps, and about 0.1% of the original orthogonal distance. Projecting out letter counts, and leaving order in the name, leaves an offset of 0.53 to 0.64 and about 26% to 32% of the distance. The separation of the controls from the six languages is letter order.

![The control offset across layers. The band is the range of three letter shuffles.](paper/figures/english_offset_by_layer.png)

The arrangement of the six languages is a different spelling fact. The Spearman correlation between centroid gaps and letter-count gaps is 0.54 to 0.84. The same correlation for character 2–3 grams is between −0.05 and 0.31, and none of those 15-pair tests is significant. Letter counts say which of the six languages sit near each other. Letter order says where the controls sit relative to all six.

![Spearman correlation between language-centroid gaps and spelling gaps.](paper/figures/language_gap_vs_spelling.png)

## 5. What the result supports

The six-language direction in these residuals is a spelling direction. A probe can read it, a character model reads it at least as well, and destroying letter order removes both the probe's accuracy and the control names' offset from the language plane. The finding is the same from distilgpt2 through Qwen2.5-1.5B, and it does not depend on the sentence wrapped around the name.

The sample is about 100 names in each of six languages, in one script, with two sentences. The largest model is 1.5B parameters. Those bounds are the right scope for the claim. Within them, the subspace is spelling.

## Reproducing the figures

`scripts/run_publish.py` writes `results/publish/`. The notebook `notebooks/language_visualizations.ipynb` draws the three figures above from those files. The name table and the split are `data/names.csv` and `data/config.json`.
