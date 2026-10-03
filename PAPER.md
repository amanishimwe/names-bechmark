# Where personal names sit in the residual stream

## Abstract

This paper asks where given names sit in the internal representation of five language models: distilgpt2, SmolLM2-360M, Qwen3-0.6B, Qwen2.5-0.5B, and Qwen2.5-1.5B. The readout is the residual stream on the tokenizer pieces of the name. Gisu, Kinyarwanda, Luganda, Lusoga, Runyankore, and Swahili each occupy a centroid, and those six centroids span one subspace in every model. Latin-script control names sit off that subspace by 1.4 to 2.0 language-gaps. Inside the subspace they land among the six languages, so a plot of the plane looks mixed, and a six-way probe with no control class still assigns every control name to one of the six. The same map appears in a second sentence. The off-plane location is spelling: projecting character 2–3 grams out of the residual removes it.

## 1. Question

The names are held fixed and the model is changed. The question is whether the six languages occupy one region of the residual stream, whether the control names share that region, and whether those answers agree across models.

## 2. Data and readout

The table has 1,200 prompts and a fixed split, seed `20261002`, with 606 training rows and 594 test rows. Six hundred names are given names in the six languages, 100 each. Six hundred are Latin-script controls matched on character length: 299 frequent European given names and 301 uncommon ones. Mugisha and Rukundo are listed under both Kinyarwanda and Runyankore. Those four rows stay in the training split and are excluded from the six-way probe, which leaves 298 language names and 296 control names in the test set.

The sentence is `The name {name} comes from`. A second sentence, `The person {name} is called`, repeats the map. Activations are the residual block output on the tokenizer pieces inside the name span. Two readouts are reported: the last piece, and the mean of the pieces. The final layer norm is not part of either readout.

The language subspace is the span of the six training language centroids. The control offset is the distance from the control centroid to that subspace, divided by the average gap between the language centroids. A value of 1 means the controls are as close to the subspace as the languages are to each other. The interval is a 400-draw bootstrap that resamples training names within each language. The layer reported in the table is the one with the highest six-way training accuracy. The offset is also measured at every layer.

## 3. The six languages share a subspace. The controls sit off it.

The six centroids define the subspace, so their arrangement is the language plane. The control centroid does not lie in it. At the last piece the offset is 1.43 to 1.62 language-gaps. At the mean of the pieces it is 1.73 to 1.97. Every 95% interval is above 1, including the interval for the smallest offset across layers. The five models agree. The second sentence reproduces each offset to within 0.03 gaps.

| Model | Last piece | 95% interval | Mean of pieces | 95% interval |
| --- | ---: | --- | ---: | --- |
| distilgpt2 | 1.62 | 1.48–1.76 | 1.86 | 1.69–2.03 |
| SmolLM2-360M | 1.43 | 1.32–1.54 | 1.78 | 1.62–1.95 |
| Qwen3-0.6B | 1.62 | 1.48–1.75 | 1.97 | 1.77–2.18 |
| Qwen2.5-0.5B | 1.52 | 1.39–1.66 | 1.87 | 1.69–2.05 |
| Qwen2.5-1.5B | 1.44 | 1.32–1.55 | 1.73 | 1.57–1.89 |

![Control offset across layers. Values above the dotted line are farther from the language subspace than the languages are from each other.](paper/figures/english_offset_by_layer.png)

Inside the plane, the test names from the six languages and the control names occupy the same cloud. The separation is the component this plot leaves out.

![Held-out names from the four models under 1B, projected into the six-language subspace at the last piece. Control names are drawn in black.](paper/figures/english_projection_last.png)

A probe trained only on the six languages has nowhere to put a control name, and it still assigns every control test name to one of the six, at a confidence comparable to the names that belong to those languages. On SmolLM2-360M, Qwen3-0.6B, and Qwen2.5-0.5B, Swahili receives the largest share. On distilgpt2 the assignments are more spread out. This assignment plot is for those four models.

![Language assigned to control test names by a probe that never saw a control name in training.](paper/figures/english_probe_assignment.png)

A probe that is given the controls as a seventh class recovers them at about 0.89 to 0.95 recall. The six language centroids span five dimensions. The seven-class probe uses six. The controls add a direction the six languages do not span.

## 4. That direction is spelling

Character 2–3 grams of the name string score 0.544 on the 298 language test names. The last-piece residual probe is below that line on the four models under 1B. At Qwen2.5-1.5B it is 0.527 against 0.544 (p = 0.67). The mean of the pieces is 0.594 at 1.5B (p = 0.11). Reading the residual does not beat reading the letters.

Projecting those character 2–3 grams out of the residual removes the control offset. About 0.1% of the orthogonal distance remains, and the offset falls to 0.14–0.20 gaps. Shuffling the letters inside each name, three times, drops the last-piece offset from about 1.5 gaps to 0.65–0.80. Removing only letter counts, and keeping their order, leaves an offset of 0.53–0.64 and about 26% to 32% of the distance. Letter order is what places the controls off the language plane.

![The same offset after a second sentence, a letter shuffle, and the two spelling projections. The dotted line is one language-gap.](paper/figures/english_spelling_checks.png)

Which of the six languages sit near each other is a second fact about the same representations. The Spearman correlation between centroid gaps and letter-count gaps is 0.54 to 0.84. The correlation with character 2–3 gram gaps is between −0.05 and 0.31, over 15 pairs, and is not significant. Letter counts arrange the six languages inside the plane. Letter order places the controls outside it.

![Spearman correlation between gaps among the six language centroids and gaps among their spelling vectors.](paper/figures/language_gap_vs_spelling.png)

## 5. Scope

The map is the same in all five models and in both sentences. The controls sit off the subspace the six languages span, and they overlap those languages once the view is restricted to that subspace. The off-plane direction is spelling.

The sample is about 100 names in each of six languages, in one script, with two sentences. The largest model is 1.5B parameters.

## Reproducing the figures

`scripts/run_publish.py` writes `results/publish/`. `scripts/run_english_subspace.py` writes the projections and the probe assignments. The notebook `notebooks/language_visualizations.ipynb` draws the figures. The name table and the split are `data/names.csv` and `data/config.json`.
