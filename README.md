# Language geometry of personal names

Where given names sit in the residual stream of distilgpt2, SmolLM2-360M, Qwen3-0.6B, Qwen2.5-0.5B, and Qwen2.5-1.5B, and whether that residual treats the six languages differently from frequent European names. The write-up is [PAPER.md](PAPER.md). The plots are drawn by [notebooks/language_visualizations.ipynb](notebooks/language_visualizations.ipynb). Figures used in the paper are in `paper/figures/`.

## Layout

| Path | Role |
| --- | --- |
| `data/names.csv` | 1,200 prompts and the fixed split |
| `data/config.json` | Seed, prompt template, and split sizes |
| `scripts/common.py` | Shared loader, `.env`, and residual-layer lookup. The other scripts import it |
| `scripts/run_language_mech.py` | Ablation, activation patching, attention versus MLP |
| `scripts/run_paper_strength.py` | Confusion matrices and the paired test against character n-grams |
| `scripts/run_english_subspace.py` | Where the control names sit relative to the six-language subspace |
| `scripts/run_publish.py` | Offset, letter shuffle, n-gram projection, and the second sentence |
| `scripts/run_fairness.py` | Recognition and continuation gaps versus frequent names |
| `results/` | JSON written by the scripts, and the notebook figures. Not in the repository |

Python is formatted with Black.

## Data

Six hundred names are given names in Gisu, Kinyarwanda, Luganda, Lusoga, Runyankore, and Swahili, 100 each. The other 600 are Latin-script controls matched on character length: 299 frequent European given names (`control_common`) and 301 uncommon ones (`control_rare`).

Every row uses the same prompt:

```text
The name {name} comes from
```

`name_char_start` and `name_char_end` mark the name inside that string. Activations are read at the last tokenizer piece inside the span, and at the mean of the pieces inside the span. Token counts depend on the tokenizer. The final layer norm is not part of the readout.

| Column | Use |
| --- | --- |
| `language_code` | Six-way probe, only where `include_in_language_probe` is 1 |
| `is_east_african` | 1 for the 600 names in the six languages. The controls are 0 |
| `set` | Language group, `control_common`, or `control_rare` |
| `split` | `train` fits the probe (606 rows). `test` scores it (594 rows) |
| `char_len` | Character length used to match controls to the six languages |
| `gender` | Recorded on the row. The probes do not use it |
| `place_labels` | Stored on the row. The scripts do not score it |

The split seed is `20261002`. Mugisha and Rukundo are listed under both Kinyarwanda and Runyankore, so those four rows stay in `train` and are left out of the six-way probe. The six-way test set has 298 names. The control test set has 296.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt scipy
```

`scipy` is imported by `run_publish.py` and `run_paper_strength.py` and is not listed in `requirements.txt`. Gated Hugging Face models read `HF_TOKEN` from `.env` in the project root. That file is not committed. The scripts use Apple MPS when it is available, and the CPU otherwise. Qwen2.5-1.5B is run in float16 with batch size 4. The smaller models use float32 and batch size 16.

## Experiments

Each script writes one JSON file per model under `results/`. The notebook reads those files.

`scripts/run_language_mech.py` scores the six-way language label. It projects a direction out of the residual and refits the probe, patches the name residual from one language into another, separates the attention write from the MLP write, and compares the probe with token position and character n-grams. Results go to `results/language_mech/`. The default model is distilgpt2. The four models under 1B:

```bash
python scripts/run_language_mech.py --batch-size 16 \
  --models distilgpt2 HuggingFaceTB/SmolLM2-360M Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B
```

`scripts/run_paper_strength.py` writes confusion matrices, a paired test of the residual probe against character 2–3 grams, and a within-name character shuffle. The layer is the one with the highest six-way training accuracy. Results go to `results/paper_strength/`. Pass the models explicitly:

```bash
python scripts/run_paper_strength.py --batch-size 16 \
  --models distilgpt2 HuggingFaceTB/SmolLM2-360M Qwen/Qwen3-0.6B Qwen/Qwen2.5-0.5B
```

`scripts/run_english_subspace.py` builds the subspace from the six training language centroids and measures how far the control names sit from it. It also records the two-dimensional projection and the language a six-way probe assigns to each control name. Results go to `results/english_subspace/`. The default run is the four models under 1B, in float32:

```bash
python scripts/run_english_subspace.py --batch-size 16
```

`scripts/run_publish.py` is the check behind the paper. It rebuilds the control offset, shuffles the letters inside each name three times, projects character n-grams out of the residual, and repeats the offset in a second sentence, `The person {name} is called`. Results go to `results/publish/`.

```bash
python scripts/run_publish.py --batch-size 16
python scripts/run_publish.py --batch-size 4 --dtype float16 --models Qwen/Qwen2.5-1.5B
```

`scripts/run_fairness.py` compares the six languages with the 299 frequent names only. It measures how probable the name is, per character, in the dataset sentence, and how much the model prefers a positive continuation after `The person named {name} `. The three pairs are fixed: passed versus failed the exam, was hired versus was rejected, and is trusted versus is doubted. On the six-language names, every name piece at every layer is then replaced by the frequent-name centroid, and the continuations are scored again. A random vector of the same size is the control intervention. Frequent names are left unchanged. Results go to `results/fairness/`.

```bash
python scripts/run_fairness.py --batch-size 16
python scripts/run_fairness.py --batch-size 4 --dtype float16 --models Qwen/Qwen2.5-1.5B
```
