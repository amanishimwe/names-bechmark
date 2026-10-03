# Language geometry of personal names

A six-language probe of open-weight language models, and the spelling checks that explain it. The write-up is in [PAPER.md](PAPER.md). The plots are in [notebooks/language_visualizations.ipynb](notebooks/language_visualizations.ipynb).

The names are in `data/names.csv`. There are 1,200 prompts. Six hundred are given names in Gisu, Kinyarwanda, Luganda, Lusoga, Runyankore, and Swahili, 100 each. The other 600 are Latin-script controls matched on character length: 299 frequent European given names and 301 uncommon ones. The paper treats those controls as the comparison set.

Every row uses the same prompt:

```text
The name {name} comes from
```

`name_char_start` and `name_char_end` mark the name inside that string. Activations are read at the last tokenizer piece inside the span, and at the mean of the pieces inside the span. Token counts depend on the tokenizer.

| Column | Use |
| --- | --- |
| `language_code` | Six-way probe, only where `include_in_language_probe` is 1 |
| `is_east_african` | Marks the 600 names in the six languages. The controls are 0 |
| `split` | `train` fits the probe (606 rows). `test` scores it (594 rows) |

The split seed is `20261002`. Mugisha and Rukundo are listed under both Kinyarwanda and Runyankore, so those four rows stay in `train` and are left out of the six-way probe. The six-way test set has 298 names. The control test set has 296.

## Rerun

```bash
python scripts/run_publish.py --batch-size 16
python scripts/run_publish.py --batch-size 4 --dtype float16 --models Qwen/Qwen2.5-1.5B
```

`scripts/run_publish.py` rebuilds the six-language subspace, the control offset, the letter shuffle, the character n-gram projection, and the second sentence (`The person {name} is called`). The notebook reads `results/`, which is produced by that script and is not part of the repository.
