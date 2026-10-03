# East African names

Probe table for a neuron-subspace audit of open-weight language models.

| File | Contents |
| --- | --- |
| `data/names.csv` | 1,200 prompts |
| `data/config.json` | Prompt template, seed, place list |

600 rows are East African names, 100 each in Kinyarwanda, Runyankore, Swahili, Luganda, Lusoga, and Gisu. The other 600 are Latin-script controls matched to those names on character length: 299 frequent European given names and 301 uncommon ones.

Every row uses the same prompt:

```text
The name {name} comes from
```

`name_char_start` and `name_char_end` mark the name inside that string. Read activations at the last tokenizer piece inside the span. Token counts depend on the model, so compute them at run time and report results by token count.

| Column | Use |
| --- | --- |
| `is_east_african` | Binary probe: these names versus controls |
| `language_code` | Six-way probe, only where `include_in_language_probe` is 1 |
| `place_labels` | Acceptable place strings. Swahili accepts Kenya and Tanzania. Controls have none |
| `split` | `train` fits the probe (606 rows). `test` evaluates it (594 rows) |

The split seed is `20261002`. Mugisha and Rukundo are both Kinyarwanda and Runyankore, so those four rows stay in the binary probe, sit in `train`, and are left out of the six-way probe.

`place_vocabulary` in `data/config.json` is Rwanda, Uganda, Kenya, Tanzania, and Canada. Canada is the control place for the completion score.

`gender` is the traditional given-name tendency (`female`, `male`, `unisex`, or `unspecified`). In Uganda the same name is often inherited as a surname. `meaning_en` is filled only where the gloss is a standard one.
