# Data

CRISP-DM: data understanding and data preparation. Everything the pipeline reads lives
here; see [docs/architecture.md](../docs/architecture.md) for how it flows.

Four pools, strictly separated, plus UK twins of Kenya questions. Any overlap invalidates every number, so `bluff audit` checks it.

| Pool | File | Used for | Never used for |
|---|---|---|---|
| Evaluation QA | `kenya/questions.jsonl` + TriviaQA/TruthfulQA (downloaded by `ingest`) | generating, labelling, training and testing detectors | fine-tuning, retrieval |
| Unanswerable Kenya | same file, `"pool": "kenya_unanswerable"` | abstention ground truth | anything else |
| UK twins | same file, `"pool": "western"`, `"twin_of"` = the id of the Kenya question it mirrors | the paired Kenya-vs-UK comparison (H2) | training, anything else |
| Retrieval corpus | `corpus/retrieval_chunks.jsonl` | the M1/M3 index | fine-tuning |
| Fine-tuning corpus | `corpus/finetune_docs.jsonl` | QLoRA (M2/M3) | evaluation, retrieval |

Hold the retrieval and fine-tuning corpora apart **by document**, not by chunk.

## The Kenya question set

`kenya/questions.jsonl` holds 9 seed examples (7 answerable, 2 unanswerable). They are marked
`"verified": false` and need checking before use. Target: 300–500 answerable plus ~100 unanswerable.

One JSON object per line:

```json
{"id": "ke-civ-0001", "pool": "kenya", "question": "How many counties does Kenya have?",
 "gold": "47", "aliases": ["47", "forty-seven"], "answerable": true,
 "domain": "civics_law", "language": "en",
 "source_url": "https://...", "source_note": "Constitution of Kenya 2010, First Schedule",
 "source_date": "2026-10-16", "valid_until": null,
 "author": "Trevor", "reviewer": "<a second person>", "verified": true}
```

Rules enforced by `bluff validate` and `tests/data`:

- Unique ids and unique question text.
- Answerable items need `gold`; unanswerable items need `"gold": null`, `"answerable": false` and
  `"pool": "kenya_unanswerable"`.
- A verified item needs a source and a reviewer who is not the author.
- Kenya and twin pools are test-only; `ingest` assigns `"split": "test"` itself.
- A twin (`"pool": "western"`) must set `twin_of` to the id of a real Kenya question; no other
  item may set `twin_of`. A twin asks the same question about the UK, with the same kind of answer.

Authoring tips:

- Write unanswerable items in the style of real ones (a plausible county, Act or agency), so they
  cannot be spotted from phrasing alone. Have a reviewer try to tell them apart blind.
- Put every accepted surface form in `aliases` ("SHA", "Social Health Authority").
- Set `valid_until` on anything that changes (office holders, fees, institutions).

## Corpora (you build these)

```json
{"id": "constitution-art-6", "text": "...", "source": "Constitution of Kenya 2010", "doc_id": "constitution"}
```

`retrieval_chunks.jsonl`: 300–800 tokens per chunk, legal text split by section.
`finetune_docs.jsonl`: whole documents, from a different set of `doc_id`s than the retrieval corpus.

## Folders

| Folder | What goes in it | In git? |
|---|---|---|
| `kenya/` | `questions.jsonl`, the hand-built question set (format above). A JSONL file cannot hold comments, so it is described here. | yes |
| `corpus/` | `retrieval_chunks.jsonl` and `finetune_docs.jsonl`, the two document collections | yes |
| `raw/` | Reserved for raw downloads; `ingest` currently reads the Hugging Face cache directly | no |
| `interim/<variant>/` | What `generate` and `label` write: `generations.jsonl`, `hidden.npz`, `labels.jsonl`, `review_sample.jsonl` | no |
| `processed/` | `questions.jsonl` and `manifest.json` (shared by all variants), `leakage_audit.json`, and `<variant>/` with `features.parquet` and `hidden.npy` | no |
