# Document collections

The two Kenyan document collections live here. Both come from one pool of public documents
(the Constitution, Acts, county pages, Kenya-related Wikipedia articles), split **by
document**, so no document is in both.

| File | Used by | One line per |
|---|---|---|
| `retrieval_chunks.jsonl` | M1 and M3: the passages pasted into the prompt (`bluff/retrieval.py`) | chunk of 300–800 tokens |
| `finetune_docs.jsonl` | M2 and M3: QLoRA fine-tuning (`bluff/pipeline/finetune_qlora.py`) | whole document |

Rules: no document in both files, and no evaluation question's wording in either; `bluff audit`
checks the second. CRISP-DM: data preparation. Formats and examples: [../README.md](../README.md).

## How they are built (16 November)

`bluff build-corpus --config configs/variants/m1_rag.yaml` downloads every source listed in
[`sources.yaml`](sources.yaml) and writes both files plus `manifest.json`. Sources: the
**Constitution of Kenya 2010** (Wikisource, 25 parts: preamble, 18 chapters, 6 schedules; all to
retrieval) and **63 Kenya-related English Wikipedia articles** across the six question domains,
each assigned to one collection. Result: **441 retrieval passages** (about 300–780 tokens, mean
~407 words) from 63 documents, and **26 fine-tuning documents** (~61,000 words). Every line
records its URL and licence (Wikipedia: CC BY-SA 4.0). Two swaps were made on purpose:
"Ethnic groups in Kenya" (a redirect to "Demographics of Kenya", which would have put one
article in both collections) → "Kenya Colony"; "Religion in Kenya" (shares an 8-word phrase
with a TriviaQA question, caught by the leakage audit) → "Kikuyu people". The leakage audit
then found 0 overlaps. Downloaded pages are cached in `data/raw/corpus/` (not in git).
