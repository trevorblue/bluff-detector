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
