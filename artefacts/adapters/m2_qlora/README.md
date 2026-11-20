---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- qlora
---

# M2 adapter: Qwen2.5-1.5B-Instruct + LoRA trained on Kenyan documents

This is the small add-on ("adapter") that turns the base model (M0) into **M2** in the Bluff
Detector project. It is not a standalone model: it is loaded on top of
`Qwen/Qwen2.5-1.5B-Instruct` at the pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

**What it is for.** Testing hypothesis H3: does fine-tuning change what the model *says*
("I don't know") more than what it *knows* inside (its hidden states)? Result: yes. Fine-tuning
roughly halved "I don't know" (Kenya 28% → 13%, trivia 35% → 20%) without making answers more
correct (Kenya wrong 73% → 75%, not significant). See `notebooks/results.ipynb`, section H3.

## How it was trained

| Setting | Value |
|---|---|
| Method | QLoRA: base model frozen and loaded in 4-bit (NF4) during training only; LoRA weights trained in 16-bit |
| LoRA | rank r = 16, alpha = 32, dropout 0.05, on all 7 projection layers (q, k, v, o, gate, up, down) |
| Trainable weights | 18.5 million (about 1.2% of the 1.5 billion) |
| Data | `data/corpus/finetune_docs.jsonl`: 26 Kenyan documents (Acts, Wikipedia articles), checked to share no document with the evaluation questions or the retrieval index (`bluff audit`) |
| Windows | 106 windows of up to 1,024 tokens (whole documents cut into pieces, not truncated) |
| Training | plain next-word prediction, 3 epochs, batch 2 × gradient accumulation 4, learning rate 2e-4, cosine schedule, 42 steps, ~8 minutes on one Kaggle T4 |
| Loss | 2.47 (first step) → 2.26 (last step); average 2.33 (`training_log.json`) |
| Code | `src/bluff/pipeline/finetune_qlora.py` (`bluff finetune_qlora --config configs/variants/m2_qlora.yaml`) |

## Files

| File | What it is |
|---|---|
| `adapter_model.safetensors` | the trained LoRA weights (74 MB) |
| `adapter_config.json` | the LoRA settings peft needs to load them |
| `training_log.json` | loss per step and the settings above |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | copies of the base model's tokenizer, saved with the adapter (unchanged) |

## How to use it

The pipeline loads it automatically for M2 and M3 (`model.adapter` in
`configs/variants/m2_qlora.yaml`). By hand:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
model = PeftModel.from_pretrained(base, "artefacts/adapters/m2_qlora")
```

## Limits

Trained on only 26 documents, so it learned the *style* of confident Kenyan text more than new
facts. It is a research artefact for measuring that effect, not a better Kenya question-answerer.
