"""QLoRA fine-tuning for variant M2: a small adapter trained on the Kenyan fine-tuning documents.

Needs `pip install .[llm,qlora]` and a CUDA GPU (the 4-bit base needs bitsandbytes); not run in
CI. Settings come from `finetune:` in the config (epochs, batch_size, grad_accum,
learning_rate). Writes the adapter and `training_log.json` (loss per step) for the report.

Data rule: train ONLY on data/corpus/finetune_docs.jsonl, which must be disjoint by
document from the evaluation questions and the retrieval index. Run `bluff audit` first.

CRISP-DM: modeling (builds the M2 generator variant).
Not part of `bluff all`: run it once, before M2, with
`bluff finetune_qlora --config configs/variants/m2_qlora.yaml`.
Reads:  data/corpus/finetune_docs.jsonl
Writes: artefacts/adapters/m2_qlora/ (the adapter M2 and M3 load)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from bluff.backends import resolve_dtype
from bluff.utils import read_jsonl, write_json

log = logging.getLogger(__name__)
WINDOW = 1024  # tokens per training window


def run(cfg: dict[str, Any], output_dir: str = "artefacts/adapters/m2_qlora") -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    corpus = Path(cfg["data"]["finetune_corpus"])
    if not corpus.exists():
        raise FileNotFoundError(f"{corpus} missing - build the fine-tuning corpus first")
    docs = read_jsonl(corpus)
    name, revision = cfg["model"]["name"], cfg["model"].get("revision")
    compute_dtype = resolve_dtype(torch, cfg["model"].get("dtype", "auto"))

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, quantization_config=bnb, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,  # TODO: sweep r in {8, 16, 32} on the pilot
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                        "down_proj"],  # TODO: check module names for your base model
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Whole documents cut into 1,024-token windows: truncating would keep only the start of
    # each Act and throw the rest away.
    windows = [ids[i : i + WINDOW] for d in docs
               for ids in [tok(d["text"])["input_ids"]] for i in range(0, len(ids), WINDOW)]
    ds = Dataset.from_dict({"input_ids": windows})
    log.info("%d documents -> %d training windows of up to %d tokens", len(docs), len(ds), WINDOW)
    ft = cfg.get("finetune", {})
    # The corpus is small (~80k tokens, ~80 windows): several passes at 8 windows a step give
    # ~30 optimizer steps; a single pass at 16 a step gave only ~5, too few to change the model.
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=ft.get("batch_size", 2),
        gradient_accumulation_steps=ft.get("grad_accum", 4),
        num_train_epochs=ft.get("epochs", 3),
        learning_rate=ft.get("learning_rate", 2e-4),
        lr_scheduler_type="cosine",
        # 3% of the steps as warm-up (transformers 5 removed `warmup_ratio`; steps work in all)
        warmup_steps=max(1, round(0.03 * ft.get("epochs", 3) * math.ceil(
            len(ds) / (ft.get("batch_size", 2) * ft.get("grad_accum", 4))))),
        logging_steps=1,
        save_strategy="no",  # only the final adapter is kept (below), not optimizer checkpoints
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=cfg["seed"],
        report_to=["mlflow"] if cfg["tracking"]["mlflow"] else [],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    result = trainer.train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    write_json(Path(output_dir) / "training_log.json", {
        "documents": len(docs), "windows": len(ds), "window_tokens": WINDOW,
        "epochs": args.num_train_epochs, "steps": result.global_step,
        "train_loss": result.training_loss,
        "loss_per_step": [h["loss"] for h in trainer.state.log_history if "loss" in h],
        "lora": {"r": lora.r, "alpha": lora.lora_alpha,
                 "target_modules": sorted(lora.target_modules)},
    })
    log.info("adapter saved to %s after %d steps (training loss %.3f)",
             output_dir, result.global_step, result.training_loss)
