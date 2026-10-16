"""QLoRA fine-tuning for variant M2 - a template, not yet run.

Needs `pip install .[llm,qlora]` and a CUDA GPU. Not exercised in CI. Read every TODO
before the first run, and run a tiny pilot (a few hundred steps) before the full job.

Data rule: train ONLY on data/corpus/finetune_docs.jsonl, which must be disjoint by
document from the evaluation questions and the retrieval index. Run `bluff audit` first.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bluff.utils import read_jsonl

log = logging.getLogger(__name__)


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
    name = cfg["model"]["name"]

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
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

    ds = Dataset.from_list([{"text": d["text"]} for d in docs])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=1024), batched=True,
                remove_columns=["text"])
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,  # TODO: pick from the pilot's loss curve
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        seed=cfg["seed"],
        report_to=["mlflow"] if cfg["tracking"]["mlflow"] else [],
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    log.info("adapter saved to %s - point model.adapter at it in the M2 config", output_dir)
