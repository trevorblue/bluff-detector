"""Stage 2 - ask every question under both conditions and record the model's internals.

open    : 1 generation per question; hidden states saved (these feed the probes)
offered : `offered_templates` generations per question; only the answer text is needed

CRISP-DM: data preparation. The detectors' training data does not exist until the model
runs; this stage creates it. The only stage that needs a GPU for a real model.
In the pipeline: audit -> [generate] -> label
Reads:  data/processed/questions.jsonl (+ the retrieval corpus for M1 and M3)
Writes: data/interim/<variant>/generations.jsonl, data/interim/<variant>/hidden.npz
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bluff.backends import build_backend
from bluff.config import paths
from bluff.prompts import build_prompt
from bluff.retrieval import build_retriever
from bluff.schema import Generation, QuestionItem
from bluff.utils import ensure_dir, read_jsonl, set_seed, write_jsonl

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg["seed"])
    p = paths(cfg)
    items = [QuestionItem(**r) for r in read_jsonl(p.questions)]
    limit = cfg["generation"].get("limit")
    if limit:
        items = items[:limit]
    backend = build_backend(cfg)
    retriever = build_retriever(cfg)
    k = cfg["retrieval"]["top_k"]
    variant = cfg["variant"]["id"]

    records: list[dict[str, Any]] = []
    hidden_rows: list[np.ndarray] = []
    for n, item in enumerate(items, 1):
        passages = retriever.search(item.question, k) if retriever else []
        texts = [ps.text for ps in passages]
        # Only the fake test double ever sees the correct response.
        reference = None
        if backend.uses_reference:
            reference = item.gold if item.answerable else "There is no such thing."
        for condition in cfg["generation"]["conditions"]:
            n_templates = 1 if condition == "open" else cfg["generation"]["offered_templates"]
            for t in range(n_templates):
                prompt = build_prompt(item.question, condition, t, texts)
                out = backend.generate(
                    prompt, question=item.question, condition=condition, reference=reference
                )
                rec = Generation(
                    qid=item.id,
                    variant=variant,
                    condition=condition,
                    template=t,
                    prompt=prompt,
                    answer=out.answer,
                    token_probs=out.token_probs,
                    token_entropies=out.token_entropies,
                    latency_ms=out.latency_ms,
                    retrieved_ids=[ps.id for ps in passages],
                    retrieval_scores=[ps.score for ps in passages],
                )
                if condition == "open":
                    rec.hidden_index = len(hidden_rows)
                    hidden_rows.append(out.hidden)
                records.append(rec.model_dump())
        if n % 100 == 0:
            log.info("generated %d / %d questions", n, len(items))

    ensure_dir(p.interim)
    write_jsonl(p.interim / "generations.jsonl", records)
    np.savez_compressed(
        p.interim / "hidden.npz",
        hidden=np.stack(hidden_rows).astype(np.float16),
        layers=np.asarray(cfg["model"]["layers"]),
        positions=np.asarray(cfg["model"]["positions"]),
    )
    log.info("wrote %d generations for %d questions", len(records), len(items))
    return {"questions": len(items), "generations": len(records), "backend": backend.name}
