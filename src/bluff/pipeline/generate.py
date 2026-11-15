"""Stage 2 - ask every question under both conditions and record the model's internals.

open    : 1 generation per question; hidden states saved (these feed the probes)
offered : `offered_templates` generations per question; only the answer text is needed

Progress is saved every `generation.checkpoint_every` questions. With `generation.resume`,
a rerun keeps the questions already answered and asks only the new ones, so a free GPU
session that times out loses little, and questions added later cost only their own time.

CRISP-DM: data preparation. The detectors' training data does not exist until the model
runs; this stage creates it. The only stage that needs a GPU for a real model.
In the pipeline: audit -> [generate] -> label
Reads:  data/processed/questions.jsonl (+ the retrieval corpus for M1 and M3)
Writes: data/interim/<variant>/generations.jsonl, hidden.npz, run_info.json
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bluff.backends import build_backend
from bluff.config import paths
from bluff.prompts import build_prompt
from bluff.retrieval import build_retriever
from bluff.schema import Generation, QuestionItem
from bluff.utils import ensure_dir, read_json, read_jsonl, set_seed, write_json, write_jsonl

log = logging.getLogger(__name__)

# Must match between a saved run and its resumption, or the numbers are not comparable.
RUN_KEYS = ("model", "revision", "dtype", "device")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg["seed"])
    p = paths(cfg)
    gen = cfg["generation"]
    items = [QuestionItem(**r) for r in read_jsonl(p.questions)]
    limit = gen.get("limit")
    if limit:
        items = items[:limit]
    per_question = sum(1 if c == "open" else gen["offered_templates"] for c in gen["conditions"])

    records, hidden_rows = [], []
    if gen.get("resume"):
        records, hidden_rows = _resume(p.interim, cfg, {it.id for it in items}, per_question)
    done = {r["qid"] for r in records}
    todo = [it for it in items if it.id not in done]
    if done:
        log.info("resuming: %d questions already answered, %d to go", len(done), len(todo))
    if not todo:
        return {"questions": len(items), "generations": len(records), "new": 0}

    backend = build_backend(cfg)
    _check_same_run(p.interim, backend.run_info, resuming=bool(done))
    retriever = build_retriever(cfg)
    k = cfg["retrieval"]["top_k"]
    variant = cfg["variant"]["id"]
    every = gen.get("checkpoint_every", 50)

    for n, item in enumerate(todo, 1):
        passages = retriever.search(item.question, k) if retriever else []
        texts = [ps.text for ps in passages]
        # Only the fake test double ever sees the correct response.
        reference = None
        if backend.uses_reference:
            reference = item.gold if item.answerable else "There is no such thing."
        for condition in gen["conditions"]:
            n_templates = 1 if condition == "open" else gen["offered_templates"]
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
        if n % every == 0:
            _save(p.interim, cfg, records, hidden_rows, backend.run_info)
            log.info("saved: %d / %d new questions answered", n, len(todo))

    _save(p.interim, cfg, records, hidden_rows, backend.run_info)
    log.info("wrote %d generations for %d questions (%d new)", len(records), len(items), len(todo))
    return {"questions": len(items), "generations": len(records), "new": len(todo),
            "backend": backend.name, **backend.run_info}


def _save(folder: Path, cfg: dict[str, Any], records: list[dict], hidden_rows: list,
          run_info: dict[str, Any]) -> None:
    """Write via temporary files, so a session killed mid-save leaves the last good copy."""
    ensure_dir(folder)
    write_jsonl(folder / "generations.jsonl.tmp", records)
    with open(folder / "hidden.npz.tmp", "wb") as f:
        np.savez_compressed(
            f,
            hidden=np.stack(hidden_rows).astype(np.float16),
            layers=np.asarray(cfg["model"]["layers"]),
            positions=np.asarray(cfg["model"]["positions"]),
        )
    os.replace(folder / "generations.jsonl.tmp", folder / "generations.jsonl")
    os.replace(folder / "hidden.npz.tmp", folder / "hidden.npz")
    write_json(folder / "run_info.json", run_info)


def _resume(folder: Path, cfg: dict[str, Any], wanted: set[str],
            per_question: int) -> tuple[list[dict], list]:
    """Load a previous run's complete questions, re-indexing their hidden states."""
    gens_path, hidden_path = folder / "generations.jsonl", folder / "hidden.npz"
    if not (gens_path.exists() and hidden_path.exists()):
        return [], []
    store = np.load(hidden_path)
    if list(store["layers"]) != list(cfg["model"]["layers"]):
        log.warning("saved run used layers %s, config says %s: starting fresh",
                    list(store["layers"]), cfg["model"]["layers"])
        return [], []
    hidden = store["hidden"]
    by_question: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(gens_path):
        by_question[rec["qid"]].append(rec)
    records, rows = [], []
    for qid, recs in by_question.items():
        if qid not in wanted or len(recs) != per_question:
            continue  # dropped from the question set, or cut off mid-question
        for rec in recs:
            if rec.get("hidden_index") is not None:
                rows.append(hidden[rec["hidden_index"]])
                rec["hidden_index"] = len(rows) - 1
            records.append(rec)
    return records, rows


def _check_same_run(folder: Path, run_info: dict[str, Any], resuming: bool) -> None:
    info_path = folder / "run_info.json"
    if not (resuming and info_path.exists()):
        return
    old = read_json(info_path)
    diff = {k: (old.get(k), run_info.get(k)) for k in RUN_KEYS if old.get(k) != run_info.get(k)}
    if diff:
        raise ValueError(
            f"cannot resume: this machine differs from the saved run {diff}. Use the same GPU "
            "type and settings, or delete data/interim/<variant>/ to start the variant again."
        )
