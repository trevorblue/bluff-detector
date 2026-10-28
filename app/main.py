"""Bluff Detector API.

    POST /v1/ask    question in -> the model answers -> the detector scores the answer from
                    the model's own internals -> answer, answer with citation, or abstain
    GET  /health    liveness: the process is up
    GET  /ready     readiness: generator and detector are loaded (503 until then)
    GET  /metadata  provenance of the live detector: git SHA, data hash, thresholds

Environment
    BLUFF_CONFIG        variant config (default configs/variants/dev_fake.yaml)
    BLUFF_ARTEFACT_DIR  detector artefact folder (default: the variant's serving folder)

CRISP-DM: deployment. Uses what the pipeline built: the detector in
artefacts/<variant>/serving/. Start it with `make serve`; interactive docs at /docs.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from app.schemas import AskRequest, AskResponse
from bluff import __version__
from bluff.backends import build_backend
from bluff.config import load_config, paths
from bluff.decision import verdict
from bluff.detectors import load_detector
from bluff.prompts import build_prompt
from bluff.retrieval import build_retriever
from bluff.signals import summary_features
from bluff.utils import read_json

log = logging.getLogger("bluff.api")
STATE: dict[str, Any] = {}


def _load_state() -> None:
    cfg = load_config(os.environ.get("BLUFF_CONFIG", "configs/variants/dev_fake.yaml"))
    art = Path(os.environ.get("BLUFF_ARTEFACT_DIR") or paths(cfg).serving)
    if not (art / "detector.joblib").exists():
        log.error("no detector artefact at %s - run the pipeline first; /ready stays 503", art)
        return
    STATE["cfg"] = cfg
    STATE["detector"] = load_detector(art)
    STATE["meta"] = read_json(art / "metadata.json")
    STATE["backend"] = build_backend(cfg)  # load once, never per request
    STATE["retriever"] = build_retriever(cfg)
    log.info("ready: variant %s, detector %s", cfg["variant"]["id"], STATE["meta"]["detector"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield
    STATE.clear()


app = FastAPI(title="Bluff Detector", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> dict[str, bool]:
    if "detector" not in STATE:
        response.status_code = 503
        return {"ready": False}
    return {"ready": True}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if "meta" not in STATE:
        raise HTTPException(status_code=503, detail="not ready")
    return STATE["meta"]


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # A plain `def`: generation and scoring are CPU/GPU-bound, so FastAPI runs this in a
    # thread pool and the event loop stays free for /health and /ready.
    if "detector" not in STATE:
        raise HTTPException(status_code=503, detail="not ready")
    start = time.perf_counter()
    cfg, det, meta = STATE["cfg"], STATE["detector"], STATE["meta"]

    retriever = STATE["retriever"]
    passages = retriever.search(req.question, cfg["retrieval"]["top_k"]) if retriever else []
    prompt = build_prompt(req.question, "open", 0, [p.text for p in passages])
    out = STATE["backend"].generate(prompt, question=req.question, condition="open")

    feats = summary_features(out.token_probs, out.token_entropies, [p.score for p in passages])
    p_wrong = float(det.predict_proba_wrong(pd.DataFrame([feats]), out.hidden[None, ...])[0])
    threshold = meta["t_abstain"]
    v = verdict(p_wrong, threshold, has_sources=bool(passages))

    return AskResponse(
        request_id=str(uuid.uuid4()),
        verdict=v,
        answer=None if v == "abstain" else out.answer,
        bluff_score=p_wrong,
        threshold=threshold,
        sources=[p.id for p in passages] if v != "abstain" else [],
        variant=meta["variant"],
        detector=meta["detector"],
        model_version=meta["version"],
        latency_ms=(time.perf_counter() - start) * 1000,
    )
