"""Bluff Detector API.

    GET  /          the demo page: a question box that calls /v1/ask (app/static/index.html)
    POST /v1/ask    question in -> the model answers -> the detector scores the answer from
                    the model's own internals -> answer, answer with citation, or abstain
    GET  /health    liveness: the process is up
    GET  /ready     readiness: generator and detector are loaded (503 until then)
    GET  /metadata  provenance of the live detector: git SHA, data hash, thresholds

Environment
    BLUFF_CONFIG        variant config (default configs/variants/dev_fake.yaml)
    BLUFF_ARTEFACT_DIR  detector artefact folder (default: the variant's serving folder)
    BLUFF_MODEL_DIR     optional local folder holding the pinned model's files (instead of
                        downloading it from Hugging Face)
    BLUFF_DEVICE        optional override of model.device, e.g. "cpu" on a laptop
    BLUFF_REQUEST_LOG   where each request's audit record is appended for monitoring
                        (default logs/requests.jsonl; "off" disables it)

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
from fastapi.responses import HTMLResponse

from app.schemas import AskRequest, AskResponse
from bluff import __version__
from bluff.backends import build_backend
from bluff.config import load_config, paths
from bluff.decision import verdict
from bluff.detectors import load_detector
from bluff.monitoring import RequestLog
from bluff.prompts import build_prompt
from bluff.retrieval import build_retriever
from bluff.signals import summary_features
from bluff.utils import read_json

log = logging.getLogger("bluff.api")
STATE: dict[str, Any] = {}


def _load_state() -> None:
    cfg = load_config(os.environ.get("BLUFF_CONFIG", "configs/variants/dev_fake.yaml"))
    if os.environ.get("BLUFF_MODEL_DIR"):  # a local copy of the pinned model, e.g. on a laptop
        cfg["model"]["name"], cfg["model"]["revision"] = os.environ["BLUFF_MODEL_DIR"], None
    if os.environ.get("BLUFF_DEVICE"):  # e.g. "cpu": keep the whole model in RAM, no disk offload
        cfg["model"]["device"] = os.environ["BLUFF_DEVICE"]
    art = Path(os.environ.get("BLUFF_ARTEFACT_DIR") or paths(cfg).serving)
    if not (art / "detector.joblib").exists():
        log.error("no detector artefact at %s - run the pipeline first; /ready stays 503", art)
        return
    STATE["cfg"] = cfg
    STATE["detector"] = load_detector(art)
    STATE["meta"] = read_json(art / "metadata.json")
    STATE["backend"] = build_backend(cfg)  # load once, never per request
    STATE["retriever"] = build_retriever(cfg)
    STATE["request_log"] = RequestLog(os.environ.get("BLUFF_REQUEST_LOG", "logs/requests.jsonl"))
    log.info("ready: variant %s, detector %s", cfg["variant"]["id"], STATE["meta"]["detector"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield
    STATE.clear()


app = FastAPI(title="Bluff Detector", version=__version__, lifespan=lifespan)
DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    """The live guard's demo page: a question box that calls /v1/ask."""
    return HTMLResponse(DEMO_PAGE.read_text(encoding="utf-8"))


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

    response = AskResponse(
        request_id=str(uuid.uuid4()),
        verdict=v,
        answer=None if v == "abstain" else out.answer,
        withheld_answer=out.answer if v == "abstain" and req.reveal else None,
        bluff_score=p_wrong,
        threshold=threshold,
        sources=[p.id for p in passages] if v != "abstain" else [],
        variant=meta["variant"],
        detector=meta["detector"],
        model_version=meta["version"],
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    STATE["request_log"].write({  # monitoring: `bluff monitor` reads this log
        "request_id": response.request_id, "question": req.question, "verdict": v,
        "answer": out.answer, "bluff_score": p_wrong, "threshold": threshold,
        "latency_ms": response.latency_ms, "variant": meta["variant"],
        "detector": meta["detector"], "model_version": meta["version"], "features": feats,
    })
    return response
