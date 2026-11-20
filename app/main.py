"""Bluff Detector API.

    GET  /          the demo page: a question box that calls /v1/ask (app/static/index.html)
    POST /v1/ask    question in -> the model answers -> the detector scores the answer from
                    the model's own internals -> answer, answer with citation, or abstain
    GET  /health    liveness: the process is up
    GET  /ready     readiness: generator and detector are loaded (503 until then)
    GET  /metadata  provenance of the live detector: git SHA, data hash, thresholds
    GET  /v1/variants    the model versions this server answers with (M0, M1, M2...)
    GET  /v1/model-card  a variant's detectors: test metrics and typical confidence numbers
    GET  /v1/results     the project's headline results, hypothesis by hypothesis, as saved by
                         notebooks/results.ipynb (reports/results_summary.json)

Environment
    BLUFF_CONFIG        the default variant's config (default configs/variants/dev_fake.yaml)
    BLUFF_VARIANTS      more variants to serve with the same model, e.g. "m1_rag,m2_qlora";
                        a request picks one with "variant" (and a detector with "detector")
    BLUFF_ARTEFACT_DIR  detector artefact folder (default: the variant's serving folder)
    BLUFF_MODEL_DIR     optional local folder holding the pinned model's files (instead of
                        downloading it from Hugging Face)
    BLUFF_DEVICE        optional override of model.device, e.g. "cpu" on a laptop
    BLUFF_REQUEST_LOG   where each request's audit record is appended for monitoring
                        (default logs/requests.jsonl; "off" disables it)
    BLUFF_RESULTS       the headline results file (default reports/results_summary.json)

CRISP-DM: deployment. Uses what the pipeline built: the detector in
artefacts/<variant>/serving/. Start it with `make serve`; interactive docs at /docs.
"""

from __future__ import annotations

import dataclasses
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

from app.schemas import AskRequest, AskResponse, Passage, Token
from bluff import __version__
from bluff.backends import build_backend
from bluff.config import load_config, paths
from bluff.decision import verdict
from bluff.detectors import load_detector
from bluff.monitoring import RequestLog
from bluff.prompts import build_prompt
from bluff.retrieval import build_retriever
from bluff.schema import TRAIN_POOLS
from bluff.signals import SUMMARY_FEATURES, summary_features
from bluff.utils import read_json

log = logging.getLogger("bluff.api")
STATE: dict[str, Any] = {}
DETECTOR_NAMES = ["logreg", "xgboost", "probe_linear", "probe_mlp"]


def model_card(cfg: dict[str, Any], meta: dict[str, Any], art: Path) -> dict[str, Any]:
    """What the demo page shows under "About this detector": provenance, thresholds, the test
    metrics from the last `bluff evaluate`, and typical confidence numbers of right and wrong
    training answers. The parts whose files are missing (e.g. inside Docker) are left out."""
    p = paths(cfg)
    # Use the results of the same run as the served detector: <root>/artefacts/<variant>/serving
    # sits next to <root>/reports/<variant> and <root>/data/processed/<variant>.
    root, variant = art.parent.parent.parent, meta.get("variant", cfg["variant"]["id"])
    if (root / "reports" / variant / "metrics.json").exists():
        p = dataclasses.replace(p, reports=root / "reports" / variant,
                       features=root / "data" / "processed" / variant)
    card: dict[str, Any] = {k: meta.get(k) for k in (
        "variant", "detector", "version", "base_model", "trained_at", "n_train", "n_val",
        "t_abstain", "tau_flag", "cost_matrix", "probe_layer", "probe_position", "git_sha")}
    card["validation"] = meta.get("val", {})
    card["description"] = cfg["variant"].get("description")
    card["model"] = cfg["model"].get("name")
    card["adapter"] = cfg["model"].get("adapter")
    card["retrieval"] = bool(cfg.get("retrieval", {}).get("enabled"))
    card["device"] = cfg["model"].get("device")
    card["backend"] = cfg["model"].get("backend")
    if (p.reports / "metrics.json").exists():
        m = read_json(p.reports / "metrics.json")
        card["test"] = {
            det: {s: {"n": v.get("n"), "wrong_rate": v.get("wrong_rate"), "auroc": v.get("auroc"),
                      "auroc_lo": (v.get("auroc_ci") or {}).get("lo"),
                      "auroc_hi": (v.get("auroc_ci") or {}).get("hi"),
                      "pr_auc": v.get("pr_auc"), "ece": v.get("ece"),
                      "risk_at_80": v.get("risk@0.8")} for s, v in dm["sets"].items()}
            for det, dm in m["detectors"].items()}
        card["behaviour"] = {
            s: {"n": a.get("n"), "abstain_rate": a.get("abstain_rate"),
                "hallucination_rate": a.get("hallucination_rate"),
                "inflation_rate": a.get("inflation_rate"),
                **{k: a.get(k) for k in ("silent_knowing_rate", "internal_detection",
                                         "expressed_detection", "verbalisation_gap")}}
            for s, a in m["abstention"].items()}
    if (p.features / "features.parquet").exists():
        f = pd.read_parquet(p.features / "features.parquet")
        f = f[f["pool"].isin(TRAIN_POOLS) & (f["split"] == "train")]
        card["signal_reference"] = {
            name: {"right": float(f.loc[f.y_wrong == 0, name].median()),
                   "wrong": float(f.loc[f.y_wrong == 1, name].median())}
            for name in SUMMARY_FEATURES}
    return card


def _load_variant(config_path: str, artefact_dir: str | None = None) -> dict[str, Any] | None:
    """One model version (M0, M1, M2...): its config, its four detectors and its retriever."""
    cfg = load_config(config_path)
    if os.environ.get("BLUFF_MODEL_DIR"):  # a local copy of the pinned model, e.g. on a laptop
        cfg["model"]["name"], cfg["model"]["revision"] = os.environ["BLUFF_MODEL_DIR"], None
    if os.environ.get("BLUFF_DEVICE"):  # e.g. "cpu": keep the whole model in RAM, no disk offload
        cfg["model"]["device"] = os.environ["BLUFF_DEVICE"]
    art = Path(artefact_dir or paths(cfg).serving)
    if not (art / "detector.joblib").exists():
        log.error("no detector artefact at %s - run the pipeline first", art)
        return None
    meta = read_json(art / "metadata.json")
    # Every trained detector of this variant, so each answer can be scored "inside vs outside"
    # and any of them can be chosen to decide. They sit next to the served one
    # (artefacts/<variant>/<name>/), from the same training run.
    names = [n for n in DETECTOR_NAMES if (art.parent / n / "detector.joblib").exists()]
    return {
        "cfg": cfg, "meta": meta, "detector": load_detector(art),
        "all_detectors": {n: load_detector(art.parent / n) for n in names},
        "all_meta": {n: read_json(art.parent / n / "metadata.json") for n in names},
        "card": model_card(cfg, meta, art),
        "retriever": build_retriever(cfg),
    }


def _same_generator(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("backend", "name", "revision", "layers", "positions")
    return all(a["model"].get(k) == b["model"].get(k) for k in keys)


def _load_state() -> None:
    """Load the default variant (BLUFF_CONFIG) and any others named in BLUFF_VARIANTS.

    Every variant shares ONE copy of the language model: M0 and M1 are the base model (M1 adds
    retrieval), M2 is the base model with its adapter switched on. So serving M0, M1 and M2
    together costs one model in memory, not three."""
    default = _load_variant(os.environ.get("BLUFF_CONFIG", "configs/variants/dev_fake.yaml"),
                            os.environ.get("BLUFF_ARTEFACT_DIR"))
    if default is None:
        log.error("default variant not loaded; /ready stays 503")
        return
    variants = {default["cfg"]["variant"]["id"]: default}
    for vid in [v.strip() for v in os.environ.get("BLUFF_VARIANTS", "").split(",") if v.strip()]:
        if vid in variants:
            continue
        v = _load_variant(f"configs/variants/{vid}.yaml")
        if v is None:
            continue
        if not _same_generator(v["cfg"], default["cfg"]):
            log.error("variant %s uses a different base model or layers; not loaded", vid)
            continue
        variants[vid] = v
    # The generator is built once, from the variant that needs the adapter if there is one.
    with_adapter = [v for v in variants.values() if v["cfg"]["model"].get("adapter")]
    if len({v["cfg"]["model"]["adapter"] for v in with_adapter}) > 1:
        log.error("more than one adapter requested; only the first is loaded")
    gen_cfg = (with_adapter[0] if with_adapter else default)["cfg"]
    STATE["variants"] = variants
    STATE["default"] = default["cfg"]["variant"]["id"]
    # The default variant's pieces, under their old names (used by /metadata and the tests).
    STATE.update(cfg=default["cfg"], detector=default["detector"], meta=default["meta"],
                 all_detectors=default["all_detectors"], card=default["card"],
                 retriever=default["retriever"])
    STATE["backend"] = build_backend(gen_cfg)  # load once, never per request
    STATE["request_log"] = RequestLog(os.environ.get("BLUFF_REQUEST_LOG", "logs/requests.jsonl"))
    log.info("ready: variants %s (default %s)", ", ".join(variants), STATE["default"])


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


def _variant(vid: str | None) -> tuple[str, dict[str, Any]]:
    if "variants" not in STATE:
        raise HTTPException(status_code=503, detail="not ready")
    vid = vid or STATE["default"]
    if vid not in STATE["variants"]:
        raise HTTPException(status_code=400, detail=f"variant {vid!r} is not loaded; loaded: "
                            f"{', '.join(STATE['variants'])}")
    return vid, STATE["variants"][vid]


@app.get("/v1/variants")
def get_variants() -> list[dict[str, Any]]:
    """The model versions this server can answer with, and the detectors each one has."""
    _variant(None)
    return [{"id": vid, "default": vid == STATE["default"],
             "description": v["cfg"]["variant"].get("description"),
             "retrieval": v["retriever"] is not None,
             "adapter": v["cfg"]["model"].get("adapter"),
             "served_detector": v["meta"]["detector"], "detectors": list(v["all_detectors"])}
            for vid, v in STATE["variants"].items()]


@app.get("/v1/model-card")
def get_model_card(variant: str | None = None) -> dict[str, Any]:
    """Provenance, thresholds and test metrics of a variant's detectors (for the demo page)."""
    return _variant(variant)[1]["card"]


@app.get("/v1/results")
def get_results() -> dict[str, Any]:
    """The headline results of every hypothesis (for the demo page's "What we found" tab).

    Written by the last cell of notebooks/results.ipynb, so the page and the notebook show the
    same numbers. Independent of which variant this server is running."""
    path = Path(os.environ.get("BLUFF_RESULTS", "reports/results_summary.json"))
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"{path} not found - run notebooks/results.ipynb")
    return read_json(path)


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # A plain `def`: generation and scoring are CPU/GPU-bound, so FastAPI runs this in a
    # thread pool and the event loop stays free for /health and /ready.
    vid, V = _variant(req.variant)
    start = time.perf_counter()
    cfg, meta = V["cfg"], V["meta"]
    name = req.detector or meta["detector"]  # the served (champion) detector unless chosen
    if name not in V["all_detectors"]:
        raise HTTPException(status_code=400, detail=f"detector {name!r} not available for {vid}")
    det = V["detector"] if name == meta["detector"] else V["all_detectors"][name]
    det_meta = meta if name == meta["detector"] else V["all_meta"][name]

    retriever = V["retriever"]
    passages = retriever.search(req.question, cfg["retrieval"]["top_k"]) if retriever else []
    prompt = build_prompt(req.question, "open", 0, [p.text for p in passages])
    out = STATE["backend"].generate(prompt, question=req.question, condition="open",
                                    use_adapter=bool(cfg["model"].get("adapter")))

    feats = summary_features(out.token_probs, out.token_entropies, [p.score for p in passages])
    row, hidden = pd.DataFrame([feats]), out.hidden[None, ...]
    p_wrong = float(det.predict_proba_wrong(row, hidden)[0])
    threshold = det_meta.get("t_abstain", meta["t_abstain"])
    v = verdict(p_wrong, threshold, has_sources=bool(passages))
    shown = v != "abstain" or req.reveal  # word-level detail would reveal a blocked answer
    words = out.meta.get("tokens") or [""] * len(out.token_probs)

    response = AskResponse(
        request_id=str(uuid.uuid4()),
        verdict=v,
        answer=None if v == "abstain" else out.answer,
        withheld_answer=out.answer if v == "abstain" and req.reveal else None,
        bluff_score=p_wrong,
        threshold=threshold,
        sources=[p.id for p in passages] if v != "abstain" else [],
        passages=[Passage(id=p.id, text=p.text[:600], score=p.score) for p in passages]
        if shown else [],
        variant=vid,
        detector=name,
        model_version=det_meta.get("version", meta["version"]),
        latency_ms=(time.perf_counter() - start) * 1000,
        all_detectors={n: float(d.predict_proba_wrong(row, hidden)[0])
                       for n, d in V["all_detectors"].items()},
        signals=feats,
        tokens=[Token(text=w, prob=pr, entropy=e) for w, pr, e in
                zip(words, out.token_probs, out.token_entropies, strict=False)] if shown else [],
    )
    STATE["request_log"].write({  # monitoring: `bluff monitor` reads this log
        "request_id": response.request_id, "question": req.question, "verdict": v,
        "answer": out.answer, "bluff_score": p_wrong, "threshold": threshold,
        "latency_ms": response.latency_ms, "variant": vid,
        "detector": name, "model_version": response.model_version, "features": feats,
    })
    return response
