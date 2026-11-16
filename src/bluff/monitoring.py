"""Monitoring: log every live request, and compare live traffic with the training data (drift).

Two parts:
- RequestLog: the API appends each call's audit record (verdict, P(wrong), the 8 confidence
  numbers, latency) as one line of JSON to a log file.
- `bluff monitor`: reads that log and compares it with what the detector was built on:
    data drift        the 8 confidence numbers of live answers vs the training answers
    prediction drift  the live P(wrong) scores vs the validation scores, and the live block rate
                      vs the block rate the threshold gives on validation
  Each comparison gets a PSI (Population Stability Index) and a two-sample KS test, and a
  status: OK (PSI < 0.1), WATCH (0.1 to 0.25) or DRIFT (above 0.25, or KS p < 0.01). With
  fewer than 30 live requests the report says it is too early to judge.

CRISP-DM: deployment (keeping the deployed model honest).
Run:    bluff monitor --config configs/variants/m0_base.yaml [--log logs/requests.jsonl]
Reads:  the request log, data/processed/<variant>/features.parquet,
        reports/<variant>/scores.parquet, artefacts/<variant>/serving/metadata.json
Writes: reports/<variant>/monitoring/drift_report.md and drift.json
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from bluff.config import paths
from bluff.schema import TRAIN_POOLS
from bluff.signals import SUMMARY_FEATURES
from bluff.utils import ensure_dir, read_json, read_jsonl, write_json

log = logging.getLogger(__name__)

MIN_LIVE = 30
PSI_WATCH, PSI_DRIFT, KS_ALPHA = 0.1, 0.25, 0.01


class RequestLog:
    """Appends one JSON line per request. Never breaks a request: a write error is logged once."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path and path.lower() != "off" else None
        self._lock = threading.Lock()
        self._warned = False

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record})
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as e:
            if not self._warned:
                log.warning("request log disabled: cannot write %s (%s)", self.path, e)
                self._warned = True


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over the reference's quantile bins (0 = same distribution)."""
    ref, cur = np.asarray(reference, dtype=float), np.asarray(live, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant reference: any change is drift
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / ref.size
    q = np.histogram(cur, edges)[0] / cur.size
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _check(name: str, kind: str, ref: np.ndarray, cur: np.ndarray) -> dict[str, Any]:
    value = psi(ref, cur)
    p = float(ks_2samp(ref, cur).pvalue) if len(ref) and len(cur) else float("nan")
    status = ("DRIFT" if value > PSI_DRIFT or p < KS_ALPHA
              else "WATCH" if value > PSI_WATCH else "OK")
    return {"check": name, "kind": kind, "psi": value, "ks_p": p, "status": status,
            "reference_mean": float(np.mean(ref)), "live_mean": float(np.mean(cur))}


def run(cfg: dict[str, Any], log_path: str | Path = "logs/requests.jsonl") -> dict[str, Any]:
    p = paths(cfg)
    live = pd.DataFrame(read_jsonl(log_path)) if Path(log_path).exists() else pd.DataFrame()
    meta = read_json(p.serving / "metadata.json")
    if not live.empty and "variant" in live.columns:  # only traffic served by this variant
        live = live[live["variant"] == meta["variant"]]
    feats = pd.read_parquet(p.features / "features.parquet")
    ref = feats[feats["pool"].isin(TRAIN_POOLS) & (feats["split"] == "train")]
    scores = pd.read_parquet(p.reports / "scores.parquet")
    val_scores = scores[scores["pool"].isin(TRAIN_POOLS) & (scores["split"] == "val")]
    served = f"p_{meta['detector']}"
    threshold = meta["t_abstain"]

    checks: list[dict[str, Any]] = []
    if not live.empty:
        live_feats = pd.DataFrame(list(live["features"]))
        for f in SUMMARY_FEATURES:
            if f in live_feats and ref[f].nunique() > 1:  # skip constants (e.g. no retrieval)
                checks.append(_check(f, "data", ref[f].to_numpy(), live_feats[f].to_numpy()))
        checks.append(_check("bluff_score (P(wrong))", "prediction",
                             val_scores[served].to_numpy(), live["bluff_score"].to_numpy()))

    ref_block = float((val_scores[served] > threshold).mean())
    lat = live["latency_ms"].to_numpy() if not live.empty else np.array([])
    summary = {
        "variant": meta["variant"], "detector": meta["detector"],
        "model_version": meta["version"],
        "generated": datetime.now().isoformat(timespec="seconds"),
        "live_requests": int(len(live)), "enough_to_judge": bool(len(live) >= MIN_LIVE),
        "first_request": None if live.empty else str(live["ts"].min()),
        "last_request": None if live.empty else str(live["ts"].max()),
        "block_rate": {"validation": ref_block,
                       "live": None if live.empty else float((live["verdict"] == "abstain").mean())},
        "latency_ms": {} if lat.size == 0 else {
            k: float(np.percentile(lat, q)) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
        "checks": checks,
        "overall": ("TOO EARLY" if len(live) < MIN_LIVE
                    else "DRIFT" if any(c["status"] == "DRIFT" for c in checks)
                    else "WATCH" if any(c["status"] == "WATCH" for c in checks) else "OK"),
    }
    out = ensure_dir(p.reports / "monitoring")
    write_json(out / "drift.json", summary)
    _write_report(out / "drift_report.md", summary, threshold, log_path)
    log.info("monitoring: %d live requests, overall %s", len(live), summary["overall"])
    return {k: summary[k] for k in ("live_requests", "overall", "block_rate", "latency_ms")} | {
        "report": str(out / "drift_report.md")}


def _write_report(path: Path, s: dict[str, Any], threshold: float, log_path: str | Path) -> None:
    def num(x: float | None, d: int = 3) -> str:
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

    lines = [
        f"# Monitoring report: {s['variant']}", "",
        f"Generated {s['generated']} from `{log_path}`. Serving `{s['detector']}`, version "
        f"`{s['model_version']}`.", "",
        f"**Overall: {s['overall']}** · {s['live_requests']} live requests"
        + (f" ({s['first_request']} to {s['last_request']})" if s["live_requests"] else ""), "",
    ]
    if not s["enough_to_judge"]:
        lines += [f"> Fewer than {MIN_LIVE} requests: the checks below are shown, but it is too "
                  "early to call drift.", ""]
    br, lat = s["block_rate"], s["latency_ms"]
    lines += [
        "## Behaviour", "",
        "| | Validation baseline | Live |", "|---|---|---|",
        f"| Block rate (P(wrong) > {threshold}) | {num(br['validation'])} | {num(br['live'])} |",
        f"| Latency p50 / p95 / p99 (ms) | - | {num(lat.get('p50'), 0)} / "
        f"{num(lat.get('p95'), 0)} / {num(lat.get('p99'), 0)} |", "",
        "## Drift checks", "",
        "PSI: under 0.1 = stable, 0.1–0.25 = watch, above 0.25 = drift. KS p under 0.01 = the "
        "two samples differ. *Data* checks compare the 8 confidence numbers of live answers with "
        "the training answers; the *prediction* check compares live P(wrong) with validation.", "",
        "| Check | Kind | Reference mean | Live mean | PSI | KS p | Status |",
        "|---|---|---|---|---|---|---|",
        *[f"| {c['check']} | {c['kind']} | {num(c['reference_mean'])} | {num(c['live_mean'])} "
          f"| {num(c['psi'])} | {num(c['ks_p'], 4)} | {c['status']} |" for c in s["checks"]],
        "",
        "## What to do", "",
        "- **Data drift** (the questions changed): check what users are asking; if it is a new "
        "kind of question, add similar questions with known answers and retrain.",
        "- **Prediction drift** (P(wrong) or the block rate moved) without data drift: the model "
        "or its hardware changed (e.g. a laptop CPU in bfloat16 instead of the T4 in float16). "
        "Re-measure the detector on that setup before trusting its threshold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
