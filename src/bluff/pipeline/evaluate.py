"""Stage 6 - score the held-out sets and compute every reported metric.

Test sets
    trivia_test         in-distribution held-out questions
    kenya               out-of-distribution: the hand-built Kenya set
    kenya_unanswerable  questions about things that do not exist (abstention ground truth)
    western             UK twins of Kenya questions; also compared pair by pair ("twins")

Outputs reports/<variant>/metrics.json, scores.parquet (per-row scores, used by the
compare stage) and summary.md.

CRISP-DM: evaluation.
In the pipeline: train -> [evaluate] -> register
Reads:  data/processed/<variant>/ (features), artefacts/<variant>/<detector>/
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from bluff.config import paths
from bluff.detectors import load_detector
from bluff.metrics import (
    aurc,
    auroc,
    bootstrap_ci,
    ece,
    four_cells,
    hallucination_rate,
    inflation_rate,
    mcnemar,
    paired_bootstrap_diff,
    percentiles,
    pr_auc,
    reliability_curve,
    risk_at_coverage,
    silent_knowing_rate,
    verbalisation_gap,
)
from bluff.pipeline.featurize import load_features
from bluff.schema import TRAIN_POOLS
from bluff.utils import read_json, read_jsonl, write_json

log = logging.getLogger(__name__)


def test_sets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    in_dist = df["pool"].isin(TRAIN_POOLS)
    sets = {
        "trivia_test": (in_dist & (df["split"] == "test")).to_numpy(),
        "kenya": (df["pool"] == "kenya").to_numpy(),
        "kenya_unanswerable": (df["pool"] == "kenya_unanswerable").to_numpy(),
        "western": (df["pool"] == "western").to_numpy(),
    }
    return {k: v for k, v in sets.items() if v.any()}


def twin_pairs(
    cfg: dict[str, Any], df: pd.DataFrame, score: np.ndarray
) -> dict[str, Any] | None:
    """Each Kenya question against its UK twin: same shape, other country (H2, controlled).

    Paired on the same question structure, so a difference comes from the country rather
    than from how the questions were written. Differences are Kenya minus UK.
    """
    twin_of = {r["id"]: r["twin_of"] for r in read_jsonl(paths(cfg).questions)
               if r.get("twin_of")}
    row = {qid: i for i, qid in enumerate(df["qid"])}
    pairs = [(row[k], row[w]) for w, k in twin_of.items() if w in row and k in row]
    if not pairs:
        return None
    ke, uk = (np.asarray(ix) for ix in zip(*pairs, strict=True))
    y = df["y_wrong"].to_numpy()
    a = df["abstained"].to_numpy()
    n_boot, seed = cfg["evaluation"]["bootstrap"], cfg["seed"]

    def rate(x: np.ndarray) -> float:
        return float(np.mean(x))

    def paired_test(x: np.ndarray) -> dict[str, float]:
        t = mcnemar(x[uk], x[ke])
        return {"uk_only": t["before_only"], "kenya_only": t["after_only"],
                "p_value": t["p_value"]}

    return {
        "n_pairs": len(pairs),
        "wrong_rate": {"kenya": rate(y[ke]), "uk": rate(y[uk])},
        "wrong_diff": paired_bootstrap_diff(rate, (y[uk],), (y[ke],), n_boot, seed=seed),
        "mcnemar_wrong": paired_test(y),
        "abstain_rate": {"kenya": rate(a[ke]), "uk": rate(a[uk])},
        "mcnemar_abstain": paired_test(a),
        "mean_bluff_score": {"kenya": rate(score[ke]), "uk": rate(score[uk])},
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    p = paths(cfg)
    df, hidden = load_features(cfg)
    sets = test_sets(df)
    n_boot = cfg["evaluation"]["bootstrap"]
    coverages = cfg["evaluation"]["coverages"]
    served = cfg["detectors"]["serve"]

    scores = df[["qid", "pool", "split", "y_wrong", "abstained", "open_abstained"]].copy()
    detectors_out: dict[str, Any] = {}
    meta_by_det: dict[str, Any] = {}
    for name in cfg["detectors"]["train"]:
        det = load_detector(p.artefacts / name)
        meta = read_json(p.artefacts / name / "metadata.json")
        meta_by_det[name] = meta
        start = time.perf_counter()
        s = det.predict_proba_wrong(df, hidden)
        per_row_ms = (time.perf_counter() - start) * 1000 / max(len(df), 1)
        scores[f"p_{name}"] = s
        per_set = {}
        for set_name, m in sets.items():
            y = df.loc[m, "y_wrong"].to_numpy()
            per_set[set_name] = {
                "n": int(m.sum()),
                "wrong_rate": float(y.mean()),
                "auroc": auroc(y, s[m]),
                "auroc_ci": bootstrap_ci(auroc, y, s[m], n_boot=n_boot, seed=cfg["seed"]),
                "pr_auc": pr_auc(y, s[m]),
                "ece": ece(y, s[m]),
                "aurc": aurc(y, s[m]),
                **{f"risk@{c}": risk_at_coverage(y, s[m], c) for c in coverages},
                "reliability": reliability_curve(y, s[m]),
            }
        gap = None
        if "trivia_test" in per_set and "kenya" in per_set:
            gap = per_set["trivia_test"]["auroc"] - per_set["kenya"]["auroc"]
        detectors_out[name] = {
            "sets": per_set,
            "ood_auroc_drop": gap,
            "score_ms_per_row": per_row_ms,
        }

    # Knowing vs saying, measured with the served detector and its validation-fixed tau.
    tau = meta_by_det[served]["tau_flag"]
    s_served = scores[f"p_{served}"].to_numpy()
    scores["flag"] = (s_served > tau).astype(int)
    abstention: dict[str, Any] = {}
    for set_name, m in sets.items():
        y = df.loc[m, "y_wrong"].to_numpy()
        a = df.loc[m, "abstained"].to_numpy()
        oa = df.loc[m, "open_abstained"].to_numpy()
        flag = scores.loc[m, "flag"].to_numpy()
        s = s_served[m]
        vg = verbalisation_gap(y, s, a)
        abstention[set_name] = {
            "n": int(m.sum()),
            "abstain_rate": float(a.mean()),
            "hallucination_rate": hallucination_rate(y, oa),
            "cells": four_cells(flag, a, y),
            "silent_knowing_rate": bootstrap_ci(
                silent_knowing_rate, flag, a, y, n_boot=n_boot, seed=cfg["seed"]
            ),
            "inflation_rate": inflation_rate(a, y),
            "internal_detection": bootstrap_ci(auroc, y, s, n_boot=n_boot, seed=cfg["seed"]),
            "expressed_detection": bootstrap_ci(
                lambda yy, aa: auroc(yy, aa.astype(float)), y, a, n_boot=n_boot, seed=cfg["seed"]
            ),
            "verbalisation_gap": bootstrap_ci(
                lambda yy, ss, aa: auroc(yy, ss) - auroc(yy, aa.astype(float)),
                y, s, a, n_boot=n_boot, seed=cfg["seed"],
            ),
            "vg_point": vg,
        }

    report = {
        "variant": cfg["variant"]["id"],
        "served_detector": served,
        "tau_flag": tau,
        "t_abstain": meta_by_det[served]["t_abstain"],
        "detectors": detectors_out,
        "abstention": abstention,
        "twins": twin_pairs(cfg, df, s_served),
        "generation_latency_ms": percentiles(df["latency_ms"].to_numpy()),
    }
    report = _clean(report)
    p.reports.mkdir(parents=True, exist_ok=True)
    write_json(p.reports / "metrics.json", report)
    scores.to_parquet(p.reports / "scores.parquet", index=False)
    (p.reports / "summary.md").write_text(render_summary(report), encoding="utf-8")
    log.info("wrote %s", p.reports / "metrics.json")
    return report


def _clean(obj: Any) -> Any:
    """NaN -> None so metrics.json is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float | np.floating) and math.isnan(float(obj)):
        return None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _f(x: Any, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _ci(d: dict[str, Any]) -> str:
    return f"{_f(d['point'])} [{_f(d['lo'])}, {_f(d['hi'])}]"


def render_summary(r: dict[str, Any]) -> str:
    lines = [f"# Results: {r['variant']}", "", f"Served detector: `{r['served_detector']}`  ",
             f"tau_flag {_f(r['tau_flag'])} · t_abstain {_f(r['t_abstain'])}", "",
             "## Detection (AUROC, higher is better)", "",
             "| Detector | " + " | ".join(next(iter(r["detectors"].values()))["sets"]) +
             " | OOD drop |", "|---" * (len(next(iter(r["detectors"].values()))["sets"]) + 2) +
             "|"]
    for name, d in r["detectors"].items():
        cells = [_f(s["auroc"]) for s in d["sets"].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {_f(d['ood_auroc_drop'])} |")
    lines += ["", "## Knowing vs saying (served detector, 95% bootstrap CI)", "",
              "| Set | n | Abstain rate | Silent knowing | Internal | Expressed | Gap |",
              "|---|---|---|---|---|---|---|"]
    for set_name, a in r["abstention"].items():
        lines.append(
            f"| {set_name} | {a['n']} | {_f(a['abstain_rate'])} | {_ci(a['silent_knowing_rate'])}"
            f" | {_ci(a['internal_detection'])} | {_ci(a['expressed_detection'])}"
            f" | {_ci(a['verbalisation_gap'])} |"
        )
    t = r.get("twins")
    if t:
        lines += ["", f"## Kenya vs UK twins (paired, {t['n_pairs']} pairs)", "",
                  "| | Kenya | UK twin |", "|---|---|---|"]
        for label, key in (("Wrong rate", "wrong_rate"), ("Abstain rate", "abstain_rate"),
                           ("Mean bluff score", "mean_bluff_score")):
            lines.append(f"| {label} | {_f(t[key]['kenya'])} | {_f(t[key]['uk'])} |")
        lines += ["", f"Wrong-rate difference, Kenya minus UK: {_ci(t['wrong_diff'])} · "
                  f"McNemar p {_f(t['mcnemar_wrong']['p_value'])}"]
    lat = r["generation_latency_ms"]
    lines += ["", f"Generation latency ms: p50 {_f(lat['p50'], 1)}, p95 {_f(lat['p95'], 1)}, "
              f"p99 {_f(lat['p99'], 1)}", ""]
    return "\n".join(lines)
