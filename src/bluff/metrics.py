"""Evaluation metrics.

Conventions used everywhere in this project:
    y_wrong : 1 if the open-condition answer was wrong, else 0   (the positive class)
    score   : a detector's P(wrong), higher = more likely a bluff
    flag    : score > tau, i.e. "the probe says uncertain"
    a       : 1 if the model took the offered "I don't know" option, else 0

CRISP-DM: evaluation.
Used by: pipeline/train.py (validation scores), pipeline/evaluate.py and
pipeline/compare.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------- detection quality


def _two_classes(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def auroc(y_wrong: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_wrong).astype(int)
    return float(roc_auc_score(y, score)) if _two_classes(y) else float("nan")


def pr_auc(y_wrong: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_wrong).astype(int)
    return float(average_precision_score(y, score)) if _two_classes(y) else float("nan")


def ece(y_wrong: np.ndarray, p_wrong: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of P(wrong), equal-width bins."""
    y = np.asarray(y_wrong, dtype=float)
    p = np.clip(np.asarray(p_wrong, dtype=float), 0.0, 1.0)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def reliability_curve(y_wrong: np.ndarray, p_wrong: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y_wrong, dtype=float)
    p = np.clip(np.asarray(p_wrong, dtype=float), 0.0, 1.0)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out.append(
                {"bin": b, "n": int(mask.sum()), "mean_p": p[mask].mean(), "freq": y[mask].mean()}
            )
    return out


# ---------------------------------------------------------------- selective prediction


def risk_coverage(y_wrong: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most-confident questions first. Returns (coverage, risk) arrays."""
    y = np.asarray(y_wrong, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float), kind="stable")
    cum_wrong = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    return k / len(y), cum_wrong / k


def aurc(y_wrong: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    if len(y_wrong) == 0:
        return float("nan")
    _, risk = risk_coverage(y_wrong, score)
    return float(risk.mean())


def risk_at_coverage(y_wrong: np.ndarray, score: np.ndarray, coverage: float) -> float:
    """Error rate among the answers given, when only `coverage` of questions are answered."""
    n = len(y_wrong)
    if n == 0:
        return float("nan")
    _, risk = risk_coverage(y_wrong, score)
    k = max(1, math.ceil(coverage * n))
    return float(risk[k - 1])


# ---------------------------------------------------------------- knowing vs saying


def four_cells(flag: np.ndarray, a: np.ndarray, y_wrong: np.ndarray) -> dict[str, dict[str, int]]:
    """Probe verdict x model behaviour, each split by whether the open answer was wrong."""
    flag, a, y = (np.asarray(x).astype(bool) for x in (flag, a, y_wrong))
    cells = {
        "honest_uncertainty": flag & a,
        "silent_knowing": flag & ~a,
        "abstention_inflation": ~flag & a,
        "earned_confidence": ~flag & ~a,
    }
    return {
        name: {"n": int(m.sum()), "wrong": int((m & y).sum()), "correct": int((m & ~y).sum())}
        for name, m in cells.items()
    }


def silent_knowing_rate(flag: np.ndarray, a: np.ndarray, y_wrong: np.ndarray) -> float:
    """Among confidently delivered wrong answers (a=0, wrong), the share the probe flagged."""
    flag, a, y = (np.asarray(x).astype(bool) for x in (flag, a, y_wrong))
    denom = ~a & y
    return float(flag[denom].mean()) if denom.any() else float("nan")


def inflation_rate(a: np.ndarray, y_wrong: np.ndarray) -> float:
    """Among abstentions, the share the model answered correctly in the open condition."""
    a, y = np.asarray(a).astype(bool), np.asarray(y_wrong).astype(bool)
    return float((~y[a]).mean()) if a.any() else float("nan")


def verbalisation_gap(y_wrong: np.ndarray, score: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """Internal detection (probe AUROC) minus expressed detection (abstention AUROC).

    AUROC of a binary predictor equals its balanced accuracy, so expressed detection is
    0.5 when abstention carries no information about errors.
    Positive gap = the model knows more than it says.
    """
    internal = auroc(y_wrong, score)
    expressed = auroc(y_wrong, np.asarray(a, dtype=float))
    return {"internal": internal, "expressed": expressed, "gap": internal - expressed}


def hallucination_rate(y_wrong: np.ndarray, open_abstained: np.ndarray) -> float:
    """Share of all answers that were wrong and stated rather than refused."""
    y, oa = np.asarray(y_wrong).astype(bool), np.asarray(open_abstained).astype(bool)
    return float((y & ~oa).mean()) if len(y) else float("nan")


# ---------------------------------------------------------------- statistics


def bootstrap_ci(
    fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap over rows (questions). NaN resamples are dropped."""
    arrays = tuple(np.asarray(x) for x in arrays)
    n = len(arrays[0])
    point = fn(*arrays)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*(x[idx] for x in arrays))
        if not math.isnan(v):
            stats.append(v)
    if not stats:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": len(stats)}


def paired_bootstrap_diff(
    fn: Callable[..., float],
    arrays_a: tuple[np.ndarray, ...],
    arrays_b: tuple[np.ndarray, ...],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """CI for fn(B) - fn(A) when A and B were scored on the same questions, row-aligned."""
    arrays_a = tuple(np.asarray(x) for x in arrays_a)
    arrays_b = tuple(np.asarray(x) for x in arrays_b)
    n = len(arrays_a[0])
    point = fn(*arrays_b) - fn(*arrays_a)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = fn(*(x[idx] for x in arrays_b)) - fn(*(x[idx] for x in arrays_a))
        if not math.isnan(d):
            diffs.append(d)
    if not diffs:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "p_le_0": None}
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "p_le_0": float(np.mean(np.asarray(diffs) <= 0)),
    }


def mcnemar(a_before: np.ndarray, a_after: np.ndarray) -> dict[str, float]:
    """Exact McNemar test for a change in a paired binary outcome (e.g. abstention)."""
    x, y = np.asarray(a_before).astype(bool), np.asarray(a_after).astype(bool)
    b = int((x & ~y).sum())  # abstained before, not after
    c = int((~x & y).sum())  # abstained after, not before
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"before_only": b, "after_only": c, "p_value": p}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    p50, p95, p99 = np.percentile(v, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}
