"""Turning a generation into detector inputs.

This module is the single implementation of feature extraction, used by both the training
pipeline (featurize) and the API. One implementation means no training-serving skew.

CRISP-DM: data preparation (feature engineering).
Used by: pipeline/featurize.py, detectors.py and app/main.py.
"""

from __future__ import annotations

import numpy as np

SUMMARY_FEATURES = [
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "max_entropy",
    "last_entropy",
    "n_tokens",
    "frac_low_prob",
    "retrieval_top_score",
]

_EPS = 1e-12


def summary_features(
    token_probs: list[float],
    token_entropies: list[float],
    retrieval_scores: list[float] | None = None,
) -> dict[str, float]:
    """Summary statistics of the output distribution: the 'pulse and temperature'."""
    p = np.clip(np.asarray(token_probs, dtype=float), _EPS, 1.0)
    h = np.asarray(token_entropies, dtype=float)
    if p.size == 0:
        return {name: 0.0 for name in SUMMARY_FEATURES}
    logp = np.log(p)
    return {
        "mean_logprob": float(logp.mean()),
        "min_logprob": float(logp.min()),
        "mean_entropy": float(h.mean()) if h.size else 0.0,
        "max_entropy": float(h.max()) if h.size else 0.0,
        "last_entropy": float(h[-1]) if h.size else 0.0,
        "n_tokens": float(p.size),
        "frac_low_prob": float((p < 0.5).mean()),
        "retrieval_top_score": float(max(retrieval_scores)) if retrieval_scores else 0.0,
    }


def layer_index(layers: list[int], layer: int) -> int:
    return list(layers).index(layer)


def position_index(positions: list[str], position: str) -> int:
    return list(positions).index(position)


def select_hidden(hidden: np.ndarray, layer_idx: int, position_idx: int) -> np.ndarray:
    """Pick one (layer, position) slice. hidden is [L, P, D] or batched [N, L, P, D]."""
    if hidden.ndim == 3:
        return hidden[layer_idx, position_idx].astype(np.float32)
    if hidden.ndim == 4:
        return hidden[:, layer_idx, position_idx].astype(np.float32)
    raise ValueError(f"expected 3 or 4 dims, got shape {hidden.shape}")
