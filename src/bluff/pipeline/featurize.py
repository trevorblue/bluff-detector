"""Stage 4 - one row per question: labels + summary features, and the aligned hidden
states. Feature extraction is bluff.signals, the same code the API runs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bluff.config import paths
from bluff.signals import summary_features
from bluff.utils import ensure_dir, read_jsonl

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    p = paths(cfg)
    labels = {r["qid"]: r for r in read_jsonl(p.interim / "labels.jsonl")}
    gens = [g for g in read_jsonl(p.interim / "generations.jsonl") if g["condition"] == "open"]
    store = np.load(p.interim / "hidden.npz")
    hidden_all = store["hidden"]

    rows, hidden_rows = [], []
    for g in gens:
        lab = labels[g["qid"]]
        feats = summary_features(g["token_probs"], g["token_entropies"], g["retrieval_scores"])
        rows.append(
            {
                "qid": g["qid"],
                "pool": lab["pool"],
                "split": lab["split"],
                "y_wrong": int(not lab["correct"]),
                "abstained": int(lab["abstained"]),
                "open_abstained": int(lab["open_abstained"]),
                "latency_ms": g["latency_ms"],
                **feats,
            }
        )
        hidden_rows.append(hidden_all[g["hidden_index"]])

    df = pd.DataFrame(rows)
    ensure_dir(p.features)
    df.to_parquet(p.features / "features.parquet", index=False)
    np.save(p.features / "hidden.npy", np.stack(hidden_rows))
    log.info("featurized %d rows (%d wrong)", len(df), int(df["y_wrong"].sum()))
    return {"rows": len(df), "wrong_rate": float(df["y_wrong"].mean())}


def load_features(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    p = paths(cfg)
    return pd.read_parquet(p.features / "features.parquet"), np.load(p.features / "hidden.npy")
