"""Compare two variants on identical questions - the hypothesis-3 test.

H3 (fine-tuning breaks saying, not knowing) predicts, from base (A) to fine-tuned (B):
    expressed detection falls, internal detection holds, so the verbalisation gap grows.
Paired bootstrap on the same question rows; exact McNemar for the change in abstention.

CRISP-DM: evaluation.
Not part of `bluff all`: run `bluff compare --config A.yaml --other B.yaml` once both
variants have been evaluated.
Reads:  reports/<A>/scores.parquet, reports/<B>/scores.parquet
Writes: reports/compare/<A>__vs__<B>.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bluff.config import paths
from bluff.metrics import auroc, holm, mcnemar, paired_bootstrap_diff
from bluff.utils import write_json

log = logging.getLogger(__name__)


def _internal(y, s, a):
    return auroc(y, s)


def _expressed(y, s, a):
    return auroc(y, a.astype(float))


def _gap(y, s, a):
    return auroc(y, s) - auroc(y, a.astype(float))


def run(cfg_a: dict[str, Any], cfg_b: dict[str, Any]) -> dict[str, Any]:
    pa, pb = paths(cfg_a), paths(cfg_b)
    sa = pd.read_parquet(pa.reports / "scores.parquet")
    sb = pd.read_parquet(pb.reports / "scores.parquet")
    det_a, det_b = cfg_a["detectors"]["serve"], cfg_b["detectors"]["serve"]
    m = sa.merge(sb, on=["qid", "pool", "split"], suffixes=("_a", "_b"))
    n_boot, seed = cfg_a["evaluation"]["bootstrap"], cfg_a["seed"]

    out: dict[str, Any] = {
        "a": cfg_a["variant"]["id"],
        "b": cfg_b["variant"]["id"],
        "sets": {},
    }
    for set_name, mask in {
        "trivia_test": (m["split"] == "test") & ~m["pool"].str.startswith("kenya"),
        "kenya": m["pool"] == "kenya",
    }.items():
        sub = m[mask.to_numpy()]
        if sub.empty:
            continue
        arr_a = (sub["y_wrong_a"].to_numpy(), sub[f"p_{det_a}_a"].to_numpy(),
                 sub["abstained_a"].to_numpy())
        arr_b = (sub["y_wrong_b"].to_numpy(), sub[f"p_{det_b}_b"].to_numpy(),
                 sub["abstained_b"].to_numpy())
        res = {
            "n": len(sub),
            "delta_internal": paired_bootstrap_diff(_internal, arr_a, arr_b, n_boot, seed=seed),
            "delta_expressed": paired_bootstrap_diff(_expressed, arr_a, arr_b, n_boot, seed=seed),
            "delta_gap": paired_bootstrap_diff(_gap, arr_a, arr_b, n_boot, seed=seed),
            "abstain_rate": {"a": float(arr_a[2].mean()), "b": float(arr_b[2].mean())},
            "mcnemar_abstain": mcnemar(arr_a[2], arr_b[2]),
            "wrong_rate": {"a": float(arr_a[0].mean()), "b": float(arr_b[0].mean())},
        }
        out["sets"][set_name] = res
    pvals = {f"{k}/mcnemar": v["mcnemar_abstain"]["p_value"] for k, v in out["sets"].items()}
    out["holm_adjusted"] = holm(pvals) if pvals else {}
    dest = Path(cfg_a["paths"]["reports_dir"]) / "compare" / f"{out['a']}__vs__{out['b']}.json"
    write_json(dest, _nan_to_none(out))
    log.info("wrote %s", dest)
    return out


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
