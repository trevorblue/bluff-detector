"""Monitoring: PSI behaves, and `bluff monitor` tells stable traffic from drifted traffic."""

import numpy as np
import pandas as pd
import pytest

from bluff import monitoring
from bluff.config import paths
from bluff.signals import SUMMARY_FEATURES
from bluff.utils import read_json

pytestmark = pytest.mark.integration


def test_psi_is_zero_for_the_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert monitoring.psi(ref, rng.normal(size=5000)) < 0.02
    assert monitoring.psi(ref, rng.normal(loc=1.0, size=5000)) > 0.25


def _write_log(path, feats: pd.DataFrame, shift: float, meta: dict):
    log = monitoring.RequestLog(str(path))
    for _, row in feats.iterrows():
        f = {k: float(row[k]) + (shift if k != "retrieval_top_score" else 0.0)
             for k in SUMMARY_FEATURES}
        log.write({"request_id": "x", "question": "q?", "verdict": "answer", "bluff_score": 0.1,
                   "threshold": meta["t_abstain"], "latency_ms": 5.0, "variant": meta["variant"],
                   "detector": meta["detector"], "model_version": meta["version"], "features": f})


def test_monitor_flags_shifted_traffic_only(trained, tmp_path):
    p = paths(trained)
    meta = read_json(p.serving / "metadata.json")
    feats = pd.read_parquet(p.features / "features.parquet")
    sample = feats[feats.split == "test"].head(60)

    _write_log(tmp_path / "same.jsonl", sample, 0.0, meta)
    same = monitoring.run(trained, tmp_path / "same.jsonl")
    _write_log(tmp_path / "shifted.jsonl", sample, 5.0, meta)
    shifted = monitoring.run(trained, tmp_path / "shifted.jsonl")

    assert same["live_requests"] == shifted["live_requests"] == 60
    checks = read_json(p.reports / "monitoring" / "drift.json")["checks"]
    assert shifted["overall"] == "DRIFT"
    assert all(c["status"] == "DRIFT" for c in checks if c["kind"] == "data")
    assert (p.reports / "monitoring" / "drift_report.md").exists()


def test_monitor_with_no_traffic_says_too_early(trained, tmp_path):
    result = monitoring.run(trained, tmp_path / "missing.jsonl")
    assert result["live_requests"] == 0 and result["overall"] == "TOO EARLY"
