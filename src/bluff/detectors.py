"""The detectors (the measuring instruments) and their on-disk artefact.

Implemented now: logreg and xgboost on summary features; linear and MLP probes on hidden
states. Planned (stretch): an LSTM over the per-token entropy sequence, and semantic
entropy as the offline reference (see bluff/pipeline/semantic_entropy.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from bluff.signals import SUMMARY_FEATURES, select_hidden

SUMMARY_KINDS = {"logreg", "xgboost"}
HIDDEN_KINDS = {"probe_linear", "probe_mlp"}
PLANNED = {"lstm_entropy", "semantic_entropy"}


@dataclass
class Detector:
    name: str
    input: str  # "summary" | "hidden"
    estimator: Any
    feature_names: list[str] = field(default_factory=list)
    layer: int | None = None
    position: str | None = None
    layer_idx: int | None = None
    position_idx: int | None = None

    def inputs(self, summary: pd.DataFrame, hidden: np.ndarray | None) -> np.ndarray:
        """The one place detector inputs are selected, for training and serving alike."""
        if self.input == "summary":
            return summary[self.feature_names].to_numpy(dtype=np.float32)
        if hidden is None:
            raise ValueError(f"{self.name} needs hidden states")
        return select_hidden(hidden, self.layer_idx, self.position_idx)

    def fit(self, summary: pd.DataFrame, hidden: np.ndarray | None, y_wrong: np.ndarray):
        self.estimator.fit(self.inputs(summary, hidden), np.asarray(y_wrong).astype(int))
        return self

    def predict_proba_wrong(self, summary: pd.DataFrame, hidden: np.ndarray | None) -> np.ndarray:
        return self.estimator.predict_proba(self.inputs(summary, hidden))[:, 1]


def make_detector(name: str, cfg: dict[str, Any]) -> Detector:
    seed = cfg["seed"]
    if name == "logreg":
        est: Pipeline | Any = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        )
        return Detector(name, "summary", est, feature_names=list(SUMMARY_FEATURES))
    if name == "xgboost":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        return Detector(name, "summary", est, feature_names=list(SUMMARY_FEATURES))
    if name in HIDDEN_KINDS:
        m, d = cfg["model"], cfg["detectors"]
        layer, position = d["probe_layer"], d["probe_position"]
        if name == "probe_linear":
            # Linear first: if a linear map suffices, the information is linearly encoded.
            est = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, random_state=seed)
            )
        else:
            est = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    alpha=1e-3,
                    early_stopping=True,
                    max_iter=500,
                    random_state=seed,
                ),
            )
        return Detector(
            name,
            "hidden",
            est,
            layer=layer,
            position=position,
            layer_idx=list(m["layers"]).index(layer),
            position_idx=list(m["positions"]).index(position),
        )
    if name in PLANNED:
        raise NotImplementedError(f"{name} is planned (stretch tier) - see detectors.py docstring")
    raise ValueError(f"unknown detector {name!r}")


def save_detector(det: Detector, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "detector.joblib"
    joblib.dump(det, path)
    return path


def load_detector(directory: Path) -> Detector:
    return joblib.load(Path(directory) / "detector.joblib")
