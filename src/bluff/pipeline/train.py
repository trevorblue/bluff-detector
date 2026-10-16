"""Stage 5 - train every detector on the training split, set thresholds on validation,
and package each one as an artefact (weights + input spec + thresholds + metadata).

Only trivia-style pools are used for training. Kenya never enters this stage.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from bluff import __version__
from bluff.config import paths
from bluff.decision import cost_threshold, tau_at_false_flag_rate
from bluff.detectors import make_detector, save_detector
from bluff.metrics import aurc, auroc, ece, pr_auc
from bluff.pipeline.featurize import load_features
from bluff.schema import TRAIN_POOLS
from bluff.utils import file_sha256, git_sha, set_seed, write_json

log = logging.getLogger(__name__)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(cfg["seed"])
    p = paths(cfg)
    df, hidden = load_features(cfg)
    in_dist = df["pool"].isin(TRAIN_POOLS).to_numpy()
    tr = in_dist & (df["split"] == "train").to_numpy()
    va = in_dist & (df["split"] == "val").to_numpy()
    if df.loc[tr, "y_wrong"].nunique() < 2:
        raise ValueError("training split has a single class - need right and wrong answers")

    dec = cfg["decision"]
    t_abstain = cost_threshold(dec["cost_confident_wrong"], dec["cost_needless_abstain"])
    common = {
        "variant": cfg["variant"]["id"],
        "base_model": cfg["model"]["name"],
        "adapter": cfg["model"].get("adapter"),
        "backend": cfg["model"]["backend"],
        "layers": cfg["model"]["layers"],
        "positions": cfg["model"]["positions"],
        "retrieval": cfg["retrieval"]["enabled"],
        "t_abstain": t_abstain,
        "cost_matrix": dec,
        "git_sha": git_sha(),
        "data_sha256": file_sha256(p.questions),
        "config": cfg["_config_path"],
        "seed": cfg["seed"],
        "version": f"{__version__}+{git_sha()}",
    }

    tracker = _Tracker(cfg)
    results = {}
    for name in cfg["detectors"]["train"]:
        det = make_detector(name, cfg)
        start = time.perf_counter()
        det.fit(df[tr], hidden[tr], df.loc[tr, "y_wrong"].to_numpy())
        fit_s = time.perf_counter() - start
        y_val = df.loc[va, "y_wrong"].to_numpy()
        s_val = det.predict_proba_wrong(df[va], hidden[va])
        tau = tau_at_false_flag_rate(s_val[y_val == 0], cfg["evaluation"]["probe_flag_fpr"])
        val = {
            "auroc": auroc(y_val, s_val),
            "pr_auc": pr_auc(y_val, s_val),
            "ece": ece(y_val, s_val),
            "aurc": aurc(y_val, s_val),
        }
        meta = {
            **common,
            "detector": name,
            "input": det.input,
            "probe_layer": det.layer,
            "probe_position": det.position,
            "tau_flag": tau,
            "val": val,
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "fit_seconds": round(fit_s, 3),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out_dir = p.artefacts / name
        save_detector(det, out_dir)
        write_json(out_dir / "metadata.json", meta)
        tracker.log(name, meta, out_dir)
        results[name] = val
        log.info("%-13s val AUROC %.3f  ECE %.3f", name, val["auroc"], val["ece"])

    served = cfg["detectors"]["serve"]
    if p.serving.exists():
        shutil.rmtree(p.serving)
    shutil.copytree(p.artefacts / served, p.serving)
    log.info("serving artefact: %s -> %s", served, p.serving)
    return {"val": results, "served": served, "t_abstain": t_abstain}


class _Tracker:
    """Logs runs to MLflow when tracking.mlflow is true; otherwise a no-op."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enabled = bool(cfg["tracking"]["mlflow"])
        if self.enabled:
            import mlflow

            mlflow.set_tracking_uri(cfg["tracking"]["tracking_uri"])
            mlflow.set_experiment(cfg["tracking"]["experiment"])
            self.mlflow = mlflow

    def log(self, name: str, meta: dict[str, Any], artefact_dir: Path) -> None:
        if not self.enabled:
            return
        mlflow = self.mlflow
        with mlflow.start_run(run_name=f"{meta['variant']}/{name}"):
            mlflow.set_tags(
                {"git_sha": meta["git_sha"], "data_sha256": meta["data_sha256"],
                 "variant": meta["variant"], "detector": name}
            )
            mlflow.log_params(
                {"detector": name, "base_model": meta["base_model"], "seed": meta["seed"],
                 "probe_layer": meta["probe_layer"], "probe_position": meta["probe_position"],
                 "t_abstain": meta["t_abstain"]}
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in meta["val"].items()
                                if v == v})  # drop NaN
            mlflow.log_artifacts(str(artefact_dir), artifact_path="artefact")
