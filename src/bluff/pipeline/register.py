"""Stage 7 - register the served detector and apply the promotion gates.

A new version always gets registered. It becomes @challenger only if it clears every
gate against the current @champion; the first ever version becomes @champion directly.
Promoting a challenger to champion is a deliberate, separate step (`bluff promote`), the
human sign-off point.
"""

from __future__ import annotations

import logging
from typing import Any

from bluff.config import paths
from bluff.registry import LocalRegistry
from bluff.utils import read_json

log = logging.getLogger(__name__)


def gate_report(new: dict[str, Any], old: dict[str, Any] | None, gates: dict[str, Any]) -> dict:
    """Compare served-detector metrics. `new`/`old` are summaries from summarise()."""
    checks = {}
    gap = new.get("pool_gap")
    checks["pool_gap"] = gap is None or gap <= gates["max_gap_between_pools"]
    if old is not None:
        checks["auroc"] = _ge(new["auroc"], _add(old["auroc"], gates["min_auroc_gain"]))
        checks["ece"] = _le(new["ece"], _add(old["ece"], gates["max_ece_increase"]))
    return {"passed": all(checks.values()), "checks": checks}


def summarise(metrics: dict[str, Any]) -> dict[str, Any]:
    served = metrics["detectors"][metrics["served_detector"]]
    sets = served["sets"]
    main = sets.get("trivia_test") or next(iter(sets.values()))
    return {"auroc": main["auroc"], "ece": main["ece"], "pool_gap": served["ood_auroc_drop"]}


def _add(a, b):
    return None if a is None else a + b


def _ge(a, b):
    return a is not None and (b is None or a >= b)


def _le(a, b):
    return a is not None and (b is None or a <= b)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    p = paths(cfg)
    reg = LocalRegistry(p.registry)
    name = f"{cfg['tracking']['registered_model']}-{cfg['variant']['id']}"
    metrics = read_json(p.reports / "metrics.json")
    new = summarise(metrics)
    champ = reg.get(name, "champion")
    report = gate_report(new, champ["summary"] if champ else None, cfg["gates"])
    meta = read_json(p.serving / "metadata.json")
    version = reg.register(
        name, p.serving, {"summary": new, "gates": report, "git_sha": meta["git_sha"],
                          "data_sha256": meta["data_sha256"], "detector": meta["detector"]}
    )
    if champ is None:
        reg.set_alias(name, "champion", version)
        log.info("%s v%d registered as @champion (first version)", name, version)
    elif report["passed"]:
        reg.set_alias(name, "challenger", version)
        log.info("%s v%d passed the gates -> @challenger", name, version)
    else:
        log.warning("%s v%d failed gates %s - not aliased", name, version, report["checks"])

    if cfg["tracking"]["mlflow"]:
        _mirror_to_mlflow(cfg, name, p.serving, version, champ is None, report["passed"])
    return {"model": name, "version": version, "gates": report}


def promote(cfg: dict[str, Any], version: int, alias: str = "champion") -> None:
    """Move an alias. Rollback = promote the previous version. Nothing is rebuilt."""
    p = paths(cfg)
    name = f"{cfg['tracking']['registered_model']}-{cfg['variant']['id']}"
    LocalRegistry(p.registry).set_alias(name, alias, version)
    log.info("%s @%s -> v%d", name, alias, version)


def _mirror_to_mlflow(cfg, name, artefact_dir, version, is_first, passed) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(cfg["tracking"]["tracking_uri"])
    with mlflow.start_run(run_name=f"register/{name}/v{version}") as run:
        mlflow.log_artifacts(str(artefact_dir), artifact_path="artefact")
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/artefact", name)
    client = MlflowClient()
    if is_first:
        client.set_registered_model_alias(name, "champion", mv.version)
    elif passed:
        client.set_registered_model_alias(name, "challenger", mv.version)
