"""Config loading with single inheritance, plus the paths every stage reads and writes.

CRISP-DM: shared by every phase.
Used by: every pipeline stage, the API and the tests. paths() is the one place that decides
where each stage's files live.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", None)
    base = _load((path.parent / parent).resolve()) if parent else {}
    return deep_merge(base, raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a variant config. `overrides` is merged last (used by tests to redirect paths)."""
    cfg = _load(Path(path).resolve())
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path)
    _check(cfg)
    return cfg


def _check(cfg: dict[str, Any]) -> None:
    det, model = cfg["detectors"], cfg["model"]
    if det["probe_layer"] not in model["layers"]:
        raise ValueError(f"detectors.probe_layer {det['probe_layer']} not in model.layers")
    if det["probe_position"] not in model["positions"]:
        raise ValueError(f"detectors.probe_position {det['probe_position']} not in positions")
    if det["serve"] not in det["train"]:
        raise ValueError("detectors.serve must be one of detectors.train")


@dataclass(frozen=True)
class Paths:
    questions: Path  # shared by all variants: same questions, same splits
    manifest: Path
    interim: Path  # raw generations for this variant
    features: Path  # feature tables for this variant
    artefacts: Path  # trained detectors for this variant
    serving: Path  # the detector the API loads
    reports: Path
    registry: Path


def paths(cfg: dict[str, Any]) -> Paths:
    vid = cfg["variant"]["id"]
    data = Path(cfg["paths"]["data_dir"])
    art = Path(cfg["paths"]["artefacts_dir"])
    return Paths(
        questions=data / "processed" / "questions.jsonl",
        manifest=data / "processed" / "manifest.json",
        interim=data / "interim" / vid,
        features=data / "processed" / vid,
        artefacts=art / vid,
        serving=art / vid / "serving",
        reports=Path(cfg["paths"]["reports_dir"]) / vid,
        registry=art / "registry",
    )
