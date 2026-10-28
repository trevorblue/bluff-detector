"""Shared test setup. fake_config() points a dev_fake config at a temporary folder; the
`trained` fixture runs the whole fake pipeline once, for the integration and API tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bluff.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEV_FAKE = ROOT / "configs" / "variants" / "dev_fake.yaml"


def fake_config(tmp: Path, variant: str = "dev_fake.yaml", **extra) -> dict:
    overrides = {
        "paths": {
            "data_dir": str(tmp / "data"),
            "artefacts_dir": str(tmp / "artefacts"),
            "reports_dir": str(tmp / "reports"),
        },
        "data": {
            "kenya_questions": str(ROOT / "data" / "kenya" / "questions.jsonl"),
            "retrieval_corpus": str(tmp / "corpus" / "retrieval_chunks.jsonl"),
            "finetune_corpus": str(tmp / "corpus" / "finetune_docs.jsonl"),
            "synthetic_trivia": 240,
            "synthetic_kenya": 80,
            "synthetic_unanswerable": 30,
        },
        "evaluation": {"bootstrap": 100},
        **extra,
    }
    return load_config(ROOT / "configs" / "variants" / variant, overrides=overrides)


def run_all(cfg: dict) -> None:
    from bluff.cli import STAGES, _stage

    for name in STAGES:
        _stage(name)(cfg)


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> dict:
    """One full fake-backend pipeline run, shared by the integration and API tests."""
    cfg = fake_config(tmp_path_factory.mktemp("run"))
    run_all(cfg)
    return cfg
