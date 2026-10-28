"""Semantic entropy - the offline reference detector (Kuhn et al. 2023; Kossen et al. 2024).

Planned (Core tier, after the M0 baseline works). Never served: 5-10 generations per
question cannot meet any latency budget. Used as a label source and as the ceiling the
cheap detectors are measured against.

Recipe:
    1. For each question, sample K answers (temperature ~1.0, K = 5-10) with the open prompt.
    2. Cluster answers by meaning: two answers share a cluster if each entails the other,
       judged by an NLI model (e.g. a DeBERTa-MNLI checkpoint) or an LLM.
    3. Estimate cluster probabilities by frequency (discrete semantic entropy) or by
       summing sequence likelihoods within each cluster.
    4. SE = -sum_c p(c) log p(c). High SE = the model's meanings disagree = likely a bluff.
    5. Save per-question SE to data/interim/<variant>/semantic_entropy.jsonl and evaluate
       it as one more detector column in the evaluate stage.

Extension (from SEPs): train probes to predict SE from single-generation hidden states,
and compare against probes trained on correctness labels.

CRISP-DM: modeling (a reference detector). Not part of `bluff all`.
"""

from __future__ import annotations

from typing import Any


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError(
        "semantic entropy is planned - see the recipe in this module's docstring"
    )
