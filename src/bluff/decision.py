"""Thresholds: when the gate abstains, and when the probe 'says uncertain'.

CRISP-DM: business understanding (the cost of a wrong answer vs a needless refusal),
applied at deployment.
Used by: pipeline/train.py (sets both thresholds) and app/main.py (verdict).
"""

from __future__ import annotations

import numpy as np


def cost_threshold(cost_confident_wrong: float, cost_needless_abstain: float) -> float:
    """Abstain when the expected cost of answering exceeds the cost of refusing.

    answer costs  p * C_wrong ; abstaining costs (1 - p) * C_abstain
    => abstain iff p > C_abstain / (C_abstain + C_wrong)
    Valid only if p = P(wrong) is calibrated, so ECE is reported beside it.
    """
    if cost_confident_wrong <= 0 or cost_needless_abstain <= 0:
        raise ValueError("costs must be positive")
    return cost_needless_abstain / (cost_needless_abstain + cost_confident_wrong)


def tau_at_false_flag_rate(scores_of_correct_answers: np.ndarray, rate: float) -> float:
    """Threshold above which the probe 'says uncertain', fixed on validation data.

    Chosen so that `rate` of correct answers are flagged (a false-flag rate). Fixed before
    any Kenya question is scored, never tuned on test data.
    """
    s = np.asarray(scores_of_correct_answers, dtype=float)
    if s.size == 0:
        raise ValueError("need at least one correct validation answer to set tau")
    return float(np.quantile(s, 1.0 - rate))


def verdict(p_wrong: float, threshold: float, has_sources: bool) -> str:
    if p_wrong > threshold:
        return "abstain"
    return "answer_with_citation" if has_sources else "answer"
