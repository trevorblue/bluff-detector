"""Unit tests for bluff/metrics.py: AUROC, ECE, risk-coverage, the four cells, the
verbalisation gap, bootstrap intervals, the paired difference, McNemar and Holm.
"""

import math

import numpy as np
import pytest

from bluff import metrics as M


def test_auroc_perfect_and_nan_for_one_class():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert math.isnan(M.auroc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_ece_zero_when_frequencies_match_confidence():
    # 10 predictions at 0.3, 3 wrong -> perfectly calibrated bin
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.full(10, 0.3)
    assert M.ece(y, p) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.zeros(10)
    p = np.full(10, 0.9)  # says 90% wrong, actually all correct
    assert M.ece(y, p) == pytest.approx(0.9)


def test_risk_coverage_answers_most_confident_first():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    cov, risk = M.risk_coverage(y, s)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert M.aurc(y, s) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert M.risk_at_coverage(y, s, 0.5) == 0.0


def test_four_cells_and_rates():
    flag = np.array([1, 1, 0, 0, 1])
    a = np.array([1, 0, 1, 0, 0])
    y = np.array([1, 1, 0, 0, 1])
    cells = M.four_cells(flag, a, y)
    assert cells["honest_uncertainty"]["n"] == 1
    assert cells["silent_knowing"] == {"n": 2, "wrong": 2, "correct": 0}
    assert cells["abstention_inflation"] == {"n": 1, "wrong": 0, "correct": 1}
    # confidently delivered wrong answers: rows 1 and 4, both flagged
    assert M.silent_knowing_rate(flag, a, y) == 1.0
    # abstentions: rows 0 (wrong) and 2 (correct)
    assert M.inflation_rate(a, y) == 0.5


def test_verbalisation_gap_positive_when_probe_knows_more():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # probe separates perfectly
    a = np.array([1, 0, 0, 0, 0, 0])  # model abstains on only one error
    vg = M.verbalisation_gap(y, s, a)
    assert vg["internal"] == 1.0
    assert vg["expressed"] == pytest.approx((1 / 3 + 1.0) / 2)  # balanced accuracy
    assert vg["gap"] > 0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y * 0.5 + rng.random(200) * 0.7
    ci = M.bootstrap_ci(M.auroc, y, s, n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_paired_diff_zero_for_identical_variants():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 100), rng.random(100)
    d = M.paired_bootstrap_diff(M.auroc, (y, s), (y, s), n_boot=100)
    assert d["point"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_mcnemar_and_holm():
    before = np.array([1] * 10 + [0] * 10)
    after = np.zeros(20)
    r = M.mcnemar(before, after)
    assert r["before_only"] == 10 and r["after_only"] == 0
    assert r["p_value"] < 0.01
    adj = M.holm({"a": 0.01, "b": 0.04})
    assert adj["a"] == pytest.approx(0.02) and adj["b"] == pytest.approx(0.04)
