"""Unit tests: answer matching, abstention detection, feature extraction, both thresholds,
the answer-or-abstain verdict and the prompt templates.
"""

import numpy as np
import pytest

from bluff.decision import cost_threshold, tau_at_false_flag_rate, verdict
from bluff.matching import alias_match, is_abstention, is_correct, normalize
from bluff.prompts import build_prompt
from bluff.schema import QuestionItem
from bluff.signals import SUMMARY_FEATURES, select_hidden, summary_features


def test_normalize_strips_articles_and_punctuation():
    assert normalize("The  Social Health Authority.") == "social health authority"


def test_alias_match_whole_words_only():
    assert alias_match("It is Nairobi, Kenya", ["Nairobi"])
    assert not alias_match("Nairobian", ["Nairobi"])


@pytest.mark.parametrize(
    "text", ["I don't know.", "I do not know", "Sijui", "There is no such county"]
)
def test_abstention_phrases(text):
    assert is_abstention(text)


def test_is_correct_handles_unanswerable_and_refusals():
    real = QuestionItem(id="q", pool="kenya", question="How many counties?", gold="47",
                        aliases=["forty-seven"])
    fake = QuestionItem(id="u", pool="kenya_unanswerable", question="Name the 48th county?",
                        answerable=False)
    assert is_correct("47", real)
    assert is_correct("Kenya has forty-seven", real)
    assert not is_correct("I don't know", real)  # a refusal is not a correct answer
    assert is_correct("There is no 48th county.", fake)
    assert not is_correct("Ngong", fake)


def test_summary_features_shape_and_values():
    f = summary_features([0.9, 0.5, 0.2], [0.1, 1.0, 2.0], [3.0, 1.0])
    assert list(f) == SUMMARY_FEATURES
    assert f["n_tokens"] == 3
    assert f["min_logprob"] == pytest.approx(np.log(0.2))
    assert f["frac_low_prob"] == pytest.approx(1 / 3)
    assert f["retrieval_top_score"] == 3.0
    assert summary_features([], [])["n_tokens"] == 0.0


def test_select_hidden_single_and_batched():
    h = np.arange(2 * 2 * 3).reshape(2, 2, 3)
    assert select_hidden(h, 1, 0).tolist() == [6, 7, 8]
    assert select_hidden(h[None], 1, 0).shape == (1, 3)


def test_cost_threshold():
    assert cost_threshold(3.0, 1.0) == 0.25
    with pytest.raises(ValueError):
        cost_threshold(0, 1)


def test_tau_flags_requested_share_of_correct_answers():
    scores = np.linspace(0, 1, 101)
    tau = tau_at_false_flag_rate(scores, 0.10)
    assert np.mean(scores > tau) == pytest.approx(0.10, abs=0.011)


def test_verdict():
    assert verdict(0.9, 0.25, False) == "abstain"
    assert verdict(0.1, 0.25, True) == "answer_with_citation"
    assert verdict(0.1, 0.25, False) == "answer"


def test_prompts():
    assert "I don't know" in build_prompt("Q?", "offered", 2)
    assert "I don't know" not in build_prompt("Q?", "open")
    assert "[1] passage" in build_prompt("Q?", "open", passages=["passage"])
    with pytest.raises(ValueError):
        build_prompt("Q?", "other")
