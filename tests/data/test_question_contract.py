"""Data contracts: run on every commit, because data changes without a commit."""

from bluff.pipeline.validate import load_kenya
from bluff.schema import QuestionItem, validate_questions
from tests.conftest import ROOT

KENYA = ROOT / "data" / "kenya" / "questions.jsonl"


def test_kenya_question_set_passes_contract():
    items = load_kenya(KENYA)
    assert items, "the Kenya question file is empty"
    assert validate_questions(items) == []


def test_kenya_set_has_unanswerable_items():
    assert any(not it.answerable for it in load_kenya(KENYA))


def test_contract_catches_violations():
    bad = [
        QuestionItem(id="x", pool="kenya", question="Q one?", gold=None),
        QuestionItem(id="x", pool="kenya", question="Q two?", gold="a"),
        QuestionItem(id="u", pool="kenya", question="Q three?", answerable=False),
        QuestionItem(id="v", pool="kenya", question="Q four?", gold="a", verified=True,
                     author="t", reviewer="t", source_note="s"),
        QuestionItem(id="w", pool="kenya", question="Q five?", gold="a", split="train"),
    ]
    errors = " | ".join(validate_questions(bad))
    for expected in ("no gold", "duplicate id", "unanswerable items belong",
                     "reviewer must differ", "test-only"):
        assert expected in errors
