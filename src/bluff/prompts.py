"""Prompt templates for the two conditions.

open    -> the model answers freely. Gives correctness and the internal signals.
offered -> an explicit "I don't know" option, in several paraphrases, so abstention is
           measured structurally (did it take the option?) rather than by judging tone.
           Several templates control for prompt artifacts ("abstention inflation").
"""

from __future__ import annotations

OPEN_TEMPLATE = (
    "Answer the question with a short factual answer of a few words. Do not explain.\n"
    "Question: {question}\nAnswer:"
)

OFFERED_TEMPLATES = (
    "Answer the question with a short factual answer of a few words. "
    "If you do not know the answer, reply exactly: I don't know.\n"
    "Question: {question}\nAnswer:",
    "Question: {question}\n"
    'Give a short answer, or reply "I don\'t know" if you are not sure.\nAnswer:',
    "You may answer or decline. Reply with a short answer, or with \"I don't know\" "
    "if you cannot answer reliably.\nQuestion: {question}\nAnswer:",
)

CONTEXT_BLOCK = "Use the context if it is relevant.\nContext:\n{passages}\n\n"


def build_prompt(
    question: str, condition: str, template: int = 0, passages: list[str] | None = None
) -> str:
    if condition == "open":
        body = OPEN_TEMPLATE.format(question=question)
    elif condition == "offered":
        body = OFFERED_TEMPLATES[template % len(OFFERED_TEMPLATES)].format(question=question)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    if passages:
        numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        body = CONTEXT_BLOCK.format(passages=numbered) + body
    return body
