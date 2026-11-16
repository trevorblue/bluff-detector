"""The API contract. The response is an audit record: every field answers a question a
user, an engineer or an auditor may ask about this call.

CRISP-DM: deployment.
Used by: app/main.py. FastAPI turns these classes into input checks and the /docs page.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    reveal: bool = Field(
        default=False,
        description="demo only: when the gate abstains, also return what the model would have said",
    )


class Token(BaseModel):
    """One piece of the answer as the model produced it, with its probability and entropy."""

    text: str
    prob: float
    entropy: float


class AskResponse(BaseModel):
    request_id: str
    verdict: Literal["answer", "answer_with_citation", "abstain"]
    answer: str | None = Field(description="null when the gate abstains")
    withheld_answer: str | None = Field(
        default=None, description="the blocked answer, only when the request set reveal=true"
    )
    bluff_score: float = Field(ge=0.0, le=1.0, description="detector P(wrong)")
    threshold: float = Field(description="abstain when bluff_score > threshold")
    sources: list[str] = Field(default_factory=list, description="retrieved passage ids")
    variant: str
    detector: str
    model_version: str
    latency_ms: float
    all_detectors: dict[str, float] = Field(
        default_factory=dict,
        description="P(wrong) from every trained detector of this variant, served one included",
    )
    signals: dict[str, float] = Field(
        default_factory=dict, description="the 8 confidence numbers the summary detectors read"
    )
    tokens: list[Token] = Field(
        default_factory=list,
        description="the answer word by word with its probability (empty when an abstention "
        "is not revealed)",
    )
