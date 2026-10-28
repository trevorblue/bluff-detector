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


class AskResponse(BaseModel):
    request_id: str
    verdict: Literal["answer", "answer_with_citation", "abstain"]
    answer: str | None = Field(description="null when the gate abstains")
    bluff_score: float = Field(ge=0.0, le=1.0, description="detector P(wrong)")
    threshold: float = Field(description="abstain when bluff_score > threshold")
    sources: list[str] = Field(default_factory=list, description="retrieved passage ids")
    variant: str
    detector: str
    model_version: str
    latency_ms: float
