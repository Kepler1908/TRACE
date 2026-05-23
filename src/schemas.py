"""Pydantic schemas for LLM-structured outputs.

Used by ``pipeline.py`` and ``agent.py`` to validate planner / re-planner /
rerank responses with retry-on-error rather than silently swallowing
malformed JSON.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from .types import LLMSchemaError


class SubQuestionSpec(BaseModel):
    question: str = Field(min_length=1)
    target_corpus: str = ""
    key_entities: list[str] = Field(default_factory=list)
    spawned: bool = False

    @field_validator("key_entities", mode="before")
    @classmethod
    def _coerce_entities(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return [str(v)]


class PlannerOutput(BaseModel):
    query_analysis: str = ""
    sub_questions: list[SubQuestionSpec] = Field(default_factory=list)
    entities_to_find: list[str] = Field(default_factory=list)
    dates_mentioned: list[str] = Field(default_factory=list)

    @field_validator("sub_questions", mode="before")
    @classmethod
    def _drop_empty(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [sq for sq in v if isinstance(sq, dict) and str(sq.get("question", "")).strip()]


class ReplanOutput(BaseModel):
    revised_sub_questions: list[SubQuestionSpec] = Field(default_factory=list)
    reasoning: str = ""
    search_strategy: str = ""
    key_facts: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)

    @field_validator("revised_sub_questions", mode="before")
    @classmethod
    def _drop_empty(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        return [sq for sq in v if isinstance(sq, dict) and str(sq.get("question", "")).strip()]


def validate_planner(payload: Any) -> PlannerOutput:
    try:
        return PlannerOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMSchemaError(f"planner schema invalid: {exc.errors()[:3]}") from exc


def validate_replan(payload: Any) -> ReplanOutput:
    try:
        return ReplanOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMSchemaError(f"replan schema invalid: {exc.errors()[:3]}") from exc
