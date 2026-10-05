"""Shared schemas and readers for optional agent outputs."""
from typing import Literal
from pydantic import BaseModel, Field
from ..agent_contracts import NodeInput
from ..model_client import MODEL
from ..workflows import NodeConfig

NO_EVIDENCE = "I couldn't find enough information in your uploaded documents to answer that question."
MAX_ROUNDS = 3
FINAL_ATTEMPT_CONFIDENCE = 0.5
MAX_SEARCHES_PER_ROUND = 3
MAX_RESULTS_PER_SEARCH = 3
MAX_EVIDENCE = 10


class ModelConfig(NodeConfig):
    model: str = Field(default=MODEL, min_length=1, max_length=100)
    instructions: str = Field(default="", max_length=4000)


class PlannerConfig(ModelConfig):
    max_searches: int = Field(default=MAX_SEARCHES_PER_ROUND, ge=1, le=6)


class RetrieveConfig(NodeConfig):
    max_results_per_search: int = Field(default=MAX_RESULTS_PER_SEARCH, ge=1, le=10)
    max_evidence: int = Field(default=MAX_EVIDENCE, ge=1, le=30)


class ValidateConfig(ModelConfig):
    max_rounds: int = Field(default=MAX_ROUNDS, ge=1, le=6)
    final_confidence: float = Field(default=FINAL_ATTEMPT_CONFIDENCE, ge=0, le=1)


class SearchTask(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    purpose: str = Field(min_length=1, max_length=300)


class PlannerOutput(BaseModel):
    searches: list[SearchTask] = Field(min_length=1, max_length=6)


class Passage(BaseModel):
    chunk_id: str
    filename: str
    location: str
    text: str


class RetrievalOutput(BaseModel):
    passages: list[Passage]
    searches: list[SearchTask]
    round: int


class ValidatorOutput(BaseModel):
    decision: Literal["sufficient", "needs_more_evidence"]
    confidence: float = Field(ge=0, le=1, strict=True)
    missing_evidence: list[str] = Field(default_factory=list, max_length=MAX_SEARCHES_PER_ROUND)


class ValidationResult(ValidatorOutput):
    accepted: bool
    acceptance_reason: str
    round: int
    round_limit: int
    evidence_step: int


class GenerationOutput(BaseModel):
    status: Literal["answered", "missing_context"]
    text: str = Field(min_length=1, max_length=12000, description="For answered responses, cite every factual claim with an available source number such as [1]. For missing_context, request needed context without facts or citations.")


class SourceOutput(BaseModel):
    number: int
    filename: str
    location: str
    text: str


class ResponseOutput(GenerationOutput):
    sources: list[SourceOutput] = Field(default_factory=list)


class EvidenceRequestOutput(BaseModel):
    text: str
    missing_evidence: list[str]


def rounds(inputs: NodeInput) -> int:
    return sum(item.output.kind == "passages" for item in inputs.prior_results)


def passages(inputs: NodeInput) -> list[dict]:
    unique = {}
    for item in inputs.prior_results:
        if item.output.kind == "passages":
            for passage in item.output.data["passages"]:
                unique.setdefault(passage["chunk_id"], passage)
    return list(unique.values())


def searches(inputs: NodeInput) -> list[dict]:
    latest = inputs.latest("search_queries")
    return latest.output.data["searches"] if latest else [{"query": inputs.question, "purpose": "Answer the user question"}]


def validation(inputs: NodeInput) -> dict | None:
    latest = inputs.latest("validation")
    evidence = inputs.latest("passages")
    if latest and (not evidence or latest.output.data["evidence_step"] >= evidence.step):
        return latest.output.data
    return None


def source_payload(inputs: NodeInput) -> list[dict]:
    return [{"number": index, **{key: value for key, value in passage.items() if key != "chunk_id"}}
            for index, passage in enumerate(passages(inputs), 1)]
