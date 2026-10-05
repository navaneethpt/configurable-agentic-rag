"""Common contracts for independent agents and tools."""
from dataclasses import dataclass, field
from typing import Any


class ChatError(RuntimeError):
    """A safe error for the UI."""


@dataclass(frozen=True)
class Source:
    number: int
    filename: str
    location: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentOutput:
    kind: str
    data: dict[str, Any] | list[Any]


@dataclass(frozen=True)
class NodeExecution:
    node_id: str
    node_type: str
    step: int
    output: AgentOutput


@dataclass(frozen=True)
class SessionServices:
    library: Any


@dataclass(frozen=True)
class NodeInput:
    question: str
    history: list[dict[str, str]]
    prior_results: tuple[NodeExecution, ...]
    services: SessionServices
    node_id: str
    step: int
    is_terminal: bool

    def latest(self, kind: str) -> NodeExecution | None:
        return next((item for item in reversed(self.prior_results) if item.output.kind == kind), None)

    def outputs(self) -> list[dict[str, Any]]:
        """Serializable context; services and runtime control are never model inputs."""
        return [{"node_id": item.node_id, "node_type": item.node_type, "step": item.step,
                 "kind": item.output.kind, "data": item.output.data} for item in self.prior_results]


@dataclass(frozen=True)
class NodeResult:
    output: AgentOutput
    route: str | None = "next"
    response: Answer | None = None
    details: dict[str, Any] = field(default_factory=dict)
