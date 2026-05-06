"""Domain types for the agent layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from app.rag.types import RetrievalResult

Route = Literal["rag", "direct"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation. ``error`` is set when the call raised."""

    name: str
    input: dict[str, Any]
    output: dict[str, Any] | None
    elapsed_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One ordered event in the agent's audit log."""

    node: str
    timestamp: str
    elapsed_ms: float
    inputs: dict[str, Any]
    outputs: dict[str, Any]


class AgentState(TypedDict, total=False):
    """LangGraph state. ``Annotated[list, add]`` makes the audit fields
    append-only so each node can return only its own new entries."""

    query: str
    route: Route | None
    retrieved: RetrievalResult | None
    answer: str | None
    tool_calls: Annotated[list[ToolCall], add]
    trace: Annotated[list[TraceEvent], add]


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Immutable result returned by ``AgentRunner.run``."""

    query: str
    route: Route
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
