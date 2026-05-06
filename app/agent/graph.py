"""Assemble the LangGraph state machine for the agent.

Topology: ``router`` -> {``rag`` -> {``synthesis``, ``fallback``},
``direct``} -> END. See README for the rendered diagram.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    direct_node,
    fallback_node,
    rag_node,
    router_node,
    synthesis_node,
)
from app.agent.tools import set_retriever
from app.agent.types import AgentState
from app.llm.ollama_client import OllamaClient
from app.rag.retriever import Retriever


def _decide_route(state: AgentState) -> str:
    return "rag" if state.get("route") == "rag" else "direct"


def _decide_after_rag(state: AgentState) -> str:
    result = state.get("retrieved")
    if result is not None and len(result.hits) > 0:
        return "synthesis"
    return "fallback"


def build_agent_graph(
    *,
    llm: OllamaClient,
    retriever: Retriever,
) -> CompiledStateGraph[AgentState, Any, Any, Any]:
    """Wire and compile the agent's LangGraph.

    Args:
        llm: Ollama client used by ``router``, ``synthesis``, ``direct``.
        retriever: Bound to the ``rag_search`` tool's module-level slot.

    Returns:
        A compiled, invokable LangGraph; call ``.invoke(initial_state)``.
    """
    set_retriever(retriever)

    builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)

    builder.add_node("router", partial(router_node, llm=llm))
    builder.add_node("rag", rag_node)
    builder.add_node("synthesis", partial(synthesis_node, llm=llm))
    builder.add_node("direct", partial(direct_node, llm=llm))
    builder.add_node("fallback", fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _decide_route,
        {"rag": "rag", "direct": "direct"},
    )
    builder.add_conditional_edges(
        "rag",
        _decide_after_rag,
        {"synthesis": "synthesis", "fallback": "fallback"},
    )
    builder.add_edge("synthesis", END)
    builder.add_edge("direct", END)
    builder.add_edge("fallback", END)

    return builder.compile()


def render_mermaid(graph: CompiledStateGraph[Any, Any, Any, Any]) -> str:
    """Return a Mermaid diagram of the compiled graph (used in the README)."""
    return graph.get_graph().draw_mermaid()
