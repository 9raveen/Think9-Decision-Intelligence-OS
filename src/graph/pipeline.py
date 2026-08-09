"""LangGraph state graph: retrieve -> route -> cross_brand -> governance -> synthesize"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.cross_brand_agent import CrossBrandResult, analyze as cross_brand_analyze
from src.agents.router import RoutingResult, route as router_route
from src.agents.synthesis_agent import SynthesisResult, synthesize as synthesis_synthesize
from src.governance.rules import GovernanceResult, evaluate as governance_evaluate
from src.retrieval.hybrid_retriever import EvidenceBundle, HybridRetriever


class PipelineState(TypedDict, total=False):
    query_text: str
    bundle: EvidenceBundle
    routing: RoutingResult
    cross_brand: CrossBrandResult
    governance: GovernanceResult
    synthesis: SynthesisResult


def build_pipeline(hybrid_retriever: HybridRetriever):
    graph = StateGraph(PipelineState)

    def retrieve_node(state): return {"bundle": hybrid_retriever.retrieve(state["query_text"])}
    def route_node(state): return {"routing": router_route(state["bundle"])}
    def cross_brand_node(state): return {"cross_brand": cross_brand_analyze(state["bundle"])}
    def governance_node(state):
        cb = state["cross_brand"]
        return {"governance": governance_evaluate(cb.relevant_decisions, cb.behavior == "conflict_flag_human_review")}
    def synthesis_node(state):
        return {"synthesis": synthesis_synthesize(state["query_text"], state["bundle"], state["cross_brand"], state["governance"])}

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("cross_brand", cross_brand_node)
    graph.add_node("governance", governance_node)
    graph.add_node("synthesize", synthesis_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "route")
    graph.add_edge("route", "cross_brand")
    graph.add_edge("cross_brand", "governance")
    graph.add_edge("governance", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_query(hybrid_retriever: HybridRetriever, query_text: str) -> PipelineState:
    return build_pipeline(hybrid_retriever).invoke({"query_text": query_text})