"""
Tool: RAG Document Search
=========================

Searches company documents using the hybrid RAG pipeline.
"""

from llama_index.core.tools import FunctionTool
from retrieval.router import dynamic_route_and_filter
from engine.rag_engine import create_rag_engine

# Injected at agent startup via build_rag_tool()
_components: dict = {}


def search_company_documents(question: str) -> str:
    """Search company HR, IT, and Finance policy documents."""
    try:
        decision = dynamic_route_and_filter(question)
        engine, _ = create_rag_engine(
            question,
            _components["index"],
            _components["nodes"],
            _components["reranker"],
            _components["sim_filter"],
            filters=decision["filters"],
        )
        response = engine.query(question)
        answer = str(response)
        if hasattr(response, "source_nodes") and response.source_nodes:
            sources = [
                s.node.metadata.get("file_name", "Unknown")
                for s in response.source_nodes[:3]
            ]
            answer += f"\n\n[Sources: {', '.join(sources)}]"
        return answer
    except Exception as exc:
        return f"RAG search failed: {exc}"


def build_rag_tool(index, nodes, reranker, sim_filter) -> FunctionTool:
    """Inject RAG components and return the tool."""
    _components.update({
        "index": index,
        "nodes": nodes,
        "reranker": reranker,
        "sim_filter": sim_filter,
    })
    return rag_tool


rag_tool = FunctionTool.from_defaults(
    fn=search_company_documents,
    name="search_documents",
    description=(
        "Search the company's uploaded documents for policies, rules, "
        "procedures, leave policy, IT security, employee handbook, and "
        "other company-specific information. Use this for ANY question "
        "about internal company rules or policies."
    ),
)
