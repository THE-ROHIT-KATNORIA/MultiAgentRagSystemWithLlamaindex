"""
RAG Query Engine
================

Builds a RetrieverQueryEngine with hybrid retrieval,
reranking, and optional streaming.
"""

from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from retrieval.hybrid import create_hybrid_retriever


def create_rag_engine(question: str, index, nodes, reranker, similarity_filter,
                      filters=None, streaming: bool = False):
    """Create a dynamically filtered RAG query engine.

    Args:
        streaming: If True, the synthesizer yields tokens via response_gen.
    Returns:
        (RetrieverQueryEngine, MetadataFilters | None)
    """
    hybrid_retriever, filters = create_hybrid_retriever(
        question, index, nodes, filters=filters
    )

    postprocessors = []
    if reranker is not None:
        postprocessors.append(reranker)
    postprocessors.append(similarity_filter)

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        streaming=streaming,
    )

    engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )
    return engine, filters
