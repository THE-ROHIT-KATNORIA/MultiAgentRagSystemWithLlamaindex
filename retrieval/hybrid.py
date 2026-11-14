"""
Hybrid Retriever
================

Combines vector search (pgvector) + BM25 keyword search
via QueryFusionRetriever. BM25 index is cached per department.
"""

import logging
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from config.settings import VECTOR_TOP_K, BM25_TOP_K, FUSION_NUM_QUERIES

log = logging.getLogger(__name__)

# Module-level BM25 cache — rebuilt only when the node set changes
_bm25_cache: dict[str, BM25Retriever] = {}


def clear_bm25_cache():
    """Clear BM25 cache so newly added nodes are indexed."""
    _bm25_cache.clear()
    log.info("Cleared BM25 cache.")


def _get_bm25_retriever(nodes, cache_key: str) -> BM25Retriever:
    if cache_key not in _bm25_cache:
        _bm25_cache[cache_key] = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=BM25_TOP_K,
        )
        log.info("Built BM25 index for key '%s' (%d nodes).", cache_key, len(nodes))
    return _bm25_cache[cache_key]


def create_hybrid_retriever(question: str, index, nodes, filters=None):
    """Build a QueryFusionRetriever (vector + BM25).

    Args:
        filters: Pre-computed MetadataFilters from the router.
    Returns:
        (hybrid_retriever, filters)
    """
    # ── Vector retriever ──
    retriever_kwargs: dict = {"similarity_top_k": VECTOR_TOP_K}
    if filters:
        retriever_kwargs["filters"] = filters
    vector_retriever = index.as_retriever(**retriever_kwargs)

    # ── BM25 (filtered by department if known) ──
    if filters:
        department = filters.filters[0].value
        filtered_nodes = [
            n for n in nodes
            if n.metadata.get("department") == department
        ]
        bm25_key = f"dept_{department}"
    else:
        filtered_nodes = nodes
        bm25_key = "all"

    keyword_retriever = _get_bm25_retriever(filtered_nodes, bm25_key)

    # ── Hybrid fusion ──
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, keyword_retriever],
        similarity_top_k=VECTOR_TOP_K,
        num_queries=FUSION_NUM_QUERIES,
        use_async=False,
    )

    return hybrid_retriever, filters
