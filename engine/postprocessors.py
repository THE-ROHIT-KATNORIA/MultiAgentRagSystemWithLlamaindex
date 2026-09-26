"""
Post-processors
===============

BGE Reranker + Similarity Filter setup.
"""

import os
import logging
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
from config.settings import RERANKER_PATH, SIMILARITY_CUTOFF, RERANKER_TOP_N

log = logging.getLogger(__name__)


def build_postprocessors():
    """Return (reranker | None, similarity_filter)."""
    reranker = None
    if os.path.exists(RERANKER_PATH):
        reranker = SentenceTransformerRerank(model=RERANKER_PATH, top_n=RERANKER_TOP_N)
        log.info("Reranker loaded: %s", RERANKER_PATH)
    else:
        log.warning("Reranker not found at %s — skipping reranking.", RERANKER_PATH)

    similarity_filter = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
    return reranker, similarity_filter
