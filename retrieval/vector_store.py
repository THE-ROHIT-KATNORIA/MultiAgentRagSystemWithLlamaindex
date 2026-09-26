"""
Vector Store & Index
====================

PGVector store setup, table helpers, and index build/load.
"""

import logging
from psycopg import sql

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore

from config.settings import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD,
    VECTOR_TABLE_NAME, ACTUAL_VECTOR_TABLE,
    EMBED_DIM, REBUILD_PGVECTOR,
)
from database.connection import get_db_connection
from ingestion.loader import load_and_chunk_documents
from ingestion.node_cache import save_nodes, load_cached_nodes

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# PGVector Store Factory
# ──────────────────────────────────────────────────────────────

def make_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        table_name=VECTOR_TABLE_NAME,
        embed_dim=EMBED_DIM,
        use_jsonb=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        indexed_metadata_keys={("department", "text")},
    )


# ──────────────────────────────────────────────────────────────
# Table Helpers
# ──────────────────────────────────────────────────────────────

def vector_table_exists() -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables"
                "  WHERE table_schema='public' AND table_name=%s"
                ");",
                (ACTUAL_VECTOR_TABLE,),
            )
            return cur.fetchone()[0]


def vector_row_count() -> int:
    if not vector_table_exists():
        return 0
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM public.{}").format(
                    sql.Identifier(ACTUAL_VECTOR_TABLE)
                )
            )
            return cur.fetchone()[0]


def drop_vector_table() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(
                    sql.Identifier(ACTUAL_VECTOR_TABLE)
                )
            )
        conn.commit()


# ──────────────────────────────────────────────────────────────
# Build or Load Index
# ──────────────────────────────────────────────────────────────

def build_or_load_index():
    """Return (VectorStoreIndex, nodes).

    Loads from pgvector when data already exists, otherwise
    ingests documents, embeds, and persists.
    """
    import os
    from config.settings import NODE_CACHE_FILE

    vector_store = make_vector_store()
    existing_rows = vector_row_count()
    log.info("Existing pgvector rows: %d", existing_rows)

    if REBUILD_PGVECTOR and existing_rows > 0:
        log.info("Rebuilding pgvector index...")
        drop_vector_table()
        for f in (NODE_CACHE_FILE, NODE_CACHE_FILE.replace(".json", ".pkl")):
            if os.path.exists(f):
                os.remove(f)
        vector_store = make_vector_store()
        existing_rows = 0

    if existing_rows > 0:
        log.info("Loading existing index from PostgreSQL + pgvector...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        nodes = load_cached_nodes()
        if nodes is None:
            raise RuntimeError(
                "pgvector data exists but node cache is missing. "
                "Set REBUILD_PGVECTOR=True once to rebuild."
            )
        log.info("Loaded index with %d cached nodes.", len(nodes))
    else:
        log.info("No pgvector data found — creating new index...")
        nodes = load_and_chunk_documents()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
        save_nodes(nodes)
        log.info("New pgvector index created.")

    log.info("Final pgvector row count: %d", vector_row_count())
    return index, nodes
