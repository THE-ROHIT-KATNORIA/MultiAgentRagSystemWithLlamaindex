"""
Application Configuration
=========================

All environment variables, constants, and LLM/embedding setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Environment Variables
# ──────────────────────────────────────────────────────────────

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY is missing from your .env file.")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "katnoria_RAG")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not POSTGRES_PASSWORD:
    import getpass
    POSTGRES_PASSWORD = getpass.getpass("Enter PostgreSQL password: ")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

DATA_DIR = "./Data"
NODE_CACHE_DIR = "./NodeCache"
NODE_CACHE_FILE = os.path.join(NODE_CACHE_DIR, "nodes.json")
RERANKER_PATH = "./models/bge-reranker-base"

os.makedirs(NODE_CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Vector Store
# ──────────────────────────────────────────────────────────────

EMBED_DIM = 768
VECTOR_TABLE_NAME = "rag_documents"
ACTUAL_VECTOR_TABLE = f"data_{VECTOR_TABLE_NAME}"

# Set True to completely rebuild the pgvector index (run once, then set back)
REBUILD_PGVECTOR = False

# ──────────────────────────────────────────────────────────────
# RAG Parameters
# ──────────────────────────────────────────────────────────────

ALLOWED_DEPARTMENTS = {"HR", "IT", "FINANCE"}
MAX_INPUT_LENGTH = 2000
MAX_HISTORY_CHARS = 3000
RETRIEVAL_TOP_K = 10
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
SIMILARITY_CUTOFF = 0.3
RERANKER_TOP_N = 3
BM25_TOP_K = 10
VECTOR_TOP_K = 10
FUSION_NUM_QUERIES = 4

# ──────────────────────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────────────────────


def configure_llm():
    """Configure the global LLM and embedding model."""
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.ollama import OllamaEmbedding

    Settings.llm = Ollama(
        model="gemma4:31b",
        base_url="https://ollama.com",
        request_timeout=120.0,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
    )

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest",
        base_url="http://127.0.0.1:11434",
    )

    return Settings.llm, Settings.embed_model
