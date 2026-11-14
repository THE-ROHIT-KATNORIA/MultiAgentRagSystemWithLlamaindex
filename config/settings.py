"""
Application Configuration
=========================

All environment variables, constants, and LLM/embedding setup.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Project Root (absolute base for all relative paths)
# ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# Environment Variables
# ──────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "katnoria_RAG")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "require" if "neon.tech" in os.getenv("POSTGRES_HOST", "") else "prefer")

API_KEY = os.getenv("API_KEY", "")

if not POSTGRES_PASSWORD:
    raise ValueError("POSTGRES_PASSWORD is missing from your .env file.")

# ──────────────────────────────────────────────────────────────
# Paths (absolute, based on project root)
# ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "Data")
NODE_CACHE_DIR = os.path.join(BASE_DIR, "NodeCache")
NODE_CACHE_FILE = os.path.join(NODE_CACHE_DIR, "nodes.json")
RERANKER_PATH = os.path.join(BASE_DIR, "models", "bge-reranker-base")

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
# Default to false for cloud environments with 512MB RAM (set to true on plans with >=1GB RAM)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() in ("true", "1", "yes")
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
    import torch
    torch.set_num_threads(1)

    from llama_index.core import Settings
    from llama_index.llms.groq import Groq
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.llm = Groq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
    )

    return Settings.llm, Settings.embed_model
