# ──────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for MultiAgentRagSystem
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system dependencies ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──
COPY . .

# ── Pre-cache BGE reranker & embedding models at build time ──
RUN python -c "\
import os; \
from sentence_transformers import CrossEncoder; \
from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
path = './models/bge-reranker-base'; \
os.makedirs('./models', exist_ok=True); \
(CrossEncoder('BAAI/bge-reranker-base').save(path) if not os.path.exists(path) else None); \
HuggingFaceEmbedding(model_name='BAAI/bge-base-en-v1.5'); \
print('Models cached successfully')"

# ── Expose port and start ──
EXPOSE 8000

CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8000"]

