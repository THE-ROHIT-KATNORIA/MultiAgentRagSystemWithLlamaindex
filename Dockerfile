# ──────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for MultiAgentRagSystem
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable memory tuning
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MALLOC_TRIM_THRESHOLD_=100000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

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

# ── Pre-cache BGE embedding model only (no heavy CrossEncoder) ──
RUN python -c "\
from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
HuggingFaceEmbedding(model_name='BAAI/bge-base-en-v1.5'); \
print('Embedding model cached successfully')"

# ── Expose port and start ──
EXPOSE 8000

CMD ["sh", "-c", "uvicorn run_api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

