# ──────────────────────────────────────────────────────────────
# Dockerfile for Hugging Face Spaces (16GB RAM) & Cloud Deploy
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

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

# ── Create non-root user for Hugging Face Spaces security & permissions ──
RUN useradd -m -u 1000 user && \
    mkdir -p /app/NodeCache /app/Data /home/user/.cache && \
    chown -R user:user /app /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["sh", "-c", "uvicorn run_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
