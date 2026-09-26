# 🔍 Advanced LlamaIndex RAG + Agent

An enterprise-grade Retrieval-Augmented Generation (RAG) system with a
FastAPI REST API, multi-tool agent, hybrid retrieval, streaming, and
evaluation suite. Built with LlamaIndex, PostgreSQL/pgvector, and Ollama.

---

## Project Structure

```
LlamaIndex/
│
├── config/                     # Configuration
│   ├── __init__.py
│   └── settings.py             # All env vars, constants, LLM setup
│
├── database/                   # PostgreSQL layer
│   ├── __init__.py
│   ├── connection.py           # Connection pool (psycopg_pool)
│   ├── memory.py               # Chat history CRUD
│   └── notes.py                # Agent notes CRUD
│
├── ingestion/                  # Document ingestion
│   ├── __init__.py
│   ├── loader.py               # Load & chunk documents, assign departments
│   └── node_cache.py           # JSON node cache (BM25 reuse)
│
├── retrieval/                  # Retrieval layer
│   ├── __init__.py
│   ├── vector_store.py         # PGVector store + build/load index
│   ├── router.py               # Dynamic LLM router + metadata filter
│   └── hybrid.py               # Vector + BM25 QueryFusionRetriever
│
├── engine/                     # Query engines
│   ├── __init__.py
│   ├── postprocessors.py       # BGE Reranker + SimilarityFilter
│   ├── rag_engine.py           # RetrieverQueryEngine (streaming)
│   ├── sub_question.py         # SubQuestionQueryEngine
│   └── query_rewriter.py       # Coreference resolution
│
├── tools/                      # Agent tools (one file per tool)
│   ├── __init__.py             # ALL_TOOLS list
│   ├── rag_tool.py             # RAG document search
│   ├── web_tool.py             # DuckDuckGo web search
│   ├── calculator_tool.py      # Safe math evaluator
│   ├── datetime_tool.py        # Current date/time
│   ├── company_tool.py         # Working hours, holidays, contacts
│   ├── email_tool.py           # LLM-powered email drafter
│   ├── summarizer_tool.py      # Text summarizer
│   └── notes_tool.py           # PostgreSQL-backed notes
│
├── agent/                      # Agent definition
│   ├── __init__.py
│   └── agent.py                # FunctionAgent + system prompt
│
├── Data/                       # Source documents (PDF)
│   ├── HR_Leave_Policy.pdf
│   ├── IT_Security_Policy.pdf
│   └── Employee_Policy.pdf
│
├── NodeCache/                  # BM25 node cache (auto-generated)
│   └── nodes.json
│
├── models/                     # Local ML models
│   └── bge-reranker-base/      # BGE reranker (download once)
│
├── api/                        # FastAPI REST API
│   ├── app.py                  # App factory + lifespan startup
│   ├── schemas.py              # Pydantic request/response models
│   ├── state.py                # Shared app state (index, agent)
│   └── routers/
│       ├── rag_router.py       # /rag/query, /rag/query/stream, /rag/history
│       ├── agent_router.py     # /agent/chat, /agent/tools
│       └── notes_router.py     # /notes CRUD
│
├── run_rag.py                  # ▶ RAG chat loop entry point
├── run_agent.py                # ▶ Agent chat loop entry point
├── run_evaluate.py             # ▶ Evaluation suite entry point
├── run_api.py                  # ▶ FastAPI server
├── evaluation_dataset.json     # Ground-truth Q&A pairs
├── .env                        # Environment variables (git-ignored)
└── README.md
```

---

## Architecture

```
User Question
    │
    ▼
Query Rewriter ──── Chat History (PostgreSQL)
    │
    ▼
Dynamic LLM Router  ──────────────────────────────────────────┐
    │                                                          │
    ├── RAG ──────► Hybrid Retriever ──► BGE Reranker ──► Answer (streaming)
    │                │                                         │
    │                ├── pgvector (vector search)              │
    │                └── BM25 (keyword search, cached)         │
    │                                                          │
    ├── COMPLEX_RAG ► SubQuestionQueryEngine ─────────────► Answer
    │                                                          │
    └── DIRECT_LLM ──────────────────────────────────────► Answer (streaming)
                                                               │
                                                     Save to PostgreSQL
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
OLLAMA_API_KEY=your_key_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=katnoria_RAG
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

### 3. PostgreSQL setup

```sql
CREATE DATABASE "katnoria_RAG";
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Download BGE Reranker (once)

```python
from sentence_transformers import CrossEncoder
import os
os.makedirs("./models", exist_ok=True)
model = CrossEncoder("BAAI/bge-reranker-base")
model.save("./models/bge-reranker-base", safe_serialization=False)
```

### 5. First run — build the index

In `config/settings.py`, set:
```python
REBUILD_PGVECTOR = True
```
Run `python run_rag.py` once, then set it back to `False`.

---

## Usage

```bash
# 1. FastAPI REST Server (with Swagger UI at http://localhost:8000/docs)
python run_api.py

# 2. Interactive RAG Terminal Chat
python run_rag.py

# 3. Interactive Multi-Tool Agent
python run_agent.py

# 4. Evaluation Benchmark Suite
python run_evaluate.py
```

---

## FastAPI REST API Endpoints

Interactive Swagger documentation is available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check & model readiness |
| `POST` | `/rag/query` | Single-turn / multi-turn RAG query with source citations |
| `POST` | `/rag/query/stream` | Token-by-token streaming RAG response (Server-Sent Events) |
| `GET` | `/rag/history` | Retrieve conversation history for a session ID |
| `POST` | `/agent/chat` | Multi-tool agent execution (selects among 8 tools automatically) |
| `GET` | `/agent/tools` | Inspect available agent tools and descriptions |
| `POST` | `/notes/save` | Persist an agent note to PostgreSQL |
| `GET` | `/notes/list` | List all saved notes |
| `GET` | `/notes/search` | Search notes by title keyword |
| `DELETE` | `/notes/delete` | Delete a note by title |

---

## Agent Tools

| Tool | Description |
|------|-------------|
| `search_documents` | RAG search over company documents |
| `web_search` | Internet search via DuckDuckGo |
| `calculator` | Math expressions (`sqrt`, `log`, `factorial`, etc.) |
| `date_time` | Current date, time, day of week |
| `company_info` | Working hours, holidays, contact emails |
| `draft_email` | LLM-powered email drafting |
| `summarize_text` | Text summarization (concise/bullets/detailed) |
| `notes` | Save/list/search/delete notes in PostgreSQL |

---

## Key Features

- **Hybrid retrieval** — pgvector + BM25 with QueryFusion (4 query variants)
- **Dynamic LLM router** — single LLM call decides route + department filter
- **Streaming** — tokens appear in real-time for RAG and DIRECT_LLM routes
- **FastAPI REST API** — async REST service with SSE streaming and interactive Swagger UI
- **BGE reranker** — cross-encoder reranking for higher precision
- **PostgreSQL memory** — permanent chat history across sessions
- **Connection pooling** — psycopg_pool (min 2, max 10 connections)
- **Evaluation suite** — Hit Rate, MRR, Faithfulness, Relevancy, Correctness

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama Cloud — `gemma4:31b` |
| Embeddings | Local Ollama — `nomic-embed-text` (768-dim) |
| Vector Store | PostgreSQL + pgvector |
| Keyword Search | BM25 (bm25s) |
| Retrieval Fusion | LlamaIndex QueryFusionRetriever |
| Reranker | BGE reranker-base (CrossEncoder) |
| REST Framework | FastAPI + Uvicorn + Pydantic v2 |
| Memory | PostgreSQL (`chat_memory` table) |
| Agent | LlamaIndex FunctionAgent |
| Web Search | DuckDuckGo (no API key required) |

