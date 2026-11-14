"""
FastAPI Application Factory
============================

Creates and configures the FastAPI app with:
- Lifespan startup (LLM, index, agent initialisation)
- All routers mounted
- CORS middleware
- API key authentication
- Health endpoint
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import BASE_DIR
from api.state import state
from api.schemas import HealthResponse
from api.routers.rag_router import router as rag_router
from api.routers.agent_router import router as agent_router
from api.routers.notes_router import router as notes_router

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# API Key Authentication Middleware
# ──────────────────────────────────────────────────────────────

# Paths that don't require an API key
_PUBLIC_PATHS = {"/", "/chat", "/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid X-API-Key header (when API_KEY is set)."""

    async def dispatch(self, request: Request, call_next):
        api_key = os.getenv("API_KEY", "")

        # Skip auth if no key is configured, or if the path is public
        if not api_key or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        request_key = request.headers.get("X-API-Key", "")
        if request_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

        return await call_next(request)


# ──────────────────────────────────────────────────────────────
# Lifespan — runs once at startup and shutdown
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all heavy components once at startup."""
    log.info("Starting up — initialising RAG pipeline and agent...")

    from config.settings import configure_llm
    from database.memory import create_memory_table
    from retrieval.vector_store import build_or_load_index
    from engine.postprocessors import build_postprocessors
    from tools.rag_tool import build_rag_tool
    from tools import ALL_TOOLS
    from agent.agent import build_agent

    configure_llm()
    create_memory_table()

    state.index, state.nodes = build_or_load_index()
    state.reranker, state.sim_filter = build_postprocessors()

    # Inject RAG components into the RAG tool
    build_rag_tool(state.index, state.nodes, state.reranker, state.sim_filter)

    state.agent = build_agent(ALL_TOOLS)
    state.ready = True

    log.info("✅ API ready.")
    yield

    # Shutdown
    log.info("Shutting down...")
    state.ready = False


# ──────────────────────────────────────────────────────────────
# App Factory
# ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="MultiAgentRagSystemWithLlamaindex API",
        description=(
            "Enterprise Multi-Agent RAG system with hybrid retrieval, dynamic routing, "
            "streaming, and a multi-tool agent."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── API Key Auth ──
    app.add_middleware(APIKeyMiddleware)

    # ── CORS ──
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Check if the API is up and the pipeline is ready."""
        from llama_index.core import Settings
        from config.settings import GROQ_MODEL
        return HealthResponse(
            status="ready" if state.ready else "initialising",
            version="1.0.0",
            llm_model=f"{GROQ_MODEL} (Groq)",
            embed_model="BAAI/bge-base-en-v1.5",
        )

    @app.get("/chat", tags=["UI"])
    async def chat_ui():
        index_file = os.path.join(BASE_DIR, "static", "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file, media_type="text/html")
        return {
            "message": "Chat UI index.html not found.",
            "docs": "/docs",
            "health": "/health",
        }

    # ── Routers ──
    app.include_router(rag_router)
    app.include_router(agent_router)
    app.include_router(notes_router)

    return app
