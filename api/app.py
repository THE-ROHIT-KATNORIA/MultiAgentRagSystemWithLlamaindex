"""
FastAPI Application Factory
============================

Creates and configures the FastAPI app with:
- Lifespan startup (LLM, index, agent initialisation)
- All routers mounted
- CORS middleware
- Health endpoint
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.state import state
from api.schemas import HealthResponse
from api.routers.rag_router import router as rag_router
from api.routers.agent_router import router as agent_router
from api.routers.notes_router import router as notes_router

log = logging.getLogger(__name__)


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

    # ── CORS ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],       # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ──
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """Check if the API is up and the pipeline is ready."""
        from llama_index.core import Settings
        return HealthResponse(
            status="ready" if state.ready else "initialising",
            version="1.0.0",
            llm_model="gemma4:31b",
            embed_model="nomic-embed-text:latest",
        )

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "message": "LlamaIndex RAG API is running.",
            "docs": "/docs",
            "health": "/health",
        }

    # ── Routers ──
    app.include_router(rag_router)
    app.include_router(agent_router)
    app.include_router(notes_router)

    return app
