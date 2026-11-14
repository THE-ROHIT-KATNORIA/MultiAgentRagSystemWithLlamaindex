"""
RAG Router
==========

POST /rag/query   — query the RAG pipeline
GET  /rag/history — fetch chat history for a session
"""

import logging
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from api.schemas import RAGRequest, RAGResponse, SourceNode, HistoryRequest, HistoryResponse
from api.state import state

from database.memory import get_chat_history, save_message
from retrieval.router import dynamic_route_and_filter
from engine.rag_engine import create_rag_engine
from engine.sub_question import create_sub_question_engine
from engine.query_rewriter import rewrite_query
from config.settings import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.loader import assign_department
from ingestion.node_cache import save_nodes
from retrieval.hybrid import clear_bm25_cache
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

log = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


def _check_ready():
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service is still initialising. Try again shortly.")


# ──────────────────────────────────────────────────────────────
# POST /rag/query
# ──────────────────────────────────────────────────────────────

@router.post("/query", response_model=RAGResponse, summary="Query the RAG pipeline")
async def rag_query(body: RAGRequest):
    """
    Ask a question. The pipeline will:
    - Rewrite the query using chat history
    - Route to RAG / COMPLEX_RAG / DIRECT_LLM
    - Retrieve, rerank, and synthesise an answer
    - Save the exchange to PostgreSQL
    """
    _check_ready()

    try:
        history = get_chat_history(body.session_id, limit=20)
        rewritten = rewrite_query(body.question, history)

        decision = dynamic_route_and_filter(rewritten)
        route = decision["route"]
        filters = decision["filters"]

        sources: list[SourceNode] = []
        answer = ""

        if route == "RAG":
            engine, used_filters = create_rag_engine(
                rewritten, state.index, state.nodes,
                state.reranker, state.sim_filter,
                filters=filters, streaming=False,
            )
            response = engine.query(rewritten)
            answer = str(response).strip()

            # Automatic fallback: if filtered search produced Empty Response, retry without filters
            if (not answer or answer == "Empty Response") and used_filters is not None:
                log.info("Filtered RAG returned empty; falling back to unfiltered retrieval...")
                fallback_engine, _ = create_rag_engine(
                    rewritten, state.index, state.nodes,
                    state.reranker, state.sim_filter,
                    filters=None, streaming=False,
                )
                response = fallback_engine.query(rewritten)
                answer = str(response).strip()

            if hasattr(response, "source_nodes"):
                for s in response.source_nodes:
                    meta = s.node.metadata
                    sources.append(SourceNode(
                        file_name=meta.get("file_name", "Unknown"),
                        page=str(meta.get("page_label", meta.get("page_number", "?"))),
                        score=s.score,
                        text=s.node.get_content()[:300],
                    ))

            if not answer or answer == "Empty Response":
                answer = "I could not find information answering this question in the uploaded documents."

        elif route == "COMPLEX_RAG":
            engine, used_filters = create_sub_question_engine(
                rewritten, state.index, state.nodes,
                state.reranker, state.sim_filter,
                filters=filters,
            )
            response = engine.query(rewritten)
            answer = str(response).strip()

            # Automatic fallback: if filtered search produced Empty Response, retry without filters
            if (not answer or answer == "Empty Response") and used_filters is not None:
                log.info("Filtered COMPLEX_RAG returned empty; falling back to unfiltered retrieval...")
                fallback_engine, _ = create_sub_question_engine(
                    rewritten, state.index, state.nodes,
                    state.reranker, state.sim_filter,
                    filters=None,
                )
                response = fallback_engine.query(rewritten)
                answer = str(response).strip()

            if hasattr(response, "source_nodes"):
                for s in response.source_nodes:
                    meta = s.node.metadata
                    sources.append(SourceNode(
                        file_name=meta.get("file_name", "Unknown"),
                        page=str(meta.get("page_label", meta.get("page_number", "?"))),
                        score=s.score,
                        text=s.node.get_content()[:300],
                    ))

            if not answer or answer == "Empty Response":
                answer = "I could not find information answering this question in the uploaded documents."

        else:  # DIRECT_LLM
            prompt = (
                f"Answer the user's question directly. "
                f"Do not mention company documents.\n\nQuestion: {body.question}"
            )
            answer = Settings.llm.complete(prompt).text.strip()

        save_message(body.session_id, "user", body.question)
        save_message(body.session_id, "assistant", answer)

        return RAGResponse(
            question=body.question,
            rewritten_query=rewritten,
            route=route,
            department=decision["department"],
            confidence=decision["confidence"],
            reasoning=decision["reasoning"],
            answer=answer,
            sources=sources,
            session_id=body.session_id,
        )

    except Exception as exc:
        log.error("RAG query error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────
# POST /rag/query/stream
# ──────────────────────────────────────────────────────────────

@router.post("/query/stream", summary="Stream RAG answer token by token")
async def rag_query_stream(body: RAGRequest):
    """
    Same as /rag/query but streams the answer as Server-Sent Events (SSE).
    Each event is a text chunk; final event is `data: [DONE]`.
    """
    _check_ready()

    async def token_generator():
        try:
            history = get_chat_history(body.session_id, limit=20)
            rewritten = rewrite_query(body.question, history)
            decision = dynamic_route_and_filter(rewritten)
            route = decision["route"]
            filters = decision["filters"]

            full_text = ""

            if route in ("RAG", "COMPLEX_RAG"):
                engine, _ = create_rag_engine(
                    rewritten, state.index, state.nodes,
                    state.reranker, state.sim_filter,
                    filters=filters, streaming=True,
                )
                streaming_resp = engine.query(rewritten)
                for token in streaming_resp.response_gen:
                    full_text += token
                    yield f"data: {token}\n\n"
            else:
                prompt = (
                    f"Answer the user's question directly. "
                    f"Do not mention company documents.\n\nQuestion: {body.question}"
                )
                for token in Settings.llm.stream_complete(prompt):
                    full_text += token.delta
                    yield f"data: {token.delta}\n\n"

            save_message(body.session_id, "user", body.question)
            save_message(body.session_id, "assistant", full_text)
            yield "data: [DONE]\n\n"

        except Exception as exc:
            log.error("Stream error: %s", exc)
            yield f"data: ERROR: {exc}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────
# GET /rag/history
# ──────────────────────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse, summary="Get chat history")
async def get_history(session_id: str, limit: int = 20):
    """Retrieve conversation history for a session."""
    _check_ready()
    try:
        history = get_chat_history(session_id=session_id, limit=limit)
        return HistoryResponse(session_id=session_id, history=history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────
# POST /rag/upload
# ──────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload a document into the knowledge base")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or Markdown document, chunk, embed, and store in pgvector."""
    _check_ready()
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest and chunk
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        new_nodes = splitter.get_nodes_from_documents(docs)

        dept = assign_department(file.filename)
        for node in new_nodes:
            node.metadata["file_name"] = file.filename
            node.metadata["department"] = dept

        # Insert directly into pgvector (Neon)
        state.index.insert_nodes(new_nodes)

        # Update in-memory node list and BM25 cache
        if state.nodes is not None:
            state.nodes.extend(new_nodes)
            save_nodes(state.nodes)
        clear_bm25_cache()

        return {
            "status": "success",
            "file_name": file.filename,
            "department": dept,
            "chunks_indexed": len(new_nodes),
            "message": f"Successfully indexed '{file.filename}' into knowledge base."
        }
    except Exception as exc:
        log.error("Document upload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}")
