"""
Pydantic Request / Response Models
===================================
"""

from pydantic import BaseModel, Field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# RAG
# ──────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The user's question")
    session_id: str = Field(default="default",
                            description="Conversation session ID")

class SourceNode(BaseModel):
    file_name: str
    page: str
    score: Optional[float]
    text: str

class RAGResponse(BaseModel):
    question: str
    rewritten_query: str
    route: str
    department: str
    confidence: float
    reasoning: str
    answer: str
    sources: list[SourceNode]
    session_id: str


# ──────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Question for the agent")

class AgentResponse(BaseModel):
    question: str
    answer: str


# ──────────────────────────────────────────────────────────────
# Chat History
# ──────────────────────────────────────────────────────────────

class HistoryRequest(BaseModel):
    session_id: str
    limit: int = Field(default=20, ge=1, le=100)

class HistoryResponse(BaseModel):
    session_id: str
    history: str


# ──────────────────────────────────────────────────────────────
# Notes
# ──────────────────────────────────────────────────────────────

class NoteRequest(BaseModel):
    action: str = Field(..., description="save | list | search | delete")
    title: str = Field(default="")
    content: str = Field(default="")

class NoteResponse(BaseModel):
    result: str


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_model: str
    embed_model: str
