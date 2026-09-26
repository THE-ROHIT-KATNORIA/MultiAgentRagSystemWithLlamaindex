"""
Agent Router
============

POST /agent/chat  — send a message to the agent
"""

import logging
from fastapi import APIRouter, HTTPException

from api.schemas import AgentRequest, AgentResponse
from api.state import state

log = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])


def _check_ready():
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service is still initialising.")


# ──────────────────────────────────────────────────────────────
# POST /agent/chat
# ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=AgentResponse, summary="Chat with the AI agent")
async def agent_chat(body: AgentRequest):
    """
    Send a message to the FunctionAgent. It will automatically
    pick the right tool (RAG, web search, calculator, email, etc.)
    and return the result.
    """
    _check_ready()
    try:
        response = await state.agent.run(body.question)
        return AgentResponse(question=body.question, answer=str(response))
    except Exception as exc:
        log.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────
# GET /agent/tools
# ──────────────────────────────────────────────────────────────

@router.get("/tools", summary="List all available agent tools")
async def list_tools():
    """Return all tools available to the agent."""
    _check_ready()
    tools = []
    for tool in state.agent._tools if hasattr(state.agent, "_tools") else []:
        tools.append({
            "name": tool.metadata.name,
            "description": tool.metadata.description,
        })
    return {"tools": tools}
