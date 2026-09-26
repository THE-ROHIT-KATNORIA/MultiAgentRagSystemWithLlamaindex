"""
Query Rewriter
==============

Rewrites the user's question into a standalone search query
by resolving coreferences using conversation history.
"""

import logging
from llama_index.core import Settings

log = logging.getLogger(__name__)


def rewrite_query(question: str, history: str) -> str:
    """Rewrite a question into a standalone query using chat history."""
    prompt = f"""You are a query rewriting system.

Rewrite the CURRENT user question into a standalone search query.
Use conversation history ONLY to resolve coreferences like:
it, they, them, this, that, those, previous one, how many, who is eligible.

Rules:
- Keep the original meaning.
- Make the query self-contained.
- Keep important keywords.
- Do not answer the question.
- Do not invent information.
- Return ONLY the rewritten query.

Conversation history:
{history}

Current question:
{question}

Rewritten query:
"""
    try:
        return Settings.llm.complete(prompt).text.strip()
    except Exception as exc:
        log.error("Query rewrite failed: %s — using original.", exc)
        return question
