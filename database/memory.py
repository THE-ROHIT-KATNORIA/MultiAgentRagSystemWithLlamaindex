"""
Chat Memory — PostgreSQL
========================

CRUD operations for the permanent chat_memory table.
"""

import logging
from database.connection import get_db_connection
from config.settings import MAX_HISTORY_CHARS

log = logging.getLogger(__name__)


def create_memory_table():
    """Ensure the chat_memory table exists."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_memory (
                        id           BIGSERIAL    PRIMARY KEY,
                        session_id   TEXT         NOT NULL,
                        message_role TEXT         NOT NULL,
                        message      TEXT         NOT NULL,
                        created_at   TIMESTAMPTZ  DEFAULT NOW()
                    );
                """)
            conn.commit()
        log.info("chat_memory table ready.")
    except Exception as exc:
        log.error("Failed to create memory table: %s", exc)
        raise


def save_message(session_id: str, role: str, message: str):
    """Persist one chat message."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_memory (session_id, message_role, message) "
                    "VALUES (%s, %s, %s)",
                    (session_id, role, message),
                )
            conn.commit()
    except Exception as exc:
        log.error("Failed to save message: %s", exc)


def get_chat_history(session_id: str, limit: int = 20) -> str:
    """Load recent conversation, capped to MAX_HISTORY_CHARS."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT message_role, message, created_at "
                    "FROM chat_memory "
                    "WHERE session_id = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (session_id, limit),
                )
                rows = cur.fetchall()

        rows.reverse()
        if not rows:
            return "No previous conversation."

        lines: list[str] = []
        total = 0
        for role, message, _ in rows:
            line = f"{role}: {message}"
            if total + len(line) > MAX_HISTORY_CHARS:
                lines.append("[... earlier messages truncated ...]")
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)
    except Exception as exc:
        log.error("Failed to load chat history: %s", exc)
        return "No previous conversation."
