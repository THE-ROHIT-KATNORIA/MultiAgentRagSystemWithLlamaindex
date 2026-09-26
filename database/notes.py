"""
Notes — PostgreSQL
==================

Agent note-taking with persistent storage.
"""

import logging
from database.connection import get_db_connection

log = logging.getLogger(__name__)


def ensure_notes_table():
    """Create the agent_notes table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_notes (
                        id          BIGSERIAL    PRIMARY KEY,
                        title       TEXT         NOT NULL,
                        content     TEXT         NOT NULL,
                        created_at  TIMESTAMPTZ  DEFAULT NOW()
                    );
                """)
            conn.commit()
    except Exception as exc:
        log.error("Failed to create notes table: %s", exc)


def manage_notes(action: str, title: str = "", content: str = "") -> str:
    """Save, list, search, or delete personal notes.

    Args:
        action: 'save', 'list', 'search', or 'delete'
        title: Note title (required for save/search/delete)
        content: Note content (required for save)
    """
    ensure_notes_table()
    action = action.lower().strip()

    try:
        if action == "save":
            if not title or not content:
                return "Error: both 'title' and 'content' are required."
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO agent_notes (title, content) VALUES (%s, %s)",
                        (title, content),
                    )
                conn.commit()
            return f"✅ Note saved: '{title}'"

        elif action == "list":
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT title, content, created_at "
                        "FROM agent_notes ORDER BY created_at DESC LIMIT 20"
                    )
                    rows = cur.fetchall()
            if not rows:
                return "No notes found."
            output = []
            for t, c, ts in rows:
                date_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                output.append(f"📝 {t} ({date_str})\n   {c[:100]}...")
            return "\n\n".join(output)

        elif action == "search":
            if not title:
                return "Error: provide a 'title' keyword to search."
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT title, content, created_at "
                        "FROM agent_notes WHERE title ILIKE %s "
                        "ORDER BY created_at DESC LIMIT 10",
                        (f"%{title}%",),
                    )
                    rows = cur.fetchall()
            if not rows:
                return f"No notes matching '{title}'."
            output = []
            for t, c, ts in rows:
                date_str = ts.strftime("%Y-%m-%d %H:%M") if ts else ""
                output.append(f"📝 {t} ({date_str})\n   {c}")
            return "\n\n".join(output)

        elif action == "delete":
            if not title:
                return "Error: provide the 'title' to delete."
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM agent_notes WHERE title = %s", (title,))
                    deleted = cur.rowcount
                conn.commit()
            return f"🗑️ Deleted note: '{title}'" if deleted else f"No note '{title}' found."

        else:
            return "Unknown action. Use: 'save', 'list', 'search', or 'delete'."

    except Exception as exc:
        return f"Notes operation failed: {exc}"
