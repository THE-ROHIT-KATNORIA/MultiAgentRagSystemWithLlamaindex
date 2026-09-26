"""
Tool: Notes (PostgreSQL)
========================
"""

from llama_index.core.tools import FunctionTool
from database.notes import manage_notes as _manage_notes


def notes(action: str, title: str = "", content: str = "") -> str:
    """Save, list, search, or delete personal notes.

    Args:
        action: 'save', 'list', 'search', or 'delete'
        title: Note title (required for save/search/delete)
        content: Note body (required for save)
    """
    return _manage_notes(action=action, title=title, content=content)


notes_tool = FunctionTool.from_defaults(
    fn=notes,
    name="notes",
    description=(
        "Save, list, search, or delete personal notes stored in the database. "
        "Actions: 'save' (needs title + content), 'list', "
        "'search' (needs title keyword), 'delete' (needs title). "
        "Use when the user wants to remember, record, or retrieve information."
    ),
)
