"""
Notes Router
============

CRUD endpoints for the agent's PostgreSQL notes.

POST   /notes/save    — save a note
GET    /notes/list    — list all notes
GET    /notes/search  — search by keyword
DELETE /notes/delete  — delete by title
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from api.schemas import NoteRequest, NoteResponse
from database.notes import manage_notes

log = logging.getLogger(__name__)
router = APIRouter(prefix="/notes", tags=["Notes"])


@router.post("/save", response_model=NoteResponse, summary="Save a note")
async def save_note(body: NoteRequest):
    try:
        result = manage_notes(action="save", title=body.title, content=body.content)
        return NoteResponse(result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/list", response_model=NoteResponse, summary="List all notes")
async def list_notes():
    try:
        result = manage_notes(action="list")
        return NoteResponse(result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/search", response_model=NoteResponse, summary="Search notes by title keyword")
async def search_notes(keyword: str = Query(..., description="Title keyword to search")):
    try:
        result = manage_notes(action="search", title=keyword)
        return NoteResponse(result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/delete", response_model=NoteResponse, summary="Delete a note by title")
async def delete_note(title: str = Query(..., description="Exact title of the note to delete")):
    try:
        result = manage_notes(action="delete", title=title)
        return NoteResponse(result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
