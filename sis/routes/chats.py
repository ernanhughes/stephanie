# stephanie/routes/chats.py
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

router = APIRouter()

@router.get("/chats", response_class=HTMLResponse)
def list_chats(
    request: Request,
    provider: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """
    List recent chat conversations, optionally filtered by provider.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates


    conversations = memory.chats.get_all(1000)

    return templates.TemplateResponse(
        "/chats/list.html",
        {
            "request": request,
            "chats": [c.to_dict() for c in conversations],
            "filters": {"provider": provider, "limit": limit},
        },
    )


@router.get("/chats/{chat_id}", response_class=HTMLResponse)
def chat_detail(request: Request, chat_id: int):
    """
    Detailed chat view: conversation metadata + ordered messages + turns.
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    conv = memory.chats.get_conversation(chat_id)

    return templates.TemplateResponse(
        "/chats/detail.html",
        {
            "request": request,
            "conversation": conv.to_dict(include_messages=True, include_turns=True),
        },
    )
