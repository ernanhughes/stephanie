# stephanie/routes/chats.py
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Request, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

@router.get("/chats", response_class=HTMLResponse)
def list_chats(
    request: Request,
    provider: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    memory = request.app.state.memory
    templates = request.app.state.templates

    rows = memory.chats.list_by_turn_count(limit=limit, provider=provider)
    # Shape for template: list of dicts with conversation dict + turn_count
    chats = [
        {
            "turn_count": turn_count,
            "conversation": conv.to_dict(),  # or include_messages=False if heavy
        }
        for (conv, turn_count) in rows
    ]

    return templates.TemplateResponse(
        "/chats/list.html",
        {
            "request": request,
            "chats": chats,
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


@router.get("/chats/{chat_id}/score", response_class=HTMLResponse)
def chat_score_view(
    request: Request,
    chat_id: int,
    only_unrated: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=10, le=200),
):
    memory = request.app.state.memory
    templates = request.app.state.templates

    conv = memory.chats.get_conversation(chat_id).to_dict()
    offset = (page - 1) * page_size
    turns, rated, total = memory.chats.scoring_batch(
        chat_id, only_unrated=only_unrated, limit=page_size, offset=offset
    )
    next_page = page + 1 if len(turns) == page_size else None
    prev_page = page - 1 if page > 1 else None

    return templates.TemplateResponse(
        "/chats/score.html",
        {
            "request": request,
            "conversation": conv,   # dict
            "turns": turns,         # list[dict]
            "only_unrated": only_unrated,
            "progress": {"rated": rated, "total": total},
            "page": page, "page_size": page_size,
            "next_page": next_page, "prev_page": prev_page,
        },
    )

# NEW: set star for a turn
@router.post("/chats/{chat_id}/turns/{turn_id}/star")
def set_turn_star(request: Request, chat_id: int, turn_id: int, star: int = Form(...)):
    memory = request.app.state.memory
    memory.chats.set_turn_star(turn_id, star)
    # Return to scoring page, preserving filter if present
    referer = request.headers.get("referer") or f"/chats/{chat_id}/score"
    return RedirectResponse(url=referer, status_code=303)
