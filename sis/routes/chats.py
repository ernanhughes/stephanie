"""
Chat Routes Module

This module provides FastAPI endpoints for managing and interacting with chat conversations
in the Stephanie system. It includes routes for listing conversations, viewing details,
scoring turns, and tracking annotation progress.

Key Features:
- List conversations with fil Grace tering and pagination
- Detailed conversation view with messages and turns
- Turn scoring interface with progress tracking
- Star rating system for turns
- Annotation progress monitoring

All routes are integrated with the template system for server-side rendering
and maintain session state through the application's memory store.
"""

from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Request, Query, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

# Create API router for chat-related endpoints
router = APIRouter()

@router.get("/chats", response_class=HTMLResponse)
def list_chats(
    request: Request,
    provider: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
):
    """
    Display a paginated list of chat conversations with filtering options.
    
    Args:
        request: FastAPI Request object for accessing app state
        provider: Optional filter by chat provider (e.g., 'openai')
        limit: Maximum number of conversations to display (1-1000)
        
    Returns:
        Rendered HTML template with filtered chat list
    """
    # Access application state (memory and templates)
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Retrieve conversations sorted by turn count with optional provider filter
    rows = memory.chats.list_by_turn_count(limit=limit, provider=provider)
    
    # Prepare data for template rendering
    chats = [
        {
            "turn_count": turn_count,
            "conversation": conv.to_dict(),  # Serialize conversation data
        }
        for (conv, turn_count) in rows
    ]

    # Render the chat list template
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
    Display detailed view of a specific conversation.
    
    Args:
        request: FastAPI Request object
        chat_id: ID of the conversation to display
        
    Returns:
        Rendered HTML template with conversation details or 404 error
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Retrieve complete conversation data with messages and turns
    convo = memory.chats.get_conversation_dict(
        chat_id,
        include_messages=True,
        include_turns=True,
        include_turn_message_texts=True,  # Flatten message texts for easy template access
    )
    
    # Handle case where conversation doesn't exist
    if not convo:
        return templates.TemplateResponse(
            "/chats/detail.html",
            {"request": request, "conversation": None},
            status_code=404,
        )

    # Render conversation detail template
    return templates.TemplateResponse(
        "/chats/detail.html",
        {"request": request, "conversation": convo},
    )


@router.get("/chats/{chat_id}/score", response_class=HTMLResponse)
def chat_score_view(
    request: Request,
    chat_id: int,
    only_unrated: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=10, le=200),
):
    """
    Display the scoring interface for a conversation's turns.
    
    Args:
        request: FastAPI Request object
        chat_id: ID of the conversation to score
        only_unrated: Filter to show only unrated turns (star = 0)
        page: Current page for pagination (1-based)
        page_size: Number of turns to display per page
        
    Returns:
        Rendered HTML template with scoring interface
    """
    memory = request.app.state.memory
    templates = request.app.state.templates

    # Get basic conversation info (without full message/turn data)
    conversation = memory.chats.get_conversation_dict(
        chat_id,
        include_messages=False,
        include_turns=False,
    )

    # Calculate pagination offset
    offset = (page - 1) * page_size
    
    # Retrieve a batch of turns for scoring
    turns, rated, total = memory.chats.scoring_batch(
        chat_id, only_unrated=only_unrated, limit=page_size, offset=offset
    )
    
    # Calculate pagination controls
    next_page = page + 1 if len(turns) == page_size else None
    prev_page = page - 1 if page > 1 else None

    # Render scoring template
    return templates.TemplateResponse(
        "/chats/score.html",
        {
            "request": request,
            "conversation": conversation,
            "turns": turns,
            "only_unrated": only_unrated,
            "progress": {"rated": rated, "total": total},
            "page": page, "page_size": page_size,
            "next_page": next_page, "prev_page": prev_page,
        },
    )


@router.post("/chats/{chat_id}/turns/{turn_id}/star")
def set_turn_star(request: Request, chat_id: int, turn_id: int, star: int = Form(...)):
    """
    Set a star rating for a specific turn.
    
    Args:
        request: FastAPI Request object
        chat_id: ID of the conversation containing the turn
        turn_id: ID of the turn to rate
        star: Star rating value (-5 to 5)
        
    Returns:
        Redirect back to the scoring page or referer URL
    """
    memory = request.app.state.memory
    
    # Update the turn's star rating
    memory.chats.set_turn_star(turn_id, star)
    
    # Redirect back to the scoring page, preserving any filters
    referer = request.headers.get("referer") or f"/chats/{chat_id}/score"
    return RedirectResponse(url=referer, status_code=303)


@router.get("/chats/{chat_id}/annotation/progress", response_class=JSONResponse)
def progress(request: Request, chat_id: int):
    """
    Get annotation progress statistics for a conversation.
    
    Args:
        request: FastAPI Request object
        chat_id: ID of the conversation to check
        
    Returns:
        JSON response with domain and NER annotation progress
    """
    memory = request.app.state.memory
    
    # Get annotation progress for both domains and NER
    dom = memory.chats.annotation_progress(chat_id, kind="domains")
    ner = memory.chats.annotation_progress(chat_id, kind="ner")
    
    # Return progress data as JSON
    return {"chat_id": chat_id, "domains": dom, "ner": ner}