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
import asyncio
from typing import Optional
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Query, Header, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from sis.services.chat_context import build_chat_messages as _build_chat_messages
from sis.services.chat_execution import (
    ChatInferenceFailed,
    ChatServiceUnavailable,
    ConversationBusy,
    ConversationNotFound,
    IdempotencyConflict,
)
from sis.services.worker_execution import WorkerExecutionError
from stephanie.services.execution_experience import ExperienceError

# Create API router for chat-related endpoints
router = APIRouter()


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=200)


class ChatMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


class WorkerExecutionRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=12000)
    workspace_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    backend: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")
    permission_class: Literal["inspect", "workspace_write"] = "inspect"
    dry_run: bool = True
    experience_ids: list[str] = Field(default_factory=list, max_length=3)


class WorkerDryRunRequest(BaseModel):
    title: str = Field(default="STN03 Worker transport dry-run", min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=12000)
    workspace_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    backend: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9._-]+$")


class ExecutionReviewRequest(BaseModel):
    decision: Literal["ACCEPT", "REJECT", "NEEDS_REVISION"]
    notes: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)
    accepted_scope: list[str] = Field(default_factory=list, max_length=50)


class ExperienceReviewRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT", "NEEDS_REVISION"]
    notes: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=50)


class ExperienceRevisionRequest(BaseModel):
    lesson_hypothesis: str = Field(min_length=20, max_length=1200)


class ExperienceDeactivationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)
    disposition: Literal["WITHDRAWN", "SUPERSEDED", "STALE"] = "WITHDRAWN"
    superseded_by: Optional[str] = Field(default=None, max_length=100)


class ExperienceRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=3, ge=1, le=10)


def _chat_runtime_config(request: Request):
    return request.app.state.cfg.get("chat_runtime", {})


def _require_interactive_chat(request: Request):
    settings = _chat_runtime_config(request)
    if not settings.get("enabled", False):
        raise HTTPException(status_code=404, detail="Interactive chat is disabled")
    readiness = settings.get("readiness", {})
    model = settings.get("model", {}) or {}
    if not request.app.state.auth_settings.ready:
        raise HTTPException(status_code=503, detail="SIS authentication is not configured")
    if not getattr(request.app.state, "database_ready", False):
        raise HTTPException(status_code=503, detail="The configured database is not ready")
    if not (model.get("name") and model.get("api_base")):
        raise HTTPException(status_code=503, detail="No chat model is configured")
    if not readiness.get("model_verified", False):
        raise HTTPException(status_code=503, detail="The configured model has not passed its live verification gate")
    if not readiness.get("live_persistence_verified", False):
        raise HTTPException(status_code=503, detail="Live conversation persistence has not passed its verification gate")
    return settings


def _chat_readiness(request: Request) -> dict:
    settings = _chat_runtime_config(request)
    readiness = settings.get("readiness", {})
    model = settings.get("model", {}) or {}
    auth_ready = bool(request.app.state.auth_settings.ready)
    database_ready = bool(getattr(request.app.state, "database_ready", False))
    model_configured = bool(model.get("name") and model.get("api_base"))
    model_verified = bool(readiness.get("model_verified", False))
    persistence_verified = bool(readiness.get("live_persistence_verified", False))
    configured = bool(settings.get("enabled", False))
    ready = all(
        (configured, auth_ready, database_ready, model_configured, model_verified, persistence_verified)
    )
    if not configured:
        message = "Interactive chat is disabled by configuration."
    elif not auth_ready:
        message = "Configure the local SIS account and session secret."
    elif not database_ready:
        message = "The configured database is not ready."
    elif not model_configured:
        message = "No chat model is configured."
    elif not model_verified:
        message = "The configured model has not passed a real inference check."
    elif not persistence_verified:
        message = "A real conversation has not yet passed persistence and restart verification."
    else:
        message = "Chat is ready."
    return {
        "configured": configured,
        "auth_ready": auth_ready,
        "database_ready": database_ready,
        "model_configured": model_configured,
        "model_verified": model_verified,
        "persistence_verified": persistence_verified,
        "enabled": ready,
        "model_name": model.get("name"),
        "message": message,
    }


def _owned_conversation(request: Request, chat_id: int):
    conversation = request.app.state.memory.chats.get_conversation(chat_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    owner = (conversation.meta or {}).get("sis_owner")
    if owner and owner != request.state.username:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _worker_execution_enabled(request: Request):
    settings = request.app.state.cfg.get("worker", {}) or {}
    if not settings.get("enabled", False):
        raise HTTPException(status_code=404, detail="Worker execution is disabled")
    return settings


def _require_worker_runtime_access(request: Request):
    settings = _worker_execution_enabled(request)
    if not request.app.state.auth_settings.ready:
        raise HTTPException(status_code=503, detail="SIS authentication is not configured")
    if not getattr(request.app.state, "database_ready", False):
        raise HTTPException(status_code=503, detail="The configured database is not ready")
    return settings


def _require_experience_runtime_access(request: Request):
    settings = request.app.state.cfg.get("experience_learning", {}) or {}
    if not settings.get("enabled", False):
        raise HTTPException(status_code=404, detail="Experience review and retrieval are disabled")
    if not request.app.state.auth_settings.ready:
        raise HTTPException(status_code=503, detail="SIS authentication is not configured")
    if not getattr(request.app.state, "database_ready", False):
        raise HTTPException(status_code=503, detail="The configured database is not ready")
    return settings


def _worker_not_ready_for_chat(request: Request) -> dict:
    settings = request.app.state.cfg.get("worker", {}) or {}
    return {
        "configured": bool(settings.get("enabled", False)),
        "enabled": False,
        "available": False,
        "backend": settings.get("default_backend", "opencode"),
        "model": settings.get("model"),
        "allowed_backends": list(settings.get("allowed_backends", []) or []),
        "workspaces": [],
        "dry_run_available": False,
        "message": "The existing SIS chat/readiness gate is not ready; Worker was not contacted.",
    }


def _worker_error(exc: WorkerExecutionError) -> HTTPException:
    status = 503
    if exc.code in {"WORKSPACE_NOT_AUTHORIZED", "WORKSPACE_POLICY_UNVERIFIED"}:
        status = 403
    elif exc.code in {"OBJECTIVE_INVALID", "OBJECTIVE_CONTAINS_SECRET", "IDEMPOTENCY_KEY_INVALID", "BACKEND_NOT_ALLOWED", "PERMISSION_CLASS_NOT_ALLOWED", "WORKSPACE_CONFIG_INVALID", "EXPERIENCE_REUSE_DISABLED", "EXPERIENCE_BACKEND_NOT_ISOLATED", "EXPERIENCE_CONTEXT_INVALID", "EXPERIENCE_CONTEXT_TOO_LARGE"}:
        status = 422
    elif exc.code == "IDEMPOTENCY_CONFLICT":
        status = 409
    elif exc.code == "EXECUTION_CAPACITY":
        status = 429
    elif exc.code == "EXECUTION_DISABLED":
        status = 404
    return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})


def _experience_error(exc: ExperienceError) -> HTTPException:
    status = 503
    if exc.code in {"EXECUTION_NOT_FOUND", "PLANTRACE_NOT_FOUND", "EXPERIENCE_NOT_FOUND"}:
        status = 404
    elif exc.code in {"HUMAN_ACCEPTANCE_REQUIRED", "VERIFICATION_NOT_PASSED", "EXECUTION_NOT_REVIEWABLE", "EXPERIENCE_NOT_ELIGIBLE", "EXPERIENCE_SELECTION_UNAVAILABLE", "REVIEW_HISTORY_FULL", "EXPERIENCE_REVIEW_HISTORY_FULL"}:
        status = 409
    elif exc.code.endswith("INVALID") or exc.code.endswith("REQUIRED") or exc.code in {"OBJECTIVE_MISSING", "ARTIFACT_EVIDENCE_MISSING"}:
        status = 422
    elif exc.code.endswith("CONFLICT") or exc.code.endswith("FINALIZED"):
        status = 409
    elif exc.code in {"EXPERIENCE_DISABLED", "EXPERIENCE_RETRIEVAL_DISABLED"}:
        status = 404
    return HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)})

@router.get("/chats", response_class=HTMLResponse)
async def list_chats(
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
    username = request.state.username
    chats = [
        {
            "turn_count": turn_count,
            "conversation": conv.to_dict(),  # Serialize conversation data
        }
        for (conv, turn_count) in rows
        if not (conv.meta or {}).get("sis_owner")
        or (conv.meta or {}).get("sis_owner") == username
    ]

    chat_status = _chat_readiness(request)
    worker_settings = request.app.state.cfg.get("worker", {}) or {}
    worker_enabled = bool(worker_settings.get("enabled", False))
    dry_run_only = bool(worker_settings.get("dry_run_only", False))
    if chat_status["enabled"] or worker_enabled or dry_run_only:
        worker_status = await asyncio.to_thread(
            request.app.state.worker_execution.readiness,
            dry_run=(not worker_enabled or dry_run_only),
        )
        if worker_enabled or dry_run_only:
            worker_status["configured"] = True
            worker_status["dry_run_only"] = dry_run_only
    else:
        worker_status = _worker_not_ready_for_chat(request)

    # Render the chat list template
    return templates.TemplateResponse(
        "/chats/list.html",
        {
            "request": request,
            "chats": chats,
            "filters": {"provider": provider, "limit": limit},
            "chat_status": chat_status,
            "worker_status": worker_status,
        },
    )


@router.post("/chats", status_code=201)
async def create_chat(
    request: Request,
    payload: ConversationCreateRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Create a live SIS conversation using the existing ChatStore schema."""
    if _chat_runtime_config(request).get("enabled", False):
        _require_interactive_chat(request)
    else:
        _require_worker_runtime_access(request)
    title = payload.title.strip() or "New conversation"
    request_id = idempotency_key or uuid4().hex
    try:
        conversation = await request.app.state.chat_execution.create_conversation(
            username=request.state.username,
            title=title,
            request_id=request_id,
        )
    except ValueError as exc:
        status = 422 if "Idempotency-Key" in str(exc) else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return {"conversation": conversation.to_dict(), "request_id": request_id}


@router.post("/chats/{chat_id}/messages")
async def send_chat_message(
    request: Request,
    chat_id: int,
    payload: ChatMessageRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Execute and persist one idempotent turn through the conversation service."""
    _require_interactive_chat(request)
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Message cannot be empty")
    request_id = idempotency_key or uuid4().hex
    execution = request.app.state.chat_execution
    work = asyncio.create_task(
        execution.execute_turn(
            conversation_id=chat_id,
            username=request.state.username,
            text=text,
            request_id=request_id,
        )
    )
    try:
        return await work
    except asyncio.CancelledError as exc:
        if await execution.consume_cancelled(chat_id, request_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Request cancelled. Your user message remains saved and can be retried.",
                    "request_id": request_id,
                },
            ) from exc
        work.cancel()
        raise
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ConversationBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail="Idempotency-Key was already used for another message") from exc
    except ChatServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail="Stephanie's LLM service is unavailable") from exc
    except ChatInferenceFailed as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Assistant response failed; your user message remains saved and can be retried with the same Idempotency-Key.",
                "request_id": request_id,
            },
        ) from exc
    except ValueError as exc:
        status_code = 422 if "Idempotency-Key" in str(exc) else 413
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/chats/{chat_id}/requests/{request_id}/cancel")
async def cancel_chat_request(request: Request, chat_id: int, request_id: str):
    """Cancel an in-flight model request; no assistant turn is fabricated."""
    _require_interactive_chat(request)
    try:
        cancelled = await request.app.state.chat_execution.cancel_turn(
            conversation_id=chat_id,
            username=request.state.username,
            request_id=request_id,
        )
    except ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"cancelled": cancelled}


@router.get("/chats/{chat_id}/executions/readiness")
async def worker_execution_readiness(
    request: Request,
    chat_id: int,
    mode: Literal["dry_run", "execution"] = Query(default="dry_run"),
    backend: Optional[str] = Query(default=None, max_length=80, pattern=r"^[A-Za-z0-9._-]+$"),
):
    _owned_conversation(request, chat_id)
    chat_ready = _chat_readiness(request)["enabled"]
    worker_settings = request.app.state.cfg.get("worker", {}) or {}
    worker_enabled = bool(worker_settings.get("enabled", False))
    dry_run_only = bool(worker_settings.get("dry_run_only", False))
    if chat_ready or worker_enabled or dry_run_only:
        result = await asyncio.to_thread(
            request.app.state.worker_execution.readiness,
            backend,
            dry_run=(mode == "dry_run" or dry_run_only),
        )
        if worker_enabled or dry_run_only:
            result["configured"] = True
            result["dry_run_only"] = dry_run_only
    else:
        result = _worker_not_ready_for_chat(request)
    result["chat_enabled"] = chat_ready
    return result


@router.get("/chats/{chat_id}/executions")
async def list_worker_executions(request: Request, chat_id: int):
    _owned_conversation(request, chat_id)
    return {"executions": await asyncio.to_thread(request.app.state.worker_execution.list, chat_id)}


@router.post("/chats/worker-dry-runs", status_code=202)
async def create_worker_dry_run(
    request: Request,
    payload: WorkerDryRunRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Submit a no-agent Worker transport check without enabling interactive chat."""
    settings = request.app.state.cfg.get("worker", {}) or {}
    if not settings.get("dry_run_only", False) or settings.get("enabled", False):
        raise HTTPException(status_code=404, detail="Worker dry-run-only route is not enabled")
    if not request.app.state.auth_settings.ready:
        raise HTTPException(status_code=503, detail="SIS authentication is not configured")
    if not getattr(request.app.state, "database_ready", False):
        raise HTTPException(status_code=503, detail="The configured database is not ready")
    submission_key = idempotency_key or uuid4().hex
    if len(submission_key) > 110:
        raise HTTPException(status_code=422, detail="Idempotency-Key must be at most 110 characters for a dry-run")
    readiness = await asyncio.to_thread(
        request.app.state.worker_execution.readiness, payload.backend, dry_run=True
    )
    if not readiness.get("dry_run_available"):
        raise HTTPException(status_code=503, detail=readiness.get("message", "Worker dry-run is not ready"))
    conversation = request.app.state.memory.chats.add_conversation_once(
        {
            "provider": "stephanie",
            "title": payload.title.strip() or "STN03 Worker transport dry-run",
            "meta": {
                "source": "sis_worker_dry_run",
                "sis_owner": request.state.username,
                "sis_dry_run_only": True,
            },
        },
        f"stn03-dry-run:{submission_key}",
    )
    try:
        execution = await asyncio.to_thread(
            request.app.state.worker_execution.submit,
            conversation_id=conversation.id,
            username=request.state.username,
            objective=payload.objective,
            workspace_id=payload.workspace_id,
            backend=payload.backend,
            permission_class="inspect",
            submission_key=submission_key,
            dry_run=True,
        )
    except WorkerExecutionError as exc:
        raise _worker_error(exc) from exc
    return {
        "conversation_id": conversation.id,
        "conversation_url": f"/chats/{conversation.id}",
        "execution": execution,
        "submission_key": submission_key,
        "label": "DRY RUN — NO AGENT EXECUTION",
    }


@router.post("/chats/{chat_id}/executions", status_code=202)
async def create_worker_execution(
    request: Request,
    chat_id: int,
    payload: WorkerExecutionRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    _require_worker_runtime_access(request)
    _owned_conversation(request, chat_id)
    submission_key = idempotency_key or uuid4().hex
    experience_context = None
    if payload.experience_ids:
        try:
            experience_context = await asyncio.to_thread(
                request.app.state.execution_experience.retrieve_for_task,
                payload.objective,
                limit=len(payload.experience_ids),
                experience_ids=payload.experience_ids,
            )
        except ExperienceError as exc:
            raise _experience_error(exc) from exc
    try:
        execution = await asyncio.to_thread(
            request.app.state.worker_execution.submit,
            conversation_id=chat_id,
            username=request.state.username,
            objective=payload.objective,
            workspace_id=payload.workspace_id,
            backend=payload.backend,
            permission_class=payload.permission_class,
            submission_key=submission_key,
            dry_run=payload.dry_run,
            experience_context=experience_context,
        )
    except WorkerExecutionError as exc:
        raise _worker_error(exc) from exc
    return {"execution": execution, "submission_key": submission_key}


@router.get("/chats/{chat_id}/executions/{execution_id}")
async def get_worker_execution(request: Request, chat_id: int, execution_id: str):
    _owned_conversation(request, chat_id)
    execution = await asyncio.to_thread(
        request.app.state.worker_execution.get, chat_id, execution_id
    )
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return {"execution": execution}


@router.get("/chats/{chat_id}/executions/{execution_id}/evidence", response_class=HTMLResponse)
async def view_worker_execution_evidence(request: Request, chat_id: int, execution_id: str):
    _owned_conversation(request, chat_id)
    execution = await asyncio.to_thread(
        request.app.state.worker_execution.get, chat_id, execution_id
    )
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    experience = None
    if execution.get("experience_id"):
        try:
            experience = await asyncio.to_thread(
                request.app.state.execution_experience.get_candidate,
                execution["experience_id"],
            )
        except ExperienceError:
            experience = None
    return request.app.state.templates.TemplateResponse(
        "/chats/execution_evidence.html",
        {"request": request, "execution": execution, "experience": experience, "conversation_id": chat_id},
    )


@router.post("/chats/{chat_id}/executions/{execution_id}/review")
async def review_worker_execution(
    request: Request,
    chat_id: int,
    execution_id: str,
    payload: ExecutionReviewRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        result = await asyncio.to_thread(
            request.app.state.execution_experience.review_execution,
            conversation_id=chat_id,
            execution_id=execution_id,
            reviewer=request.state.username,
            decision=payload.decision,
            review_id=idempotency_key or "",
            notes=payload.notes,
            evidence_refs=payload.evidence_refs,
            accepted_scope=payload.accepted_scope,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"review": result, "worker_acceptance_id": None}


@router.post("/chats/{chat_id}/executions/{execution_id}/experience-candidate", status_code=201)
async def create_execution_experience_candidate(request: Request, chat_id: int, execution_id: str):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        candidate = await asyncio.to_thread(
            request.app.state.execution_experience.extract_candidate,
            conversation_id=chat_id,
            execution_id=execution_id,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"experience": candidate}


@router.get("/chats/{chat_id}/executions/{execution_id}/experience-candidate")
async def get_execution_experience_candidate(request: Request, chat_id: int, execution_id: str):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    record = await asyncio.to_thread(
        request.app.state.memory.chats.get_external_execution, chat_id, execution_id
    )
    if not record:
        raise HTTPException(status_code=404, detail="Execution not found")
    experience_id = record.get("experience_id")
    if not experience_id:
        raise HTTPException(status_code=404, detail="Experience candidate has not been extracted")
    candidate = await asyncio.to_thread(
        request.app.state.execution_experience.get_candidate, experience_id
    )
    if not candidate:
        raise HTTPException(status_code=404, detail="Experience candidate not found")
    return {"experience": candidate}


@router.post("/chats/{chat_id}/executions/{execution_id}/experience-candidate/review")
async def review_execution_experience_candidate(
    request: Request,
    chat_id: int,
    execution_id: str,
    payload: ExperienceReviewRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        result = await asyncio.to_thread(
            request.app.state.execution_experience.review_candidate,
            conversation_id=chat_id,
            execution_id=execution_id,
            reviewer=request.state.username,
            decision=payload.decision,
            review_id=idempotency_key or "",
            notes=payload.notes,
            evidence_refs=payload.evidence_refs,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"experience": result}


@router.post("/chats/{chat_id}/executions/{execution_id}/experience-candidate/revise")
async def revise_execution_experience_candidate(
    request: Request,
    chat_id: int,
    execution_id: str,
    payload: ExperienceRevisionRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        result = await asyncio.to_thread(
            request.app.state.execution_experience.revise_candidate,
            conversation_id=chat_id,
            execution_id=execution_id,
            reviewer=request.state.username,
            revision_id=idempotency_key or "",
            lesson_text=payload.lesson_hypothesis,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"experience": result}


@router.post("/chats/{chat_id}/experiences/{experience_id}/deactivate")
async def deactivate_execution_experience(
    request: Request,
    chat_id: int,
    experience_id: str,
    payload: ExperienceDeactivationRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        result = await asyncio.to_thread(
            request.app.state.execution_experience.deactivate_experience,
            conversation_id=chat_id,
            experience_id=experience_id,
            reviewer=request.state.username,
            reason=payload.reason,
            review_id=idempotency_key or "",
            disposition=payload.disposition,
            superseded_by=payload.superseded_by,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"experience": result}


@router.post("/chats/{chat_id}/experiences/retrieve")
async def preview_execution_experiences(
    request: Request,
    chat_id: int,
    payload: ExperienceRetrievalRequest,
):
    _require_experience_runtime_access(request)
    _owned_conversation(request, chat_id)
    try:
        result = await asyncio.to_thread(
            request.app.state.execution_experience.retrieve_for_task,
            payload.query,
            limit=payload.limit,
        )
    except ExperienceError as exc:
        raise _experience_error(exc) from exc
    return {"retrieval": result}


@router.get("/chats/{chat_id}", response_class=HTMLResponse)
async def chat_detail(request: Request, chat_id: int):
    """
    Display detailed view of a specific conversation.
    
    Args:
        request: FastAPI Request object
        chat_id: ID of the conversation to display
        
    Returns:
        Rendered HTML template with conversation details or 404 error
    """
    _owned_conversation(request, chat_id)
    memory = request.app.state.memory
    templates = request.app.state.templates
    # Retrieve complete conversation data with messages and turns
    convo = memory.chats.get_conversation_dict(
        chat_id,
        include_messages=True,
        include_turns=True,
        include_turn_message_texts=True,  # Flatten message texts for easy template access
    )
    chat_status = _chat_readiness(request)
    worker_settings = request.app.state.cfg.get("worker", {}) or {}
    worker_enabled = bool(worker_settings.get("enabled", False))
    dry_run_only = bool(worker_settings.get("dry_run_only", False))
    if chat_status["enabled"] or worker_enabled or dry_run_only:
        worker_status = await asyncio.to_thread(
            request.app.state.worker_execution.readiness,
            dry_run=(not worker_enabled or dry_run_only),
        )
        if worker_enabled or dry_run_only:
            worker_status["configured"] = True
            worker_status["dry_run_only"] = dry_run_only
    else:
        worker_status = _worker_not_ready_for_chat(request)
    worker_executions = await asyncio.to_thread(
        request.app.state.worker_execution.list, chat_id
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
        {
            "request": request,
            "conversation": convo,
            "chat_status": chat_status,
            "worker_status": worker_status,
            "worker_executions": worker_executions,
            "experience_learning_enabled": bool((request.app.state.cfg.get("experience_learning", {}) or {}).get("enabled", False)),
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
    _owned_conversation(request, chat_id)
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
def set_turn_star(request: Request, chat_id: int, turn_id: int, star: int = Form(..., ge=-5, le=5)):
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
    _owned_conversation(request, chat_id)
    turn = memory.chats.get_turn_by_id(turn_id)
    if turn is None or turn.conversation_id != chat_id:
        raise HTTPException(status_code=404, detail="Turn not found")

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
    _owned_conversation(request, chat_id)
    # Get annotation progress for both domains and NER
    dom = memory.chats.annotation_progress(chat_id, kind="domains")
    ner = memory.chats.annotation_progress(chat_id, kind="ner")
    
    # Return progress data as JSON
    return {"chat_id": chat_id, "domains": dom, "ner": ner}
