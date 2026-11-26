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
import re
from typing import Optional
from typing import Literal
from uuid import uuid4
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, Query, Header, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field
from sis.services.chat_context import build_chat_messages as _build_chat_messages
from sis.services.mission_project_context import MissionProjectContext
from sis.services.model_preferences import ModelPreferences
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
    external_approved: bool = False
    project_context_approved: bool = False
    model_route: Optional[str] = Field(default=None, max_length=180)
    mission_project_id: Optional[str] = Field(default=None, max_length=120)


class ChatPreferencesRequest(BaseModel):
    preferred_model: str = Field(default="opencode-go/deepseek-v4.1-flash", max_length=180)
    policy: Literal["free_first", "specific"] = "free_first"
    allow_subscription_models: bool = True
    allow_metered_models: bool = False
    external_permissions: dict[str, bool] = Field(default_factory=dict)


class ConversationPreferencesRequest(BaseModel):
    model_route: Optional[str] = Field(default=None, max_length=180)
    mission_project_id: Optional[str] = Field(default=None, max_length=120)


class ProjectGrantRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    provider: str = Field(min_length=1, max_length=180)
    categories: list[str] = Field(default_factory=list, max_length=8)


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


def _model_preferences(request: Request) -> ModelPreferences:
    settings = _chat_runtime_config(request)
    return ModelPreferences(settings.get("preferences_path"))


def _model_is_selectable(option: dict | None) -> bool:
    return bool(option and option.get("availability") == "available" and option.get("status") != "blocked")


def _model_options(request: Request) -> list[dict]:
    settings = _chat_runtime_config(request)
    routing = settings.get("routing", {}) or {}
    preferences = _model_preferences(request).read()
    try:
        llm = request.app.state.container.get("llm")
        api_key_present = bool(getattr(getattr(llm, "_free_first_chat", None), "api_key", ""))
    except Exception:
        api_key_present = bool(routing.get("api_key"))

    free_route_count = sum(
        1 for route in (routing.get("routes", []) or [])
        if route.get("economic_tier") == "free"
    )
    options = [{
        "id": "automatic",
        "label": "Automatic — free first",
        "provider": "automatic",
        "economic_tier": "free_first",
        "availability": "available",
        "reason": (
            "Tries configured eligible free routes first, then an approved subscription route if enabled, or the local fallback."
            if free_route_count else
            "No free route is configured. Uses an approved subscription route if enabled, or the local fallback."
        ),
    }]
    observations = preferences.get("model_observations", {})
    for route in routing.get("routes", []) or []:
        provider = str(route.get("provider") or routing.get("provider") or "unknown")
        model = str(route.get("model") or "")
        endpoint = str(route.get("endpoint") or routing.get("endpoint") or "")
        route_id = f"{provider}/{model}"
        supported_endpoint = (
            (provider == "opencode" and endpoint == "https://opencode.ai/inference/openai/v1")
            or (provider == "opencode-go" and endpoint == "https://opencode.ai/inference/go/openai/v1")
        )
        tier = str(route.get("economic_tier", "unknown"))
        provider_label = {"opencode-go": "OpenCode Go", "opencode": "OpenCode"}.get(provider, provider)
        model_label = {"deepseek-v4.1-flash": "DeepSeek V4.1 Flash"}.get(model, model)
        tier_eligible = tier in {"free", "subscription_included"}
        subscription_enabled = tier != "subscription_included" or preferences.get("allow_subscription_models", True)
        availability = "available" if api_key_present and supported_endpoint and tier_eligible and subscription_enabled else "unavailable"
        if not tier_eligible:
            reason = "Metered or unknown-cost routes are disabled in this release"
        elif not subscription_enabled:
            reason = "Enable subscription-included routes in Settings to select this model"
        else:
            reason = "Configured route" if availability == "available" else ("Provider credentials are not configured" if not api_key_present else "No supported chat transport is configured for this provider")
        options.append({
            "id": route_id,
            "label": f"{provider_label} — {model_label}",
            "provider": provider,
            "model": model,
            "economic_tier": str(route.get("economic_tier", "unknown")),
            "availability": availability,
            "status": observations.get(route_id, {}).get("status", "unverified"),
            "status_reason": observations.get(route_id, {}).get("reason", ""),
            "reason": reason,
            "data_classes": list(route.get("allowed_data_classes", []) or []),
        })

    for model_id in ("deepseek-v4.1-flash-free", "hy4-flash-free"):
        route_id = f"opencode/{model_id}"
        options.append({"id": route_id, "label": f"OpenCode Free — {model_id}", "provider": "opencode",
                        "model": model_id, "economic_tier": "free", "availability": "unavailable",
                        "status": "blocked", "status_reason": "OpenCode documents that free inference models are available only inside the OpenCode application; direct API use by other clients is denied (HTTP 403).",
                        "reason": "No supported Stephanie direct-API transport", "data_classes": []})

    model = settings.get("model", {}) or {}
    local_name = str(model.get("name") or "")
    local_base = str(model.get("api_base") or "")
    parsed = urlparse(local_base)
    loopback_ollama = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if loopback_ollama:
        try:
            import httpx
            tags_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/api/tags"
            response = httpx.get(tags_url, timeout=1.2)
            response.raise_for_status()
            tags = response.json().get("models", [])
        except Exception:
            tags = []
    else:
        tags = []
    local_models = {local_name.removeprefix("ollama/")} if local_name.startswith("ollama/") else set()
    local_models.update(str(item.get("name")) for item in tags if isinstance(item, dict) and item.get("name"))
    for local in sorted(local_models):
        normalized_local = local.lower()
        cloud_route = "cloud" in re.split(r"[/_:-]+", normalized_local)
        embedding_route = "embed" in normalized_local or normalized_local.startswith("all-minilm") or normalized_local.startswith("bge-m3")
        local_available = loopback_ollama and (local == local_name.removeprefix("ollama/") or bool(tags)) and not cloud_route and not embedding_route
        if not loopback_ollama:
            local_reason = "Ollama endpoint is not loopback; it is not approved as a local route"
            local_tier = "unknown"
        elif cloud_route:
            local_reason = "Cloud-tagged Ollama model is not approved as a local route; external metered inference is disabled"
            local_tier = "unknown"
        elif embedding_route:
            local_reason = "Embedding model; not eligible for conversational chat"
            local_tier = "local"
        else:
            local_reason = "Runs on this computer" if local_available else "Configured route; Ollama model list could not be verified"
            local_tier = "local"
        options.append({
            "id": f"local/ollama/{local}",
            "label": f"Ollama — {local}",
            "provider": "ollama",
            "model": local,
            "economic_tier": local_tier,
            "availability": "available" if local_available else "unavailable",
            "status": observations.get(f"local/ollama/{local}", {}).get("status", "unverified"),
            "status_reason": observations.get(f"local/ollama/{local}", {}).get("reason", ""),
            "reason": local_reason,
        })
    return options


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
    routing = settings.get("routing", {}) or {}
    external_routes = [
        {
            "name": f"{item.get('provider', routing.get('provider', 'unknown'))}/{item.get('model', 'unknown')}",
            "economic_tier": item.get("economic_tier", "unknown"),
            "data_classes": list(item.get("allowed_data_classes", []) or []),
        }
        for item in (routing.get("routes", []) or [])
    ]
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
        "external_routes": external_routes,
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
    current_preferences = _model_preferences(request).read()
    request.app.state.memory.chats.update_sis_chat_preferences(
        conversation.id,
        {"model_route": current_preferences["preferred_model"], "mission_project_id": None},
    )
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
    if payload.model_route:
        options = {item["id"]: item for item in await asyncio.to_thread(_model_options, request)}
        if not _model_is_selectable(options.get(payload.model_route)):
            raise HTTPException(status_code=422, detail="Selected model is unavailable or blocked for this chat service")
    request_id = idempotency_key or uuid4().hex
    execution = request.app.state.chat_execution
    work = asyncio.create_task(
        execution.execute_turn(
            conversation_id=chat_id,
            username=request.state.username,
            text=text,
            external_approved=payload.external_approved,
            project_context_approved=payload.project_context_approved,
            model_route=payload.model_route,
            mission_project_id=payload.mission_project_id,
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
                "failure_evidence": exc.evidence,
            },
        ) from exc
    except ValueError as exc:
        status_code = 422 if "Idempotency-Key" in str(exc) else 413
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get("/chats/mission/projects")
async def mission_projects_for_chat(request: Request):
    """Expose a minimal project picker backed only by Mission GET endpoints."""
    _require_interactive_chat(request)
    mission_cfg = request.app.state.cfg.get("mission_context", {}) or {}
    if not mission_cfg.get("enabled", False):
        return JSONResponse(status_code=503, content={"available": False, "error_type": "MissionContextDisabled", "projects": []})
    context = MissionProjectContext(mission_cfg)
    try:
        result = await asyncio.to_thread(context.list_projects)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"available": False, "error_type": type(exc).__name__, "projects": []},
        )
    return {"available": True, **result}


@router.get("/chats/models")
async def chat_model_options(request: Request):
    """List configured routes and loopback Ollama models without credentials."""
    _require_interactive_chat(request)
    preferences = _model_preferences(request).read()
    options = await asyncio.to_thread(_model_options, request)
    return {"models": options, "preferences": preferences}


@router.get("/chats/preferences")
async def get_chat_preferences(request: Request):
    _require_interactive_chat(request)
    return _model_preferences(request).read()


@router.get("/chats/settings", response_class=HTMLResponse)
async def chat_settings_page(request: Request):
    _require_interactive_chat(request)
    return request.app.state.templates.TemplateResponse(
        "/chats/settings.html",
        {"request": request, "preferences": _model_preferences(request).read(),
         "model_options": await asyncio.to_thread(_model_options, request)},
    )


@router.put("/chats/preferences")
async def update_chat_preferences(request: Request, payload: ChatPreferencesRequest):
    _require_interactive_chat(request)
    options = {item["id"]: item for item in await asyncio.to_thread(_model_options, request)}
    if not _model_is_selectable(options.get(payload.preferred_model)):
        raise HTTPException(status_code=422, detail="Preferred model is not an available configured route")
    allowed_permissions = {"ordinary_project_metadata", "repository_locations", "publication_status", "approved_project_documentation"}
    permissions = {key: bool(value) for key, value in payload.external_permissions.items() if key in allowed_permissions}
    return _model_preferences(request).write({
        "preferred_model": payload.preferred_model,
        "policy": payload.policy,
        "allow_subscription_models": payload.allow_subscription_models,
        "allow_metered_models": False,
        "external_permissions": permissions,
        "model_observations": _model_preferences(request).read().get("model_observations", {}),
        "project_grants": _model_preferences(request).read().get("project_grants", []),
    })


@router.post("/chats/project-grants")
async def create_project_grant(request: Request, payload: ProjectGrantRequest):
    _require_interactive_chat(request)
    allowed = {"ordinary_project_metadata", "repository_locations", "publication_status", "approved_project_documentation"}
    categories = sorted(set(payload.categories) & allowed)
    if not categories:
        raise HTTPException(status_code=422, detail="Select at least one permitted category")
    options = {item["id"]: item for item in await asyncio.to_thread(_model_options, request)}
    route = options.get(payload.provider)
    if not _model_is_selectable(route) or route.get("provider") in {"automatic", "local"}:
        raise HTTPException(status_code=422, detail="Grant must name an available external provider route")
    prefs = _model_preferences(request).read()
    grants = [g for g in prefs.get("project_grants", []) if not (g["project_id"] == payload.project_id and g["provider"] == payload.provider)]
    grants.append({"project_id": payload.project_id, "provider": payload.provider, "categories": categories})
    prefs["project_grants"] = grants
    return _model_preferences(request).write(prefs)


@router.delete("/chats/project-grants")
async def revoke_project_grant(request: Request, project_id: str = Query(...), provider: str = Query(...)):
    _require_interactive_chat(request)
    prefs = _model_preferences(request).read()
    prefs["project_grants"] = [g for g in prefs.get("project_grants", []) if not (g["project_id"] == project_id and g["provider"] == provider)]
    return _model_preferences(request).write(prefs)


@router.put("/chats/{chat_id}/sharing")
async def grant_conversation_sharing(request: Request, chat_id: int, payload: ProjectGrantRequest):
    _require_interactive_chat(request)
    conversation = _owned_conversation(request, chat_id)
    chat_prefs = (conversation.meta or {}).get("sis_chat_preferences", {})
    if chat_prefs.get("mission_project_id") != payload.project_id or chat_prefs.get("model_route") != payload.provider:
        raise HTTPException(status_code=409, detail="Select the same project and provider in this conversation before granting permission")
    options = {item["id"]: item for item in await asyncio.to_thread(_model_options, request)}
    route = options.get(payload.provider)
    if not _model_is_selectable(route) or route.get("provider") in {"automatic", "local"}:
        raise HTTPException(status_code=422, detail="Grant must name an available external provider route")
    allowed = {"ordinary_project_metadata", "repository_locations", "publication_status", "approved_project_documentation"}
    categories = sorted(set(payload.categories) & allowed)
    if not categories:
        raise HTTPException(status_code=422, detail="Select at least one permitted category")
    updated = request.app.state.memory.chats.update_sis_chat_preferences(chat_id, {
        "sharing_grant": {"project_id": payload.project_id, "provider": payload.provider, "categories": categories}
    })
    return {"conversation_id": conversation.id, "sharing_grant": (updated.meta or {}).get("sis_chat_preferences", {}).get("sharing_grant")}


@router.delete("/chats/{chat_id}/sharing")
async def revoke_conversation_sharing(request: Request, chat_id: int):
    _require_interactive_chat(request)
    conversation = _owned_conversation(request, chat_id)
    chat_prefs = dict((conversation.meta or {}).get("sis_chat_preferences", {}))
    chat_prefs.pop("sharing_grant", None)
    updated = request.app.state.memory.chats.update_sis_chat_preferences(chat_id, chat_prefs)
    return {"conversation_id": conversation.id, "sharing_grant": (updated.meta or {}).get("sis_chat_preferences", {}).get("sharing_grant")}


@router.put("/chats/{chat_id}/preferences")
async def update_conversation_preferences(request: Request, chat_id: int, payload: ConversationPreferencesRequest):
    _require_interactive_chat(request)
    conversation = _owned_conversation(request, chat_id)
    values = {}
    if payload.model_route:
        options = {item["id"]: item for item in await asyncio.to_thread(_model_options, request)}
        if not _model_is_selectable(options.get(payload.model_route)):
            raise HTTPException(status_code=422, detail="Model route is unavailable")
        values["model_route"] = payload.model_route
    if payload.mission_project_id is not None:
        if payload.mission_project_id and not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", payload.mission_project_id):
            raise HTTPException(status_code=422, detail="Invalid Mission project identifier")
        values["mission_project_id"] = payload.mission_project_id or None
    updated = request.app.state.memory.chats.update_sis_chat_preferences(chat_id, values)
    return {"conversation_id": conversation.id, "preferences": (updated.meta or {}).get("sis_chat_preferences", {})}


@router.get("/chats/mission/projects/{project_id}/context")
async def mission_project_context_for_chat(request: Request, project_id: str):
    _require_interactive_chat(request)
    mission_cfg = request.app.state.cfg.get("mission_context", {}) or {}
    if not mission_cfg.get("enabled", False):
        raise HTTPException(status_code=503, detail="Mission project context is disabled")
    try:
        context = await asyncio.to_thread(request.app.state.chat_execution.mission_context.get_project, project_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Project context unavailable ({type(exc).__name__})") from exc
    return context


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
    global_preferences = _model_preferences(request).read()
    conversation_preferences = (convo.get("meta") or {}).get("sis_chat_preferences", {}) if convo else {}
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
            "chat_preferences": {**global_preferences, **conversation_preferences},
            "model_options": await asyncio.to_thread(_model_options, request),
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
