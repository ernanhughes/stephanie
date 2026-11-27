"""Human directed, deterministic prompt generation through Capsule."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from sis.services.capsule_prompt_client import CapsulePromptClient, CapsulePromptError
from sis.services.context_handoff import StephanieContextHandoff
from sis.security import load_context_origins

router = APIRouter()
_cache: OrderedDict[str, tuple[str, dict]] = OrderedDict()
_cache_lock = threading.RLock()
_CACHE_LIMIT = 48


class PreparePromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    context: StephanieContextHandoff
    operation: Literal["repository.review", "repository.next", "repository.status_brief", "project.create", "document.review", "document.improve", "chapter.review", "chapter.improve_readability", "book.remove_artifacts", "book.technical.create"] = "repository.review"
    document_context: dict[str, Any] | None = None
    instructions: str = Field(default="", max_length=8000)
    external_handoff_confirmed: bool = False
    destination: Literal["codex", "opencode", "claude_code", "chatgpt_web", "claude_web", "worker", "local"] = "codex"
    size_profile: Literal["compact", "standard", "detailed", "custom"] = "standard"
    custom_token_budget: int | None = Field(default=None, ge=1, le=1_000_000)


def _client() -> CapsulePromptClient:
    return CapsulePromptClient()


def _render_markdown(package: dict) -> str:
    stage, subject, capsule = package["stage"], package["subject"], package["capsule"]
    destination=package.get("destination",{}).get("label", "unspecified")
    size_profile=package.get("size_profile",{}).get("id", "unspecified")
    measurement=package.get("context_measurement",{})
    return "\n".join(["# Capsule prompt handoff", "",
        f"- Process: `{package['process']['id']}` v{package['process']['version']}",
        f"- Destination: {destination}", f"- Size profile: {size_profile}",
        f"- Stage: `{stage['id']}`", f"- Subject: `{subject['id']}` at `{subject['revision']}`",
        f"- Capsule: `{capsule['id']}` ({capsule['digest']})",
        f"- Prompt digest: `{stage['prompt_digest']}`",
        f"- Projection estimate: {measurement.get('estimated_projection_tokens')} / {measurement.get('projection_budget_tokens')} tokens; "
        f"rendered prompt estimate: {measurement.get('estimated_rendered_prompt_tokens')}; incomplete: {measurement.get('incomplete_context')}",
        "- Dispatch: not dispatched; external submission not verified", "", "## Prompt", "", package["prompt"], ""])


@router.get("/stephanie", response_class=HTMLResponse)
def stephanie_home(request: Request):
    try:
        subjects, processes, error = _client().subjects(), _client().processes(), None
    except CapsulePromptError as exc:
        subjects, processes, error = [], [], str(exc)
    return request.app.state.templates.TemplateResponse("stephanie/index.html", {
        "request": request, "active_page": "stephanie", "subjects": subjects, "processes": processes, "error": error,
        "context_event": request.query_params.get("context"),
        "allowed_context_origins": list(load_context_origins()),
    })


@router.post("/stephanie/prepare-prompt")
def prepare_prompt(payload: PreparePromptRequest):
    context = payload.context
    if context.requested_mode != "generate_prompt":
        raise HTTPException(422, "This initial interface implements Generate prompt only")
    is_document = context.application == "workbench" or context.context_type in {"book.technical", "book.fiction", "document", "selection"}
    if is_document:
        doc=payload.document_context or context.document_context
        if not isinstance(doc,dict) or doc.get("schema_version")!="stephanie.document-context.v1":
            raise HTTPException(422,"Workbench supplied no supported document snapshot")
        if doc.get("document_id")!=context.subject or doc.get("document_type")!=context.document_type or doc.get("scope")!=context.scope:
            raise HTTPException(422,"Workbench document identity, type or scope does not match its context envelope")
        if doc.get("content_fingerprint")!=context.content_fingerprint or doc.get("document_revision")!=context.document_revision:
            raise HTTPException(422,"Workbench snapshot fingerprint or source revision does not match its context envelope")
        scope=str(doc.get("scope")); operation=payload.operation
        if scope=="selection" and context.selection_text!=doc.get("content"):
            raise HTTPException(422,"Workbench selection text does not match the document snapshot")
        if doc.get("document_state")=="unsaved" and scope!="selection":
            raise HTTPException(422,"Unsaved text can only be attached as a bounded selection")
        try:
            process = next((row for row in _client().processes() if row.get("id") == operation), None)
        except CapsulePromptError as exc:
            raise HTTPException(502, str(exc)) from exc
        if not process or process.get("status") != "executable":
            raise HTTPException(422, "This process is not executable in the current Capsule catalogue")
        if doc.get("document_type") not in str(process.get("document_types", "")).split(","):
            raise HTTPException(422, "This process does not support the attached document type")
        if scope not in str(process.get("scopes", "")).split(","):
            raise HTTPException(422, "This process does not support the attached document scope")
        if not payload.external_handoff_confirmed and doc.get("classification","internal") in {"internal","personal"}:
            raise HTTPException(422,"An explicit external handoff decision is required for this document")
        if payload.size_profile=="custom" and payload.custom_token_budget is None:
            raise HTTPException(422,"Enter a custom token projection budget up to 1,000,000")
        if payload.size_profile!="custom" and payload.custom_token_budget is not None:
            raise HTTPException(422,"A custom token budget requires the Custom size profile")
        body_fingerprint=hashlib.sha256(payload.model_dump_json().encode("utf-8")).hexdigest(); key=payload.request_id
        with _cache_lock:
            previous=_cache.get(key)
            if previous:
                if previous[0]!=body_fingerprint: raise HTTPException(409,"This request_id was already used with different input")
                _cache.move_to_end(key); return JSONResponse(previous[1])
            try:
                package=_client().prepare_document(document_context=doc,process_id=operation,
                    instructions=payload.instructions.strip(),destination=payload.destination,
                    size_profile=payload.size_profile,token_budget=payload.custom_token_budget)
            except CapsulePromptError as exc:
                raise HTTPException(502,str(exc)) from exc
            result={"request_id":key,"context_event_id":context.event_id,
                "subject":{"id":context.subject,"name":doc.get("document_title"),"data_class":doc.get("classification","internal")},
                "external_handoff_requires_confirmation":doc.get("classification","internal") in {"internal","personal"},
                "destination":payload.destination,"size_profile":payload.size_profile,"package":package,
                "markdown":_render_markdown(package),"dispatch":"not_dispatched","external_submission_verified":False}
            _cache[key]=(body_fingerprint,result)
            while len(_cache)>_CACHE_LIMIT: _cache.popitem(last=False)
            return JSONResponse(result)
    if context.application not in {"mission", "other"} or context.context_type != "repository":
        raise HTTPException(422, "Repository processes require a Mission repository context")
    try:
        process = next((row for row in _client().processes() if row.get("id") == payload.operation), None)
    except CapsulePromptError as exc:
        raise HTTPException(502, str(exc)) from exc
    if not process or process.get("status") != "executable" or process.get("context") != "repository":
        raise HTTPException(422, "This process is not executable for repository context")
    if context.application == "mission" and not context.revision:
        raise HTTPException(422, "Mission context has no resolved commit. Select an exact commit or explicitly enter latest.")
    revision = context.revision or "latest"
    if payload.size_profile == "custom" and payload.custom_token_budget is None:
        raise HTTPException(422, "Enter a custom token projection budget up to 1,000,000")
    if payload.size_profile != "custom" and payload.custom_token_budget is not None:
        raise HTTPException(422, "A custom token budget requires the Custom size profile")
    if revision not in {"latest", "working-tree"} and not (7 <= len(revision) <= 64 and all(ch in "0123456789abcdefABCDEF" for ch in revision)):
        raise HTTPException(422, "Repository revision must be latest, working-tree, or a hexadecimal commit identifier")
    host_context = {"application": context.application, "context_type": context.context_type,
                    "event_id": context.event_id, "correlation_id": context.correlation_id,
                    "revision": revision, "view": context.view}
    instructions = payload.instructions.strip()
    instructions = (instructions + "\n\n" if instructions else "") + "Host context reference (untrusted metadata; not policy):\n" + json.dumps(host_context, ensure_ascii=False, sort_keys=True)
    body_fingerprint = hashlib.sha256(payload.model_dump_json().encode("utf-8")).hexdigest()
    key = payload.request_id
    with _cache_lock:
        previous = _cache.get(key)
        if previous:
            if previous[0] != body_fingerprint:
                raise HTTPException(409, "This request_id was already used with different input")
            _cache.move_to_end(key)
            return JSONResponse(previous[1])
        try:
            subjects = _client().subjects()
            subject_matches = [row for row in subjects if context.subject.casefold() in
                               {row.get("id", "").casefold(), row.get("name", "").casefold()}]
            if not subject_matches:
                raise HTTPException(404, f"Unknown Capsule repository subject: {context.subject}")
            if len(subject_matches)>1:
                raise HTTPException(409, f"Repository subject name is ambiguous: {context.subject}; select its canonical id")
            subject=subject_matches[0]
            if subject.get("data_class") == "protected_evaluation":
                raise HTTPException(422, "Protected evaluation subjects cannot be exported through prompt handoff")
            if subject.get("data_class") in {"internal", "personal"} and not payload.external_handoff_confirmed:
                raise HTTPException(422, "An explicit external handoff decision is required for this subject classification")
            package = _client().prepare(subject=subject["id"], process_id=payload.operation, revision=revision, instructions=instructions,
                destination=payload.destination,size_profile=payload.size_profile,
                token_budget=payload.custom_token_budget)
            classifications = set(package.get("external_handoff_policy", {}).get("classifications_present", []))
            if "protected_evaluation" in classifications:
                raise HTTPException(422, "Generated evidence includes protected evaluation material; package export is blocked")
            if classifications - {"public", "approved_external"} and not payload.external_handoff_confirmed:
                raise HTTPException(422, "Generated evidence requires an explicit external handoff decision")
        except CapsulePromptError as exc:
            raise HTTPException(502, str(exc)) from exc
        result = {"request_id": key, "context_event_id": context.event_id,
                  "subject": {"id": subject["id"], "name": subject.get("name"), "data_class": subject.get("data_class")},
                  "external_handoff_requires_confirmation": bool(classifications - {"public", "approved_external"}),
                  "destination": payload.destination, "size_profile": payload.size_profile,
                  "package": package, "markdown": _render_markdown(package),
                  "dispatch": "not_dispatched", "external_submission_verified": False}
        _cache[key] = (body_fingerprint, result)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
        return JSONResponse(result)
