# stephanie/models/prompt_job.py
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column, Integer, String, Text, JSON, DateTime, Float, Index, UniqueConstraint
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB  # if on Postgres; else fallback to JSON
from sqlalchemy.types import Boolean
from stephanie.models.base import Base

from pydantic import BaseModel, Field, root_validator, validator

from .base import Base

# ──────────────────────────────────────────────────────────────────────────────
# Enums & small models
# ──────────────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    high = "high"
    normal = "normal"
    low = "low"


class ResponseFormat(str, Enum):
    text = "text"                # raw text
    json_object = "json_object"  # single JSON object
    json_lines = "json_lines"    # stream of JSONL objects


class SafetyLevel(str, Enum):
    # adjust to your policy ladder
    strict = "strict"
    standard = "standard"
    permissive = "permissive"


class RetryPolicy(BaseModel):
    max_retries: int = 2
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 15.0
    jitter: float = 0.2  # proportion of randomization around backoff


class CachePolicy(BaseModel):
    use_exact: bool = True
    use_approx: bool = True
    approx_threshold: float = 0.96  # cosine similarity threshold for cache reuse
    ttl_s: Optional[int] = None     # allow eviction policy to consider TTL


class RouteHints(BaseModel):
    group_key: Optional[str] = None       # co-locate jobs (e.g., same scorable/paper)
    shard_key: Optional[str] = None       # choose shard deterministically
    tenancy: Optional[str] = None         # e.g., "prod", "staging", "research"
    target_pool: Optional[str] = None     # e.g., "provider", "local-qwen", "remote-http"
    max_concurrency_hint: Optional[int] = None  # scheduler hint only


class SafetyPolicy(BaseModel):
    level: SafetyLevel = SafetyLevel.standard
    allow_tools: bool = False
    allow_web: bool = False


class CostTrack(BaseModel):
    provider: Optional[str] = None
    currency: str = "USD"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: bool = False
    latency_ms: Optional[float] = None
    total_cost: Optional[float] = None


class Attachment(BaseModel):
    """
    Minimal attachment wrapper for images, files, or audio.
    Provide either bytes_b64 or url (workers can fetch).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "image"           # "image" | "file" | "audio"
    mime: Optional[str] = None
    bytes_b64: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_bytes(cls, data: bytes, mime: str, type_: str = "image") -> "Attachment":
        return cls(type=type_, mime=mime, bytes_b64=base64.b64encode(data).decode("utf-8"))


# Use JSONB if available, else JSON
JSONType = JSONB if 'postgresql' in str(JSON.__module__).lower() else JSON


class PromptJobORM(Base):
    """
    Persistent record for async prompt jobs.
    Mirrors (and safely superset of) the Pydantic PromptJob model.
    """
    __tablename__ = "prompt_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identities
    job_id = Column(String, nullable=False, unique=True)       # external correlation
    scorable_id = Column(String, nullable=True)
    correlation_id = Column(String, nullable=True)
    parent_job_id = Column(String, nullable=True)
    trace_id = Column(String, nullable=True)

    # targeting
    model = Column(String, nullable=False)
    target = Column(String, nullable=False, default="auto")
    priority = Column(String, nullable=False, default="normal")  # high|normal|low

    # prompt content
    prompt_text = Column(Text, nullable=True)
    messages = Column(JSONType, nullable=True)
    system = Column(Text, nullable=True)
    tools = Column(JSONType, nullable=True)
    attachments = Column(JSONType, nullable=True)
    input_vars = Column(JSONType, nullable=True)

    # generation params
    gen_params = Column(JSONType, nullable=True)  # {max_tokens, temperature, top_p, stop, frequency_penalty, presence_penalty}

    # output & validation
    response_format = Column(String, nullable=False, default="text")   # text|json_object|json_lines
    prompt_schema = Column(JSONType, nullable=True)
    force_json = Column(Boolean, nullable=False, default=False)
    enforce_schema = Column(Boolean, nullable=False, default=False)
    stream = Column(Boolean, nullable=False, default=False)

    # routing/policy
    route = Column(JSONType, nullable=True)       # RouteHints
    retry = Column(JSONType, nullable=True)       # RetryPolicy
    cache = Column(JSONType, nullable=True)       # CachePolicy
    safety = Column(JSONType, nullable=True)      # SafetyPolicy
    return_topic = Column(String, nullable=True)

    # dedupe/cache
    dedupe_key = Column(String, nullable=True)
    cache_key = Column(String, nullable=True)     # exact cache signature
    signature = Column(String, nullable=True)
    version = Column(String, nullable=False, default="v2")

    # lifecycle
    status = Column(String, nullable=False, default="queued")   # queued|running|succeeded|failed|canceled
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    deadline_ts = Column(Float, nullable=True)   # epoch seconds
    ttl_s = Column(Integer, nullable=True)       # for retention/purge

    # outcomes
    result_text = Column(Text, nullable=True)    # final text or JSON string
    result_json = Column(JSONType, nullable=True)
    partial = Column(JSONType, nullable=True)    # streaming fragments, if kept
    error = Column(Text, nullable=True)

    # telemetry
    cost = Column(JSONType, nullable=True)       # CostTrack
    latency_ms = Column(Float, nullable=True)
    provider = Column(String, nullable=True)
    cache_hit = Column(Boolean, nullable=False, default=False)

    # free-form metadata
    metadata = Column(JSONType, nullable=True)

    __table_args__ = (
        UniqueConstraint("job_id", name="uq_prompt_jobs_job_id"),
        Index("idx_prompt_jobs_status_priority_created", "status", "priority", "created_at"),
        Index("idx_prompt_jobs_scorable", "scorable_id"),
        Index("idx_prompt_jobs_dedupe", "dedupe_key"),
        Index("idx_prompt_jobs_cache_key", "cache_key"),
    )

    # convenience
    def to_dict(self):
        return {
            "id": self.id,
            "job_id": self.job_id,
            "scorable_id": self.scorable_id,
            "model": self.model,
            "target": self.target,
            "priority": self.priority,
            "status": self.status,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "tools": self.tools,
            "attachments": self.attachments,
            "gen_params": self.gen_params,
            "response_format": self.response_format,
            "prompt_schema": self.prompt_schema,
            "force_json": self.force_json,
            "enforce_schema": self.enforce_schema,
            "stream": self.stream,
            "route": self.route,
            "retry": self.retry,
            "cache": self.cache,
            "safety": self.safety,
            "return_topic": self.return_topic,
            "dedupe_key": self.dedupe_key,
            "cache_key": self.cache_key,
            "signature": self.signature,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline_ts": self.deadline_ts,
            "ttl_s": self.ttl_s,
            "result_text": self.result_text,
            "result_json": self.result_json,
            "partial": self.partial,
            "error": self.error,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata,
        }

    def __repr__(self):
        return f"<PromptJob {self.job_id} {self.status} prio={self.priority} model={self.model}>"

    # ── Validators ────────────────────────────────────────────────────────────

    @root_validator
    def _require_prompt_or_messages(cls, values):
        if not values.get("prompt_text") and not values.get("messages"):
            raise ValueError("Provide either prompt_text or messages.")
        return values

    @validator("priority", pre=True, always=True)
    def _priority_default(cls, v):
        return v or Priority.normal

    # ── Keys & hashes ─────────────────────────────────────────────────────────

    def compute_dedupe_key(self) -> str:
        """
        A stable hash that captures: model, target, prompt content, params, and scorable_id.
        Use for exact-idempotency in the dispatcher.
        """
        payload = {
            "model": self.model,
            "target": self.target,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "scorable_id": self.scorable_id,
            "response_format": self.response_format,
            "prompt_schema": self.prompt_schema,
            "force_json": self.force_json,
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def compute_cache_key(self) -> str:
        """
        Cache key for exact hits. (Approx cache uses separate ANN index.)
        """
        base = {
            "model": self.model,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "response_format": self.response_format,
            "schema": self.schema,
        }
        s = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(("CACHE:" + s).encode("utf-8")).hexdigest()

    # ── Provider payload helpers ──────────────────────────────────────────────

    def to_openai_payload(self) -> Dict[str, Any]:
        """
        Normalize to an OpenAI-compatible payload; workers can adapt for other providers.
        """
        base: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        # Remove None values for cleanliness
        base = {k: v for k, v in base.items() if v is not None}

        # messages vs. prompt
        if self.messages:
            base["messages"] = self._inject_system(self.messages, self.system)
        else:
            # text-only legacy path
            content = self.prompt_text or ""
            if self.system:
                base["messages"] = [{"role": "system", "content": self.system},
                                    {"role": "user", "content": content}]
            else:
                base["messages"] = [{"role": "user", "content": content}]

        # tools (if any)
        if self.tools:
            base["tools"] = self.tools

        # response format
        if self.response_format == ResponseFormat.json_object and (self.force_json or self.schema):
            # OpenAI's json_object mode:
            base["response_format"] = {"type": "json_object"}
        elif self.response_format == ResponseFormat.text:
            pass  # default
        elif self.response_format == ResponseFormat.json_lines:
            # not native; worker can stream with '\n' delimited JSON
            base["stream"] = True

        # streaming flag
        if self.stream:
            base["stream"] = True

        return base

    @staticmethod
    def _inject_system(messages: List[Dict[str, Any]], system: Optional[str]) -> List[Dict[str, Any]]:
        if system:
            # Ensure a system message is the first if not already present
            if not messages or messages[0].get("role") != "system":
                return [{"role": "system", "content": system}] + messages
        return messages

    # ── Convenience & summaries ───────────────────────────────────────────────

    def short_summary(self) -> str:
        model = self.model
        tgt = self.target
        prio = self.priority
        if self.prompt_text:
            head = (self.prompt_text.strip().replace("\n", " "))[:160]
        elif self.messages:
            # show last user message head
            user_msgs = [m for m in self.messages if m.get("role") == "user"]
            head = (user_msgs[-1].get("content", "") if user_msgs else "")[:160]
        else:
            head = ""
        return f"[{prio}] {model}/{tgt} — {head}"

    # ── Legacy compat (v1 callers) ────────────────────────────────────────────

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Legacy shim used by some dispatchers; keep until all callers move to pydantic dict().
        """
        return {
            "job_id": self.job_id,
            "scorable_id": self.scorable_id,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "model": self.model,
            "target": self.target,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "response_format": self.response_format,
            "schema": self.schema,
            "stream": self.stream,
            "priority": self.priority,
            "return_topic": self.return_topic,
            "metadata": self.metadata,
        }

    # ── Housekeeping before publish ───────────────────────────────────────────

    def finalize_before_publish(self) -> None:
        """
        Ensure keys that dispatchers/workers rely on are set.
        """
        if not self.correlation_id:
            self.correlation_id = self.job_id
        if not self.trace_id:
            self.trace_id = self.job_id
        if not self.dedupe_key:
            self.dedupe_key = self.compute_dedupe_key()
        if self.ttl_s is None:
            self.ttl_s = 3600
