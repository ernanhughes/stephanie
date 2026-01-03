# stephanie/services/bus/events/prompt_job.py
from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)
# ORM imports
from sqlalchemy import (JSON, Column, DateTime, Float, Index, Integer, String,
                        Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.types import Boolean

from stephanie.constants import PROMPT_RESULT_TMPL
from stephanie.orm.base import Base  # declarative base
from stephanie.utils.hash_utils import hash_text
from stephanie.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)

# Use JSONB if Postgres, else JSON
JSONType = JSONB if "postgresql" in str(JSON.__module__).lower() else JSON

# ──────────────────────────────────────────────────────────────────────────────
# Enums & small models
# ──────────────────────────────────────────────────────────────────────────────


class Priority(str, Enum):
    high = "high"
    normal = "normal"
    low = "low"


class ResponseFormat(str, Enum):
    text = "text"  # raw text
    json_object = "json_object"  # single JSON object
    json_lines = "json_lines"  # stream of JSONL objects


class SafetyLevel(str, Enum):
    strict = "strict"
    standard = "standard"
    permissive = "permissive"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    max_retries: int = 2
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 15.0
    jitter: float = 0.2  # proportion of randomization around backoff


class CachePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    use_exact: bool = True
    use_approx: bool = True
    approx_threshold: float = 0.96  # cosine similarity threshold for cache reuse
    ttl_s: Optional[int] = None  # allow eviction policy to consider TTL


class RouteHints(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    group_key: Optional[str] = None  # co-locate jobs (e.g., same scorable/paper)
    shard_key: Optional[str] = None  # choose shard deterministically
    tenancy: Optional[str] = None  # e.g., "prod", "staging", "research"
    target_pool: Optional[str] = None  # e.g., "provider", "local-qwen", "remote-http"
    max_concurrency_hint: Optional[int] = None  # scheduler hint only


class SafetyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    level: SafetyLevel = SafetyLevel.standard
    allow_tools: bool = False
    allow_web: bool = False


class CostTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

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

    model_config = ConfigDict(extra="forbid", validate_default=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "image"  # "image" | "file" | "audio"
    mime: Optional[str] = None
    bytes_b64: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_bytes(
        cls, data: bytes, mime: str, type_: str = "image"
    ) -> "Attachment":
        return cls(
            type=type_,
            mime=mime,
            bytes_b64=base64.b64encode(data).decode("utf-8"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic PromptJob event model
# ──────────────────────────────────────────────────────────────────────────────


class PromptJob(BaseModel):
    """
    Provider-agnostic prompt job that can carry either:
      - prompt_text (simple completion), or
      - messages (chat format).
    Exactly one of these must be present at minimum.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        validate_assignment=True,
        use_enum_values=False,  # keep Enum types in Python space; we serialize explicitly
    )

    # Identity & routing
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scorable_id: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Targeting
    model: dict = {
        "name": "ollama/qwen:0.5b",
        "api_base": "http://localhost:11434",
        "api_key": None,
    }
    target: str = "auto"
    route: RouteHints = Field(default_factory=RouteHints)
    priority: Priority = Priority.normal

    # Core prompt fields
    prompt_text: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    system: Optional[str] = None
    input_vars: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Attachment]] = None

    # Generation params
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Output & validation
    response_format: ResponseFormat = ResponseFormat.text
    force_json: bool = False
    prompt_schema: Optional[Dict[str, Any]] = None
    stream: bool = False
    enforce_schema: bool = False

    # Timing & lifecycle
    created_ts: float = Field(default_factory=lambda: time.time())
    deadline_ts: Optional[float] = None
    ttl_s: Optional[int] = 3600
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    cache: CachePolicy = Field(default_factory=CachePolicy)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)

    # Bus return path
    return_topic: Optional[str] = None

    # Dedupe & signatures
    dedupe_key: Optional[str] = None
    signature: Optional[str] = None
    version: str = "v2"

    # Free-form metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Telemetry (filled by worker)
    cost: Optional[CostTrack] = None

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _require_prompt_or_messages(self):
        if not self.prompt_text and not self.messages:
            raise ValueError("Provide either prompt_text or messages.")
        return self

    @field_validator("priority", mode="before")
    @classmethod
    def _priority_default(cls, v):
        if v is None:
            return Priority.normal
        if isinstance(v, Priority):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            return Priority(s) if s in {"high", "normal", "low"} else Priority.normal
        return Priority.normal

    @field_validator("messages")
    @classmethod
    def _normalize_messages(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

    # ── Stable hashing ───────────────────────────────────────────────────────

    def _stable_messages_for_hash(self) -> Any:
        """
        Make messages hashing robust:
        - ensure system injection equivalence (messages vs prompt+system),
        - drop ephemeral fields if ever present,
        - keep order stable.
        """
        msgs = self.messages or []
        if msgs and self.system:
            if msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": self.system}] + msgs
        return msgs

    def compute_dedupe_key(self) -> str:
        payload = {
            "model": self.model,
            "target": self.target,
            "prompt_text": self.prompt_text,
            "messages": self._stable_messages_for_hash(),
            "system": self.system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop,
            "scorable_id": self.scorable_id,
            "response_format": (
                self.response_format.value
                if isinstance(self.response_format, ResponseFormat)
                else str(self.response_format)
            ),
            "prompt_schema": self.prompt_schema,
            "force_json": self.force_json,
            "priority": (
                self.priority.value
                if isinstance(self.priority, Priority)
                else str(self.priority)
            ),
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hash_text(s)

    def compute_cache_key(self) -> str:
        base = {
            "model": self.model,
            "prompt_text": self.prompt_text,
            "messages": self._stable_messages_for_hash(),
            "system": self.system,
            "response_format": (
                self.response_format.value
                if isinstance(self.response_format, ResponseFormat)
                else str(self.response_format)
            ),
            "prompt_schema": self.prompt_schema,
        }
        s = json.dumps(base, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hash_text("CACHE:" + s)

    # ── Provider payload helpers ──────────────────────────────────────────────

    @staticmethod
    def _inject_system(
        messages: List[Dict[str, Any]], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        if system:
            if not messages or messages[0].get("role") != "system":
                return [{"role": "system", "content": system}] + messages
        return messages

    def to_openai_payload(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        base = {k: v for k, v in base.items() if v is not None}

        if self.messages:
            base["messages"] = self._inject_system(self.messages, self.system)
        else:
            content = self.prompt_text or ""
            if self.system:
                base["messages"] = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": content},
                ]
            else:
                base["messages"] = [{"role": "user", "content": content}]

        if self.tools:
            base["tools"] = self.tools

        if self.response_format == ResponseFormat.json_object and (
            self.force_json or self.prompt_schema
        ):
            base["response_format"] = {"type": "json_object"}
        elif self.response_format == ResponseFormat.json_lines:
            base["stream"] = True

        if self.stream:
            base["stream"] = True

        return base

    # ── Dump helpers for bus ─────────────────────────────────────────────────

    def _enum_to_value(self, v):
        from enum import Enum as _Enum

        if isinstance(v, _Enum):
            return v.value
        if isinstance(v, dict):
            return {k: self._enum_to_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._enum_to_value(x) for x in v]
        return v

    def _dump_plain(self, *, exclude_none: bool = True) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=exclude_none)
        return self._enum_to_value(d)

    def to_json(self) -> str:
        payload = self._dump_plain(exclude_none=True)
        return dumps_safe(payload, ensure_ascii=False, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    def to_bus_payload(self) -> Dict[str, Any]:
        return self._dump_plain(exclude_none=True)

    @classmethod
    def from_json(cls, s: str) -> "PromptJob":
        data = json.loads(s)
        return cls.model_validate(data)

    # ── Convenience & summaries ───────────────────────────────────────────────

    def is_chat(self) -> bool:
        return bool(self.messages)

    def short_summary(self) -> str:
        model = self.model
        tgt = self.target
        prio = self.priority
        if self.prompt_text:
            head = (self.prompt_text.strip().replace("\n", " "))[:160]
        elif self.messages:
            user_msgs = [m for m in self.messages if m.get("role") == "user"]
            head = (user_msgs[-1].get("content", "") if user_msgs else "")[:160]
        else:
            head = ""
        return f"[{prio}] {model}/{tgt} — {head}"

    # ── Legacy compat ─────────────────────────────────────────────────────────

    def to_kwargs(self) -> Dict[str, Any]:
        d = self._dump_plain(exclude_none=True)
        return {
            "job_id": d.get("job_id"),
            "scorable_id": d.get("scorable_id"),
            "prompt_text": d.get("prompt_text"),
            "messages": d.get("messages"),
            "system": d.get("system"),
            "model": d.get("model"),
            "target": d.get("target"),
            "max_tokens": d.get("max_tokens"),
            "temperature": d.get("temperature"),
            "top_p": d.get("top_p"),
            "stop": d.get("stop"),
            "response_format": d.get("response_format"),
            "prompt_schema": d.get("prompt_schema"),
            "stream": d.get("stream"),
            "priority": d.get("priority"),
            "return_topic": d.get("return_topic"),
            "metadata": d.get("metadata"),
        }

    # ── finalize_before_publish ───────────────────────────────────────────────

    def finalize_before_publish(self) -> None:
        self.job_id = self.job_id or str(uuid.uuid4())
        if not self.return_topic or self.return_topic.strip() in (
            "results.prompts.",
            "results.",
        ):
            self.return_topic = PROMPT_RESULT_TMPL.format(job=self.job_id)
        if not self.correlation_id:
            self.correlation_id = self.job_id
        if not self.trace_id:
            self.trace_id = self.job_id
        if not self.dedupe_key:
            self.dedupe_key = self.compute_dedupe_key()
        if self.ttl_s is None:
            self.ttl_s = 3600

        try:
            rk = (
                getattr(self.route, "group_key", None)
                if isinstance(self.route, RouteHints)
                else None
            )
            tp = (
                getattr(self.route, "target_pool", None)
                if isinstance(self.route, RouteHints)
                else None
            )
            log.info(
                "PromptJob ready -> job=%s model=%s target=%s priority=%s ret=%s route.group=%s route.pool=%s",
                self.job_id,
                (self.model.get("name") if isinstance(self.model, dict) else str(self.model)),
                self.target,
                (self.priority.value if isinstance(self.priority, Priority) else str(self.priority)),
                self.return_topic,
                rk,
                tp,
            )
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# ORM PromptJobORM – DB persistence for prompt_jobs
# ──────────────────────────────────────────────────────────────────────────────


class PromptJobORM(Base):
    """
    Persistent record for async prompt jobs.

    Mirrors (and is a storage superset of) the Pydantic PromptJob event model
    defined above.
    """

    __tablename__ = "prompt_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # identities
    job_id = Column(String, nullable=False, unique=True)  # external correlation
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

    # generation params (flattened in Pydantic, stored as JSON here)
    # {max_tokens, temperature, top_p, stop, frequency_penalty, presence_penalty}
    gen_params = Column(JSONType, nullable=True)

    # output & validation
    response_format = Column(
        String, nullable=False, default="text"
    )  # "text"|"json_object"|"json_lines"
    prompt_schema = Column(JSONType, nullable=True)
    force_json = Column(Boolean, nullable=False, default=False)
    enforce_schema = Column(Boolean, nullable=False, default=False)
    stream = Column(Boolean, nullable=False, default=False)

    # routing/policy
    route = Column(JSONType, nullable=True)  # RouteHints
    retry = Column(JSONType, nullable=True)  # RetryPolicy
    cache = Column(JSONType, nullable=True)  # CachePolicy
    safety = Column(JSONType, nullable=True)  # SafetyPolicy
    return_topic = Column(String, nullable=True)

    # dedupe/cache
    dedupe_key = Column(String, nullable=True)
    cache_key = Column(String, nullable=True)  # exact cache signature
    signature = Column(String, nullable=True)
    version = Column(String, nullable=False, default="v2")

    # lifecycle
    status = Column(
        String, nullable=False, default="queued"
    )  # queued|running|succeeded|failed|canceled
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    deadline_ts = Column(Float, nullable=True)  # epoch seconds
    ttl_s = Column(Integer, nullable=True)  # for retention/purge

    # outcomes
    result_text = Column(Text, nullable=True)  # final text or JSON string
    result_json = Column(JSONType, nullable=True)
    partial = Column(JSONType, nullable=True)  # streaming fragments, if kept
    error = Column(Text, nullable=True)

    # telemetry
    cost = Column(JSONType, nullable=True)  # CostTrack
    latency_ms = Column(Float, nullable=True)
    provider = Column(String, nullable=True)
    cache_hit = Column(Boolean, nullable=False, default=False)

    # free-form metadata
    # NOTE: attribute name 'metadata' is reserved by SQLAlchemy,
    # so we use 'meta' as the Python attribute and keep the column
    # name "metadata" for backwards compatibility.
    meta = Column("metadata", JSONType, nullable=True)

    __table_args__ = (
        UniqueConstraint("job_id", name="uq_prompt_jobs_job_id"),
        Index(
            "idx_prompt_jobs_status_priority_created",
            "status",
            "priority",
            "created_at",
        ),
        Index("idx_prompt_jobs_scorable", "scorable_id"),
        Index("idx_prompt_jobs_dedupe", "dedupe_key"),
        Index("idx_prompt_jobs_cache_key", "cache_key"),
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"<PromptJobORM {self.job_id} {self.status} "
            f"prio={self.priority} model={self.model}>"
        )

    def _gp(self) -> Dict[str, Any]:
        """Convenience accessor for generation params."""
        return self.gen_params or {}

    # ── Serialization ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
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
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
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
            "metadata": self.meta,
        }

    # ── Keys & hashes ---------------------------------------------------------

    def compute_dedupe_key(self) -> str:
        """
        A stable hash that captures: model, target, prompt content, params, and scorable_id.

        Used for exact-idempotency in the dispatcher.

        NOTE: This is primarily for backfills or sanity checks. The canonical
        dedupe_key is computed on the Pydantic PromptJob *before* persistence.
        """
        gp = self._gp()
        payload = {
            "model": self.model,
            "target": self.target,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "max_tokens": gp.get("max_tokens"),
            "temperature": gp.get("temperature"),
            "top_p": gp.get("top_p"),
            "stop": gp.get("stop"),
            "scorable_id": self.scorable_id,
            "response_format": self.response_format,
            "prompt_schema": self.prompt_schema,
            "force_json": self.force_json,
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hash_text(s)

    def compute_cache_key(self) -> str:
        """
        Cache key for exact hits. (Approx cache uses separate ANN index.)

        As with dedupe_key, the canonical value is computed at the Pydantic
        PromptJob layer; this method exists for backfills and debugging.
        """
        base = {
            "model": self.model,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "response_format": self.response_format,
            "prompt_schema": self.prompt_schema,
        }
        s = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return hash_text("CACHE:" + s)

    # ── Provider payload helpers ---------------------------------------------

    def to_openai_payload(self) -> Dict[str, Any]:
        """
        Normalize to an OpenAI-compatible payload; workers can adapt for other providers.

        This is mainly useful for tools that reconstruct a provider call from
        a stored job. For new jobs, use the Pydantic PromptJob.to_openai_payload().
        """
        gp = self._gp()
        base: Dict[str, Any] = {
            "model": self.model,
            "temperature": gp.get("temperature"),
            "top_p": gp.get("top_p"),
            "max_tokens": gp.get("max_tokens"),
            "stop": gp.get("stop"),
            "frequency_penalty": gp.get("frequency_penalty"),
            "presence_penalty": gp.get("presence_penalty"),
        }
        base = {k: v for k, v in base.items() if v is not None}

        if self.messages:
            base["messages"] = self._inject_system(self.messages, self.system)
        else:
            content = self.prompt_text or ""
            if self.system:
                base["messages"] = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": content},
                ]
            else:
                base["messages"] = [{"role": "user", "content": content}]

        if self.tools:
            base["tools"] = self.tools

        if self.response_format == "json_object" and (
            self.force_json or self.prompt_schema
        ):
            base["response_format"] = {"type": "json_object"}
        elif self.response_format == "json_lines":
            base["stream"] = True
        # else: "text" → default

        if self.stream:
            base["stream"] = True

        return base

    @staticmethod
    def _inject_system(
        messages: List[Dict[str, Any]], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        if system:
            if not messages or messages[0].get("role") != "system":
                return [{"role": "system", "content": system}] + messages
        return messages

    # ── Convenience & summaries ----------------------------------------------

    def short_summary(self) -> str:
        model = self.model
        tgt = self.target
        prio = self.priority
        if self.prompt_text:
            head = (self.prompt_text.strip().replace("\n", " "))[:160]
        elif self.messages:
            user_msgs = [m for m in self.messages if m.get("role") == "user"]
            head = (user_msgs[-1].get("content", "") if user_msgs else "")[:160]
        else:
            head = ""
        return f"[{prio}] {model}/{tgt} — {head}"

    # ── Legacy compat (v1 callers) -------------------------------------------

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Legacy shim used by some dispatchers; keep until all callers move to
        newer Pydantic-based APIs.

        Gen params are unpacked from gen_params for backwards compatibility.
        """
        gp = self._gp()
        return {
            "job_id": self.job_id,
            "scorable_id": self.scorable_id,
            "prompt_text": self.prompt_text,
            "messages": self.messages,
            "system": self.system,
            "model": self.model,
            "target": self.target,
            "max_tokens": gp.get("max_tokens"),
            "temperature": gp.get("temperature"),
            "top_p": gp.get("top_p"),
            "stop": gp.get("stop"),
            "response_format": self.response_format,
            "prompt_schema": self.prompt_schema,
            "stream": self.stream,
            "priority": self.priority,
            "return_topic": self.return_topic,
            "metadata": self.meta,
        }
