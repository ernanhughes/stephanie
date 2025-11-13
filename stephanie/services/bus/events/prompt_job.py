# stephanie/services/bus/events/prompt_job.py
from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
)
from stephanie.constants import PROMPT_RESULT_TMPL

import logging

from stephanie.utils.json_sanitize import dumps_safe
log = logging.getLogger(__name__)

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
    approx_threshold: float = (
        0.96  # cosine similarity threshold for cache reuse
    )
    ttl_s: Optional[int] = None  # allow eviction policy to consider TTL


class RouteHints(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    group_key: Optional[str] = (
        None  # co-locate jobs (e.g., same scorable/paper)
    )
    shard_key: Optional[str] = None  # choose shard deterministically
    tenancy: Optional[str] = None  # e.g., "prod", "staging", "research"
    target_pool: Optional[str] = (
        None  # e.g., "provider", "local-qwen", "remote-http"
    )
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
# Main job model
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

    target: str = "auto"  # logical target (router decides real executor)
    route: RouteHints = Field(default_factory=RouteHints)
    priority: Priority = Priority.normal

    # Core prompt fields
    prompt_text: Optional[str] = None  # simple prompt
    messages: Optional[List[Dict[str, Any]]] = (
        None  # [{"role": "...", "content": "..."}]
    )
    system: Optional[str] = None  # optional system override
    input_vars: Optional[Dict[str, Any]] = (
        None  # templating vars (if a template upstream)
    )
    tools: Optional[List[Dict[str, Any]]] = None  # tool schemas (OpenAI-style)
    attachments: Optional[List[Attachment]] = None

    # Generation params (provider-agnostic)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Output & validation
    response_format: ResponseFormat = ResponseFormat.text
    force_json: bool = (
        False  # request strict JSON decoding on worker if supported
    )
    schema: Optional[Dict[str, Any]] = (
        None  # JSON schema for validation (if response_format != text)
    )
    stream: bool = False  # streaming response (worker can split into chunks)
    enforce_schema: bool = False  # if true, reject outputs that don't validate

    # Timing & lifecycle
    created_ts: float = Field(default_factory=lambda: time.time())
    deadline_ts: Optional[float] = None  # absolute deadline (epoch seconds)
    ttl_s: Optional[int] = (
        3600  # keep result subjects alive (JetStream / Redis)
    )
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    cache: CachePolicy = Field(default_factory=CachePolicy)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)

    # Bus return path
    return_topic: Optional[str] = None

    # Dedupe & signatures
    dedupe_key: Optional[str] = None  # if set, dispatcher can drop duplicates
    signature: Optional[str] = (
        None  # optional HMAC/signature for zero-trust workers
    )
    version: str = "v2"  # bump when changing semantics

    # Free-form metadata (saved & echoed back by workers)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Telemetry (filled by worker; safe to leave None at publish time)
    cost: Optional[CostTrack] = None

    # ── Validators (Pydantic v2) ──────────────────────────────────────────────

    @model_validator(mode="after")
    def _require_prompt_or_messages(self):
        """
        Enforce that at least one of (prompt_text, messages) is provided.
        If both are provided, prefer 'messages' downstream but allow it.
        """
        if not self.prompt_text and not self.messages:
            raise ValueError("Provide either prompt_text or messages.")
        return self

    @field_validator("priority", mode="before")
    @classmethod
    def _priority_default(cls, v):
        # accept None, Enum, or str ("high"/"normal"/"low")
        if v is None:
            return Priority.normal
        if isinstance(v, Priority):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            return Priority(s) if s in {"high","normal","low"} else Priority.normal
        return Priority.normal

    @field_validator("messages")
    @classmethod
    def _normalize_messages(cls, v):
        # Treat [] as None; helps the "require" validator behave as expected.
        if v is not None and len(v) == 0:
            return None
        return v

    # ── Keys & hashes ─────────────────────────────────────────────────────────

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
            "response_format": (self.response_format.value if isinstance(self.response_format, ResponseFormat) else str(self.response_format)),
            "schema": self.schema,
            "force_json": self.force_json,
            "priority": (self.priority.value if isinstance(self.priority, Priority) else str(self.priority)),
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def compute_cache_key(self) -> str:
        base = {
            "model": self.model,
            "prompt_text": self.prompt_text,
            "messages": self._stable_messages_for_hash(),
            "system": self.system,
            "response_format": (self.response_format.value if isinstance(self.response_format, ResponseFormat) else str(self.response_format)),
            "schema": self.schema,
        }
        s = json.dumps(base, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(("CACHE:" + s).encode("utf-8")).hexdigest()

    # ------------------ Bus / provider payloads ------------------

    def to_json(self) -> str:
        """Stable JSON for bus publish (enums flattened)."""
        payload = self._dump_plain(exclude_none=True)
        return dumps_safe(payload, ensure_ascii=False, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    def to_bus_payload(self) -> Dict[str, Any]:
        """
        Explicit bus-ready dict (for ZeroMQ/NATS). Enums -> str values.
        Useful when the bus takes dicts (InProc/Hybrid) or when you want to
        run your own encoder.
        """
        return self._dump_plain(exclude_none=True)

    @classmethod
    def from_json(cls, s: str) -> "PromptJob":
        """Worker-side convenience for echoes/tests."""
        data = json.loads(s)
        return cls.model_validate(data)

    def to_openai_payload(self) -> Dict[str, Any]:
        # (unchanged except pulling from flattened enums already)
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
        if self.response_format == ResponseFormat.json_object and (self.force_json or self.schema):
            base["response_format"] = {"type": "json_object"}
        elif self.response_format == ResponseFormat.json_lines:
            base["stream"] = True
        if self.stream:
            base["stream"] = True
        return base

    # ------------------ finalize_before_publish ------------------

    def finalize_before_publish(self) -> None:
        self.job_id = self.job_id or str(uuid.uuid4())
        if not self.return_topic or self.return_topic.strip() in ("results.prompts.", "results."):
            self.return_topic = PROMPT_RESULT_TMPL.format(job=self.job_id)
        if not self.correlation_id:
            self.correlation_id = self.job_id
        if not self.trace_id:
            self.trace_id = self.job_id
        if not self.dedupe_key:
            self.dedupe_key = self.compute_dedupe_key()
        if self.ttl_s is None:
            self.ttl_s = 3600

        # 🔎 Better logging (don’t show signature as subject)
        try:
            rt = self.return_topic
            rk = getattr(self.route, "group_key", None) if isinstance(self.route, RouteHints) else None
            tp = getattr(self.route, "target_pool", None) if isinstance(self.route, RouteHints) else None
            log.info(
                "PromptJob ready -> job=%s model=%s target=%s priority=%s ret=%s route.group=%s route.pool=%s",
                self.job_id,
                (self.model.get("name") if isinstance(self.model, dict) else str(self.model)),
                self.target,
                (self.priority.value if isinstance(self.priority, Priority) else str(self.priority)),
                rt,
                rk,
                tp,
            )
        except Exception:
            # Never fail finalize on logging
            pass

    # ------------------ legacy shim ------------------

    def to_kwargs(self) -> Dict[str, Any]:
        d = self._dump_plain(exclude_none=True)
        # Keep expected legacy keys—ensure enums are values
        out = {
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
            "schema": d.get("schema"),
            "stream": d.get("stream"),
            "priority": d.get("priority"),
            "return_topic": d.get("return_topic"),
            "metadata": d.get("metadata"),
        }
        return {k: v for k, v in out.items() if v is not None}

    def _stable_messages_for_hash(self) -> Any:
        """
        Make messages hashing robust:
        - ensure system injection equivalence (messages vs prompt+system),
        - drop ephemeral fields if ever present,
        - keep order stable.
        """
        msgs = self.messages or []
        if msgs and self.system:
            # Ensure a leading system is present for semantic stability.
            if msgs[0].get("role") != "system":
                msgs = [{"role": "system", "content": self.system}] + msgs
        return msgs

    # ── Provider payload helpers ──────────────────────────────────────────────

    def to_json(self) -> str:
        """Stable JSON for bus publish (v2-safe)."""
        payload = self.model_dump(exclude_none=True)
        # separators trims bytes on the wire; keep unicode with ensure_ascii=False
        return dumps_safe(payload, ensure_ascii=False, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    def _enum_to_value(self, v):
        # Convert Enum -> value recursively inside dict/list payloads
        from enum import Enum as _Enum
        if isinstance(v, _Enum):
            return v.value
        if isinstance(v, dict):
            return {k: self._enum_to_value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._enum_to_value(x) for x in v]
        return v

    def _dump_plain(self, *, exclude_none: bool = True) -> Dict[str, Any]:
        """
        model_dump + Enum flattening so bus payloads stay clean strings.
        """
        d = self.model_dump(exclude_none=exclude_none)
        return self._enum_to_value(d)

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
                base["messages"] = [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": content},
                ]
            else:
                base["messages"] = [{"role": "user", "content": content}]

        # tools (if any)
        if self.tools:
            base["tools"] = self.tools

        # response format
        if self.response_format == ResponseFormat.json_object and (
            self.force_json or self.schema
        ):
            base["response_format"] = {"type": "json_object"}
        elif self.response_format == ResponseFormat.json_lines:
            # not native; worker can stream with '\n' delimited JSON
            base["stream"] = True
        # else text → default

        # streaming flag
        if self.stream:
            base["stream"] = True

        return base

    @staticmethod
    def _inject_system(
        messages: List[Dict[str, Any]], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        if system:
            # Ensure a system message is the first if not already present
            if not messages or messages[0].get("role") != "system":
                return [{"role": "system", "content": system}] + messages
        return messages

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
            # show last user message head
            user_msgs = [m for m in self.messages if m.get("role") == "user"]
            head = (user_msgs[-1].get("content", "") if user_msgs else "")[
                :160
            ]
        else:
            head = ""
        return f"[{prio}] {model}/{tgt} — {head}"

    # ── Legacy compat (v1 callers) ────────────────────────────────────────────

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Legacy shim used by some dispatchers; keep until all callers move to Pydantic v2 APIs.
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
        self.job_id = self.job_id or str(uuid.uuid4())

        # Keep topic SHORT so the bus prefixes once (→ "stephanie.results.prompts.<id>")
        if not self.return_topic or self.return_topic.strip() in ("results.prompts.", "results."):
            self.return_topic = PROMPT_RESULT_TMPL.format(job=self.job_id)
        if not self.correlation_id:
            self.correlation_id = self.job_id
        if not self.trace_id:
            self.trace_id = self.job_id
        if not self.dedupe_key:
            self.dedupe_key = self.compute_dedupe_key()
        if self.ttl_s is None:
            self.ttl_s = 3600

        log.info(
            "PromptJob -> job=%s subject=%s ret=%s",
            self.job_id, self.signature, self.return_topic
        )
