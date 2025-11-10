# stephanie/services/bus/events/prompt_job.py
from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


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


# ──────────────────────────────────────────────────────────────────────────────
# Main job model
# ──────────────────────────────────────────────────────────────────────────────

class PromptJob(BaseModel):
    # Identity & routing
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scorable_id: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Targeting
    model: str = "gpt-4o-mini"
    target: str = "auto"  # logical target (router decides real executor)
    route: RouteHints = Field(default_factory=RouteHints)
    priority: Priority = Priority.normal

    # Core prompt fields
    prompt_text: Optional[str] = None                 # simple prompt
    messages: Optional[List[Dict[str, Any]]] = None   # chat format [{"role": "...", "content": "..."}]
    system: Optional[str] = None                      # optional system override
    input_vars: Optional[Dict[str, Any]] = None       # templating vars (if a template upstream)
    tools: Optional[List[Dict[str, Any]]] = None      # tool schemas (OpenAI-style)
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
    force_json: bool = False                 # request strict JSON decoding on worker if supported
    schema: Optional[Dict[str, Any]] = None  # JSON schema for validation (if response_format != text)
    stream: bool = False                     # streaming response (worker can split into chunks)
    enforce_schema: bool = False             # if true, reject outputs that don't validate

    # Timing & lifecycle
    created_ts: float = Field(default_factory=lambda: time.time())
    deadline_ts: Optional[float] = None      # absolute deadline (epoch seconds)
    ttl_s: Optional[int] = 3600              # keep result subjects alive (JetStream / Redis)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    cache: CachePolicy = Field(default_factory=CachePolicy)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)

    # Bus return path
    return_topic: Optional[str] = None

    # Dedupe & signatures
    dedupe_key: Optional[str] = None         # if set, dispatcher can drop duplicates
    signature: Optional[str] = None          # optional HMAC/signature for zero-trust workers
    version: str = "v2"                      # bump when changing semantics

    # Free-form metadata (saved & echoed back by workers)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Telemetry (filled by worker; safe to leave None at publish time)
    cost: Optional[CostTrack] = None

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
            "schema": self.schema,
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
