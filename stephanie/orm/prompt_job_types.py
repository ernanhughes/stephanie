# stephanie/orm/prompt_job_types.py
from __future__ import annotations

import base64
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Priority(str, Enum):
    high = "high"
    normal = "normal"
    low = "low"

class ResponseFormat(str, Enum):
    text = "text"
    json_object = "json_object"
    json_lines = "json_lines"

class SafetyLevel(str, Enum):
    strict = "strict"
    standard = "standard"
    permissive = "permissive"

class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    max_retries: int = 2
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 15.0
    jitter: float = 0.2

class CachePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    use_exact: bool = True
    use_approx: bool = True
    approx_threshold: float = 0.96
    ttl_s: Optional[int] = None

class RouteHints(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)
    group_key: Optional[str] = None
    shard_key: Optional[str] = None
    tenancy: Optional[str] = None
    target_pool: Optional[str] = None
    max_concurrency_hint: Optional[int] = None

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
    model_config = ConfigDict(extra="forbid", validate_default=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "image"
    mime: Optional[str] = None
    bytes_b64: Optional[str] = None
    url: Optional[str] = None

    @classmethod
    def from_bytes(cls, data: bytes, mime: str, type_: str = "image") -> "Attachment":
        return cls(type=type_, mime=mime, bytes_b64=base64.b64encode(data).decode("utf-8"))
