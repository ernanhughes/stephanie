from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    max_retries: int = 2
    backoff_initial_ms: int = 250
    backoff_factor: float = 2.0

class PromptJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scorable_id: str
    prompt_text: str
    model: str = "gpt-4o-mini"
    return_topic: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_ts: float = Field(default_factory=lambda: time.time())
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    idempotency_key: Optional[str] = None


