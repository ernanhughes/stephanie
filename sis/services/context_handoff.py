"""Versioned, bounded context references exchanged with host applications."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StephanieContextHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["stephanie.context.v1"] = "stephanie.context.v1"
    event_type: Literal["stephanie.open"] = "stephanie.open"
    event_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    correlation_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:-]+$")
    application: Literal["mission", "workbench", "writer", "other"]
    context_type: Literal["repository", "book.technical", "book.fiction", "document", "selection"]
    subject: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    revision: str | None = Field(default=None, max_length=160, pattern=r"^(latest|working-tree|[0-9a-fA-F]{7,64})$")
    view: str | None = Field(default=None, max_length=120)
    document_id: str | None = Field(default=None, max_length=160)
    document_revision: str | None = Field(default=None, max_length=160)
    chapter_id: str | None = Field(default=None, max_length=160)
    section_id: str | None = Field(default=None, max_length=160)
    selection_id: str | None = Field(default=None, max_length=160)
    selection_text: str | None = Field(default=None, max_length=4000)
    scope: Literal["document", "book", "chapter", "section", "selection"] | None = None
    document_type: str | None = Field(default=None, max_length=80)
    content_fingerprint: str | None = Field(default=None, max_length=80)
    document_context: dict | None = None
    document_state: Literal["saved", "unsaved", "not_applicable"] = "not_applicable"
    requested_mode: Literal["generate_prompt", "chat", "process", "inspect"] = "generate_prompt"
    observed_at: datetime | None = None

