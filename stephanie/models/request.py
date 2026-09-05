# stephanie/models/request.py
"""Canonical request envelope (§7 of the Stage 1 spec)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from stephanie.models.model import Model


@dataclass(frozen=True)
class ModelMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    schema: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelRequest:
    model: Union[Model, str]

    prompt: Optional[str] = None
    messages: Optional[Sequence[ModelMessage]] = None

    system_prompt: Optional[str] = None

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    tools: Optional[Sequence[ToolDefinition]] = None
    response_schema: Optional[Mapping[str, Any]] = None

    # Preserved verbatim for later portfolio learning (model × task_type).
    # The runtime must not interpret these; callers own their meaning.
    task_type: Optional[str] = None
    purpose: Optional[str] = None

    trace_id: Optional[str] = None
    parent_trace_id: Optional[str] = None

    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def with_model(self, model: Model) -> "ModelRequest":
        from dataclasses import replace

        return replace(self, model=model)

    def to_messages(self) -> list[dict[str, str]]:
        if self.messages:
            out = [{"role": m.role, "content": m.content} for m in self.messages]
        elif self.prompt is not None:
            out = [{"role": "user", "content": self.prompt}]
        else:
            out = []
        if self.system_prompt:
            out = [{"role": "system", "content": self.system_prompt}, *out]
        return out
