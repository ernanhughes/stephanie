# stephanie/models/model.py
"""Model as a persistent computational resource (§4 of the Stage 1 spec).

Migration sources: ``stephanie/types/model.py`` (ModelSpec),
``stephanie/core/config/schema.py`` (ModelCfg), Writer ``writer-ai/config.py``
and ``writer-ai/models/*`` (naming conventions only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from stephanie.models.capability import ModelCapabilities

# Writer-style reference prefixes, longest-first so "opencode:" never
# shadows "opencode-go:". Kept as a compatibility input format.
LLAMA_CPP_PREFIXES = ("llamacpp:", "llama.cpp:")
OPENCODE_PREFIXES = ("opencode-go:", "opencode-go/", "opencode:")


# Providers Stephanie knows by name. Anything else before a ":" is only a
# provider if it looks like one (no dots/slashes — those mark Ollama-style
# "name:tag", e.g. "qwen3.6:27b" -> ollama).
KNOWN_PROVIDERS = frozenset(
    {
        "ollama",
        "ollama_chat",
        "llamacpp",
        "llama.cpp",
        "opencode",
        "opencode-go",
        "openai",
        "anthropic",
        "local",
        "litellm",
        "stub",
        "vllm",
        "hf",
    }
)


def split_model_ref(model_ref: str) -> tuple[str, str]:
    """Split ``"<provider>:<name>"``; bare names default to ``ollama``."""
    ref = (model_ref or "").strip()
    lowered = ref.lower()
    for prefix in (*OPENCODE_PREFIXES, *LLAMA_CPP_PREFIXES):
        if lowered.startswith(prefix):
            return prefix.rstrip(":/"), ref[len(prefix):]
    if ":" in ref:
        candidate, rest = ref.split(":", 1)
        token = candidate.strip().lower()
        # A "name:tag" pair such as "qwen3.6:27b" stays under the default
        # provider; a single-token prefix ("openai:", "stub:", ...) is a
        # provider namespace (Writer-style prefix routing).
        if "." not in token and "/" not in token and token:
            return token, rest.strip()
    return "ollama", ref


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    name: str

    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    context_window: Optional[int] = None

    enabled: bool = True
    local: bool = False

    metadata: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_ref(model_ref: str, **kwargs: Any) -> "Model":
        provider, name = split_model_ref(model_ref)
        # Explicit override wins over the parsed prefix.
        provider = kwargs.pop("provider", provider)
        local = kwargs.pop("local", provider in {"ollama", "llamacpp", "llama.cpp"})
        return Model(
            id=f"{provider}:{name}",
            provider=provider,
            name=name,
            local=local,
            **kwargs,
        )

    @staticmethod
    def from_model_spec(spec: Any, **kwargs: Any) -> "Model":
        """Adapt legacy ``ModelSpec`` (``types/model.py``) without a flag day."""
        name = getattr(spec, "name", None) or "ollama/qwen:0.5b"
        # Legacy names look like "ollama/qwen:0.5b" or "ollama_chat/qwen3".
        normalized = str(name).replace("/", ":").replace("ollama_chat:", "ollama:")
        provider, model_name = split_model_ref(normalized)
        metadata: Dict[str, Any] = dict(getattr(spec, "metadata", {}) or {})
        metadata.setdefault("api_base", getattr(spec, "api_base", None))
        metadata.setdefault("api_key", getattr(spec, "api_key", None))
        params = getattr(spec, "params", None)
        if params:
            metadata.setdefault("params", params)
        metadata.update(kwargs.pop("metadata", {}) or {})
        return Model(
            id=f"{provider}:{model_name}",
            provider=provider,
            name=model_name,
            metadata=metadata,
            **kwargs,
        )

    @staticmethod
    def from_model_cfg(cfg: Any, **kwargs: Any) -> "Model":
        """Adapt legacy ``ModelCfg`` (``core/config/schema.py``)."""
        if hasattr(cfg, "model_dump"):
            data = cfg.model_dump()
        elif isinstance(cfg, Mapping):
            data = dict(cfg)
        else:
            data = {
                "name": getattr(cfg, "name", "ollama_chat/qwen3"),
                "api_base": getattr(cfg, "api_base", None),
                "api_key": getattr(cfg, "api_key", None),
            }
        return Model.from_ref(
            str(data.get("name") or "ollama:qwen3").replace("/", ":").replace(
                "ollama_chat:", "ollama:"
            ),
            metadata={
                "api_base": data.get("api_base"),
                "api_key": data.get("api_key"),
            },
            **kwargs,
        )


class ModelSpecAdapter:
    """Explicit adapter alias for call sites migrating off ``ModelSpec``."""

    @staticmethod
    def to_model(spec: Any, **kwargs: Any) -> Model:
        return Model.from_model_spec(spec, **kwargs)
