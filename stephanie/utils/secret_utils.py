"""Small helpers for keeping credentials out of durable logs and run config."""

from __future__ import annotations

import re
from typing import Any

from omegaconf import OmegaConf


_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "credential",
)
_SENSITIVE_EXACT_KEYS = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "id_token",
    "api_token",
    "client_token",
    "bearer_token",
    "authorization",
    "dsn",
    "connection_string",
    "connectionstring",
}
_REDACTED = "[REDACTED]"
_URL_CREDENTIALS = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.-]*://[^:/\s@]+:)([^@\s]+)(@)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|credential)\b\s*['\"]?\s*[:=]\s*['\"]?)([^'\"\s,;}]+)"
)


def _redact_text(value: str) -> str:
    value = _URL_CREDENTIALS.sub(rf"\1{_REDACTED}\3", value)
    return _SECRET_ASSIGNMENT.sub(rf"\1{_REDACTED}", value)


def redact_secrets(value: Any) -> Any:
    """Return a JSON/YAML-safe copy with credential-valued keys masked.

    OmegaConf interpolation is intentionally not resolved here: resolving
    ``oc.env`` values before redaction could materialize a secret in memory or
    in the resulting diagnostics.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_EXACT_KEYS or any(
                part in normalized for part in _SENSITIVE_KEY_PARTS
            ):
                result[key] = _REDACTED if item is not None else None
            elif normalized in {"url", "database_url"} and isinstance(item, str):
                result[key] = _REDACTED if "://" in item and "@" in item else item
            else:
                result[key] = redact_secrets(item)
        return result

    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]

    if isinstance(value, str):
        return _redact_text(value)

    return value
