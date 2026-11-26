"""Small, owner-local conversational preferences for the isolated SIS UI."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any


DEFAULT_PREFERENCES = {
    "preferred_model": "opencode-go/deepseek-v4.1-flash",
    "policy": "free_first",
    "allow_subscription_models": True,
    "allow_metered_models": False,
    "model_observations": {},
    "external_permissions": {
        "ordinary_project_metadata": False,
        "repository_locations": False,
        "publication_status": False,
        "approved_project_documentation": False,
    },
    "project_grants": [],
}


class ModelPreferences:
    """Persist non-secret preferences outside YAML and outside conversation turns."""

    _lock = RLock()

    def __init__(self, path: str | Path | None = None):
        configured = path or os.environ.get("STEPHANIE_CHAT_PREFERENCES_PATH")
        if configured:
            self.path = Path(configured).expanduser()
        else:
            root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share")
            self.path = root / "Stephanie" / "sis-chat-preferences.json"

    @staticmethod
    def _normalize(value: Any) -> dict[str, Any]:
        value = value if isinstance(value, dict) else {}
        permissions = value.get("external_permissions") or {}
        return {
            "preferred_model": str(value.get("preferred_model") or DEFAULT_PREFERENCES["preferred_model"]),
            "policy": "free_first" if value.get("policy", "free_first") == "free_first" else "specific",
            "allow_subscription_models": bool(value.get("allow_subscription_models", True)),
            # Metered inference is intentionally hard-disabled in this release.
            "allow_metered_models": False,
            "model_observations": {
                str(key)[:180]: {
                    "status": str((observation or {}).get("status", "unverified"))[:40],
                    "reason": str((observation or {}).get("reason", ""))[:240],
                }
                for key, observation in (value.get("model_observations") or {}).items()
                if isinstance(observation, dict)
            } if isinstance(value.get("model_observations") or {}, dict) else {},
            "external_permissions": {
                key: bool(permissions.get(key, False))
                for key in DEFAULT_PREFERENCES["external_permissions"]
            },
            "project_grants": [
                {"project_id": str(row.get("project_id", ""))[:120],
                 "provider": str(row.get("provider", ""))[:180],
                 "categories": sorted({str(c) for c in row.get("categories", []) if str(c) in DEFAULT_PREFERENCES["external_permissions"]})}
                for row in (value.get("project_grants") or [])
                if isinstance(row, dict) and re.fullmatch(r"[A-Za-z0-9._-]{1,120}", str(row.get("project_id", ""))) and row.get("provider")
            ][:100],
        }

    def read(self) -> dict[str, Any]:
        with self._lock:
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = DEFAULT_PREFERENCES
            return self._normalize(value)

    def write(self, value: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(value)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="sis-prefs-", suffix=".json", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(normalized, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return normalized

    def observe_model(self, model: str, status: str, reason: str = "") -> None:
        """Record only safe route state observed by an actual conversation turn."""
        if status not in {"worked", "blocked", "unverified"}:
            return
        with self._lock:
            value = self.read()
            value["model_observations"][str(model)[:180]] = {
                "status": status,
                "reason": str(reason)[:240],
            }
            self.write(value)
