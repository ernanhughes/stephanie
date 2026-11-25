"""Bounded, read-only Mission API projections for conversational grounding."""

from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from typing import Any

import httpx


_PROJECT_QUERY = re.compile(r"(?i)\b(project|projects|work|outstanding|mission|harness|what remains|what's left|at the moment)\b")
_MAX_TEXT_FIELD_CHARS = 320
_MAX_PROJECTION_BYTES = 24_000


class MissionProjectContext:
    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.base_url = str(config.get("base_url", "http://127.0.0.1:8790")).rstrip("/")
        self.timeout = min(10.0, max(1.0, float(config.get("timeout_seconds", 4))))

    @staticmethod
    def is_project_query(text: str) -> bool:
        return bool(_PROJECT_QUERY.search(text or ""))

    def list_projects(self) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/api/projects",
            params={"include_archived": "false"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, list):
            raise ValueError("Mission returned an invalid project list")
        projects = [self._project_summary(item) for item in raw if isinstance(item, dict)]
        result = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "/api/projects",
            "total_project_count": len(projects),
            "truncated": len(projects) > 100,
            "projects": projects[:100],
        }
        return self._fit_projection(result)

    def get_project(self, project_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", project_id or ""):
            raise ValueError("Invalid Mission project identifier")
        response = httpx.get(f"{self.base_url}/api/projects/{project_id}", timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()
        if not isinstance(raw, dict):
            raise ValueError("Mission returned an invalid project record")
        return self._fit_projection({"observed_at": datetime.now(timezone.utc).isoformat(), "source": f"/api/projects/{project_id}", "project": self._project_detail(raw)})

    @staticmethod
    def _safe_text(value: Any) -> Any:
        if isinstance(value, str):
            return value[:_MAX_TEXT_FIELD_CHARS]
        return value

    @classmethod
    def _fit_projection(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Keep even oversized upstream fields/collections within a byte budget."""
        def shrink(item: Any, text_limit: int) -> Any:
            if isinstance(item, str):
                return item[:text_limit]
            if isinstance(item, list):
                return [shrink(child, text_limit) for child in item]
            if isinstance(item, dict):
                return {key: shrink(child, text_limit) for key, child in item.items()}
            return item

        result = shrink(value, _MAX_TEXT_FIELD_CHARS)
        while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > _MAX_PROJECTION_BYTES:
            collections = [result.get("projects"), *[result.get("project", {}).get(key) for key in ("repositories", "tasks", "decisions", "recommendations", "checkpoints")]]
            largest = max((items for items in collections if isinstance(items, list) and items), key=len, default=None)
            if largest is not None:
                largest.pop()
                continue
            current_limit = 80 if any(isinstance(item, str) and len(item) > 80 for item in _walk_strings(result)) else 0
            if current_limit:
                result = shrink(result, current_limit)
                continue
            break
        if "projects" in result:
            result["truncated"] = bool(result.get("truncated")) or len(result["projects"]) < result.get("total_project_count", 0)
        return result

    @staticmethod
    def _walk_strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from MissionProjectContext._walk_strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from MissionProjectContext._walk_strings(child)

    @staticmethod
    def _project_summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: MissionProjectContext._safe_text(item.get(key))
            for key in ("id", "name", "kind", "status", "priority", "current_milestone", "next_action", "open_task_count", "pending_recommendation_count", "updated_at")
        }

    @classmethod
    def _project_detail(cls, item: dict[str, Any]) -> dict[str, Any]:
        result = cls._project_summary(item)
        result["repositories"] = [
            {key: cls._safe_text(row.get(key)) for key in ("id", "name", "branch", "status")}
            for row in (item.get("repositories") or [])[:8] if isinstance(row, dict)
        ]
        for key, fields in (
            ("tasks", ("id", "title", "status", "priority")),
            ("decisions", ("id", "title", "status")),
            ("recommendations", ("id", "title", "status")),
            ("checkpoints", ("id", "title", "status", "created_at")),
        ):
            rows = [
                {field: cls._safe_text(row.get(field)) for field in fields}
                for row in (item.get(key) or [])[:8] if isinstance(row, dict)
            ]
            result[key] = rows
            result[f"{key}_total_count"] = len(item.get(key) or [])
            result[f"{key}_truncated"] = len(item.get(key) or []) > len(rows)
        return result

    def context_for(self, text: str, project_id: str | None = None) -> dict[str, Any] | None:
        if project_id:
            return self.get_project(project_id)
        if not self.is_project_query(text):
            return None
        listing = self.list_projects()
        query_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9-]{3,}", text)}
        projects = listing["projects"]
        matched = [
            row for row in projects
            if any(
                token in " ".join(str(row.get(key, "")) for key in ("id", "name", "current_milestone")).lower()
                for token in query_tokens
            )
        ]
        listing["projects"] = (matched or projects)[:20]
        listing["scope"] = "query-matched" if matched else "current-project-list"
        listing["matched_project_count"] = len(matched)
        listing["returned_project_count"] = len(listing["projects"])
        listing["truncated"] = listing["truncated"] or len(matched or projects) > 20
        return listing
