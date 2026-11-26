"""Bounded, read-only compiler for Mission project and book evidence."""

from __future__ import annotations

import json
import re
import copy
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx


_PROJECT_QUERY = re.compile(r"(?i)\b(project|projects|work|outstanding|mission|harness|what remains|what's left|at the moment|next steps|blockers)\b")
_MAX_TEXT_FIELD_CHARS = 320
_MAX_PROJECTION_BYTES = 24_000


class MissionProjectContext:
    """Read an allowlist of current facts from Mission; never reads project files."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.base_url = str(config.get("base_url", "http://127.0.0.1:8790")).rstrip("/")
        self.timeout = min(10.0, max(1.0, float(config.get("timeout_seconds", 4))))
        self.cache_seconds = min(300, max(0, int(config.get("cache_seconds", 30))))
        self.programme_docs_root = Path(config["programme_docs_root"]) if config.get("programme_docs_root") else None
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def is_project_query(text: str) -> bool:
        return bool(_PROJECT_QUERY.search(text or ""))

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        response = httpx.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_projects(self) -> dict[str, Any]:
        raw = self._get("/api/projects", params={"include_archived": "false"})
        if not isinstance(raw, list):
            raise ValueError("Mission returned an invalid project list")
        projects = [self._project_summary(item) for item in raw if isinstance(item, dict)]
        return self._fit_projection({
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source": "/api/projects",
            "total_project_count": len(projects),
            "truncated": len(projects) > 100,
            "projects": projects[:100],
        })

    def get_project(self, project_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", project_id or ""):
            raise ValueError("Invalid Mission project identifier")
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(project_id)
            if cached and now - cached[0] <= self.cache_seconds:
                result = copy.deepcopy(cached[1])
                result["cache_age_seconds"] = round(now - cached[0], 1)
                result["context_freshness"] = "cached-live-evidence"
                return result
        observed = datetime.now(timezone.utc).isoformat()
        raw = self._get(f"/api/projects/{quote(project_id, safe='._-')}")
        if not isinstance(raw, dict):
            raise ValueError("Mission returned an invalid project record")
        project = self._project_detail(raw)
        sources = [{"source": f"/api/projects/{project_id}", "observed_at": observed, "freshness": "live"}]

        publication = None
        pipeline = None
        if str(raw.get("kind", "")).lower() == "book" or raw.get("book"):
            # Mission's read-only projections join the Writer registry and
            # publication evidence without opening manuscripts or dispatching work.
            for name, path, params in (
                ("pipeline", f"/api/pipeline/books/{quote(project_id, safe='._-')}", None),
                ("publication", "/api/publication/publishing", {"project_id": project_id}),
                ("assessment", "/api/publication/assessments", {"project_id": project_id}),
            ):
                try:
                    evidence = self._get(path, params=params)
                except Exception as exc:
                    project.setdefault("missing_information", []).append(f"{name} evidence unavailable ({type(exc).__name__})")
                    continue
                source_observed = evidence.get("observed_at") if isinstance(evidence, dict) else None
                sources.append({"source": path, "observed_at": source_observed or observed, "freshness": "reported by source" if source_observed else "observed now"})
                if name == "pipeline" and isinstance(evidence, dict):
                    pipeline = self._pipeline_projection(evidence)
                if name == "publication" and isinstance(evidence, dict):
                    publication = self._publication_projection(evidence)
                if name == "assessment":
                    assessment = self._assessment_projection(evidence)
                    if assessment:
                        if publication is None:
                            publication = {}
                        publication["assessment"] = assessment

        result = {
            "available": True,
            "project": project,
            "pipeline": pipeline,
            "publication": publication,
            "sources": sources,
            "categories": ["ordinary_project_metadata", "repository_locations"] + (["publication_status"] if publication else []),
            "excluded_categories": ["source_code", "internal_project_documents", "personal_information", "credentials", "protected_evaluator_material"],
            "missing_information": list(project.get("missing_information", [])),
            "observed_at": observed,
        }
        if str(project.get("kind", "")).lower() == "book" and self.programme_docs_root:
            relationship_evidence = self._programme_relationship_projection(project.get("name") or project_id)
            if relationship_evidence:
                result["documented_relationship_evidence"] = relationship_evidence
                result["categories"].append("approved_project_documentation")
        result["external_sections"] = {
            "ordinary_project_metadata": {
                key: project.get(key) for key in ("id", "name", "description", "kind", "status", "priority", "current_milestone", "next_action", "objectives", "related_projects", "open_task_count", "pending_recommendation_count", "tasks", "decisions", "checkpoints", "recommendations", "activity", "missing_information")
            },
            "repository_locations": {
                "repositories": project.get("repositories", []),
                "book": {key: (project.get("book") or {}).get(key) for key in ("manuscript_paths", "implementation_paths")},
                "writer_chapter_dir": (pipeline or {}).get("chapter_dir"),
            },
            "publication_status": {
                "pipeline": {key: value for key, value in (pipeline or {}).items() if key != "chapter_dir"},
                "publication": publication,
            },
        }
        if result.get("documented_relationship_evidence"):
            result["external_sections"]["approved_project_documentation"] = {
                "documented_relationship_evidence": result["documented_relationship_evidence"]
            }
        result = self._fit_projection(result)
        with self._cache_lock:
            self._cache[project_id] = (time.monotonic(), copy.deepcopy(result))
        result["cache_age_seconds"] = 0
        result["context_freshness"] = "live"
        return result

    @staticmethod
    def external_projection(context: dict[str, Any], permissions: dict[str, bool]) -> tuple[dict[str, Any], list[str]]:
        sections = context.get("external_sections") or {}
        included = {key: sections[key] for key in sections if permissions.get(key, False)}
        excluded = [key for key in sections if key not in included]
        return {
            "project_id": (context.get("project") or {}).get("id"),
            "included_categories": sorted(included),
            "excluded_categories": sorted(excluded),
            "sources": context.get("sources", []),
            "observed_at": context.get("observed_at"),
            "context": included,
        }, sorted(excluded)

    @staticmethod
    def model_projection(context: dict[str, Any]) -> dict[str, Any]:
        """Remove inspector-only duplicate fields before adding evidence to a prompt."""
        keys = (
            "project", "projects", "pipeline", "publication", "sources", "categories",
            "excluded_categories", "observed_at", "cache_age_seconds", "context_freshness",
            "missing_information", "scope", "matched_project_count", "returned_project_count",
            "total_project_count", "truncated", "documented_relationship_evidence",
        )
        return {key: context[key] for key in keys if key in context}

    @classmethod
    def _project_summary(cls, item: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "name", "kind", "status", "priority", "current_milestone", "next_action", "open_task_count", "pending_recommendation_count", "updated_at")
        return {key: cls._safe_text(item.get(key)) for key in keys}

    def _programme_relationship_projection(self, project_name: str) -> list[dict[str, str]]:
        """Read three allowlisted programme ledgers and return matching lines with provenance."""
        if not self.programme_docs_root:
            return []
        needle = re.compile(rf"(?i)\b{re.escape(project_name)}\b")
        records = []
        for filename in ("ROADMAP.md", "ARCHITECTURE.md", "PUB09-REUSE-MATRIX.md"):
            path = self.programme_docs_root / filename
            try:
                if not path.is_file() or path.stat().st_size > 180_000:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if needle.search(line) and len(records) < 8:
                        records.append({"claim_status": "documented_programme_reference", "source": str(path),
                                        "line": str(number), "excerpt": self._safe_text(line.strip(), 280)})
            except (OSError, UnicodeError):
                continue
        return records

    @staticmethod
    def _safe_text(value: Any, limit: int = _MAX_TEXT_FIELD_CHARS) -> Any:
        if isinstance(value, str):
            return value[:limit]
        return value

    @classmethod
    def _project_detail(cls, item: dict[str, Any]) -> dict[str, Any]:
        project = {key: cls._safe_text(item.get(key), 1600 if key == "description" else _MAX_TEXT_FIELD_CHARS)
                   for key in ("id", "name", "description", "kind", "status", "priority", "current_milestone", "next_action", "open_task_count", "pending_recommendation_count", "updated_at", "last_activity_at")}
        project["objectives"] = [
            {key: cls._safe_text(row.get(key), 480) for key in ("id", "name", "description", "status", "url")}
            for row in (item.get("outcomes") or [])[:8] if isinstance(row, dict)
        ]
        project["related_projects"] = [
            {key: cls._safe_text(row.get(key)) for key in ("id", "name", "relationship", "status")}
            for row in (item.get("related_projects") or [])[:8] if isinstance(row, dict)
        ]
        project["repositories"] = [
            {key: cls._safe_text(row.get(key)) for key in ("id", "name", "path", "classification", "role", "branch", "commit_hash", "last_commit_at", "last_observed_at", "accessible", "dirty")}
            for row in (item.get("repositories") or [])[:8] if isinstance(row, dict)
        ]
        book = item.get("book")
        if isinstance(book, dict):
            project["book"] = {key: cls._safe_text(book.get(key)) for key in ("id", "title", "status", "current_milestone", "next_action", "manuscript_paths", "implementation_paths", "published_url", "quality_review_state", "last_activity_at")}
        for key, fields in (
            ("tasks", ("id", "title", "status", "priority", "created_at", "updated_at")),
            ("decisions", ("id", "title", "status", "created_at", "updated_at")),
            ("checkpoints", ("id", "title", "status", "created_at")),
            ("recommendations", ("id", "title", "status", "priority", "created_at")),
            ("activity", ("id", "type", "title", "detail", "source", "provenance", "occurred_at", "observed_at")),
        ):
            rows = item.get(key) or []
            project[key] = [{field: cls._safe_text(row.get(field), 360) for field in fields}
                            for row in rows[:8] if isinstance(row, dict)]
            project[f"{key}_total_count"] = len(rows)
            project[f"{key}_truncated"] = len(rows) > len(project[key])
        project["missing_information"] = []
        if not project.get("current_milestone"):
            project["missing_information"].append("Mission has no current milestone recorded")
        if not project.get("next_action"):
            project["missing_information"].append("Mission has no next action recorded")
        if not project["tasks"]:
            project["missing_information"].append("Mission has no outstanding project tasks recorded")
        if not project["objectives"]:
            project["missing_information"].append("Mission returned no linked Work Outcomes for this project")
        if not project["related_projects"]:
            project["missing_information"].append("Mission project detail did not expose related projects")
        return project

    @classmethod
    def _pipeline_projection(cls, item: dict[str, Any]) -> dict[str, Any]:
        discovery = item.get("discovery") or {}
        fingerprint = item.get("source_fingerprint") or {}
        return {
            key: cls._safe_text(item.get(key))
            for key in ("project_id", "known", "book_title", "registry_version", "registry_source", "chapter_dir", "chapter_pattern", "profile", "validation_enabled", "human_approval_required", "observed_at", "refresh_note")
        } | {
            "chapter_count": discovery.get("applicable_count", item.get("chapter_count")),
            "chapter_file_count": fingerprint.get("file_count"),
            "source_fingerprint_ok": fingerprint.get("ok"),
            "discovery_exclusions": [cls._safe_text(value) for value in (discovery.get("exclude_prefixes") or [])[:8]],
            "discovery_skipped": [cls._safe_text(value) for value in (discovery.get("skipped") or [])[:8]],
        }

    @classmethod
    def _publication_projection(cls, item: dict[str, Any]) -> dict[str, Any]:
        output = {key: cls._safe_text(item.get(key)) for key in ("project_id", "project_name", "book_id", "book_title", "manuscript_root", "chapter_count", "lifecycle", "lifecycle_note", "publication_gate", "policy_version", "fingerprint_match", "observed_at")}
        evidence = item.get("evidence") or []
        output["evidence"] = [
            {key: cls._safe_text(row.get(key), 500) for key in ("evidence_type", "source_system", "source_reference", "source_revision", "observed_at", "executed_at", "result", "freshness", "diagnostics")}
            for row in evidence[:12] if isinstance(row, dict)
        ]
        output["evidence_total_count"] = len(evidence)
        output["evidence_truncated"] = len(evidence) > len(output["evidence"])
        return output

    @classmethod
    def _assessment_projection(cls, item: Any) -> dict[str, Any] | None:
        """Keep the latest readiness assessment and its bounded gate evidence."""
        if isinstance(item, list):
            item = item[0] if item else None
        if not isinstance(item, dict):
            return None
        result = {key: cls._safe_text(item.get(key)) for key in (
            "id", "assessment_status", "freshness", "readiness_detail", "chapter_count", "created_at"
        )}
        result["gates"] = [
            {key: cls._safe_text(row.get(key), 360) for key in (
                "gate_id", "result", "blocking", "detail", "evidence_refs", "limitations"
            )}
            for row in (item.get("gates") or [])[:12] if isinstance(row, dict)
        ]
        result["issues"] = [
            {key: cls._safe_text(row.get(key), 320) for key in (
                "gate_id", "issue_type", "severity", "status", "rationale", "evidence_refs"
            )}
            for row in (item.get("issues") or [])[:12] if isinstance(row, dict)
        ]
        result["gates_truncated"] = len(item.get("gates") or []) > len(result["gates"])
        result["issues_truncated"] = len(item.get("issues") or []) > len(result["issues"])
        return result

    @classmethod
    def _fit_projection(cls, value: dict[str, Any]) -> dict[str, Any]:
        def shrink(item: Any, limit: int) -> Any:
            if isinstance(item, str):
                return item[:limit]
            if isinstance(item, list):
                return [shrink(child, limit) for child in item]
            if isinstance(item, dict):
                return {key: shrink(child, limit) for key, child in item.items()}
            return item

        result = shrink(value, _MAX_TEXT_FIELD_CHARS)
        while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > _MAX_PROJECTION_BYTES:
            changed = False
            for key in ("activity", "evidence", "tasks", "decisions", "checkpoints", "recommendations", "repositories", "projects"):
                containers = [result.get("project", {}).get(key), result.get("publication", {}).get(key), result.get(key)]
                target = max((part for part in containers if isinstance(part, list) and part), key=len, default=None)
                if target is not None:
                    target.pop()
                    changed = True
                    break
            if changed:
                continue
            strings = list(cls._walk_strings(result))
            max_len = max((len(part) for part in strings), default=0)
            if max_len <= 80:
                break
            result = shrink(result, 80)
        if isinstance(result.get("project"), dict):
            for key in ("activity", "evidence", "tasks", "decisions", "checkpoints", "recommendations", "repositories"):
                rows = result["project"].get(key)
                if isinstance(rows, list) and result["project"].get(f"{key}_total_count", len(rows)) > len(rows):
                    result["project"][f"{key}_truncated"] = True
        if isinstance(result.get("projects"), list):
            result["truncated"] = bool(result.get("truncated")) or len(result["projects"]) < int(result.get("total_project_count", 0))
        result["estimated_context_chars"] = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return result

    @classmethod
    def _walk_strings(cls, value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from cls._walk_strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_strings(child)

    def context_for(self, text: str, project_id: str | None = None) -> dict[str, Any] | None:
        if project_id:
            return self.get_project(project_id)
        if not self.is_project_query(text):
            return None
        listing = self.list_projects()
        query_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9-]{3,}", text)}
        projects = listing["projects"]
        matched = [row for row in projects if any(token in " ".join(str(row.get(key, "")) for key in ("id", "name", "current_milestone")).lower() for token in query_tokens)]
        listing["projects"] = (matched or projects)[:20]
        listing["scope"] = "query-matched" if matched else "current-project-list"
        listing["matched_project_count"] = len(matched)
        listing["returned_project_count"] = len(listing["projects"])
        listing["truncated"] = listing["truncated"] or len(matched or projects) > 20
        return listing
