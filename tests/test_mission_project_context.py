from types import SimpleNamespace
import json

import pytest

from sis.services.mission_project_context import MissionProjectContext


def test_mission_projection_uses_only_get_and_omits_unapproved_detail(monkeypatch):
    calls = []

    def get(url, *, params=None, timeout=None):
        calls.append((url, params, timeout))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [{
                "id": "stephanie",
                "name": "Stephanie",
                "description": "private repository detail should not be projected",
                "status": "active",
                "next_action": "Continue qualification",
                "notes": "private notes",
                "open_task_count": 3,
            }],
        )

    monkeypatch.setattr("sis.services.mission_project_context.httpx.get", get)
    result = MissionProjectContext({"base_url": "http://127.0.0.1:8790"}).list_projects()

    assert len(calls) == 1
    assert calls[0][0].endswith("/api/projects")
    assert calls[0][1] == {"include_archived": "false"}
    assert result["projects"][0]["id"] == "stephanie"
    assert "description" not in result["projects"][0]
    assert "notes" not in result["projects"][0]


def test_natural_harness_question_retrieves_a_bounded_current_match(monkeypatch):
    def get(url, *, params=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [
                {"id": "harness", "name": "Harness Engineering", "status": "active", "current_milestone": "H6"},
                {"id": "books", "name": "Books", "status": "active", "current_milestone": "Publication"},
            ],
        )

    monkeypatch.setattr("sis.services.mission_project_context.httpx.get", get)
    context = MissionProjectContext({}).context_for("What remains on harness engineering work?")

    assert context["source"] == "/api/projects"
    assert context["scope"] == "query-matched"
    assert [item["id"] for item in context["projects"]] == ["harness"]


def test_project_identifier_is_validated_before_read():
    with pytest.raises(ValueError, match="Invalid Mission project"):
        MissionProjectContext({}).get_project("../private")


def test_projection_caps_oversized_text_and_total_bytes(monkeypatch):
    def get(url, *, params=None, timeout=None):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [
                {"id": f"project-{index}", "name": "x" * 100_000, "next_action": "y" * 100_000}
                for index in range(100)
            ],
        )

    monkeypatch.setattr("sis.services.mission_project_context.httpx.get", get)
    result = MissionProjectContext({}).list_projects()
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    assert len(encoded) <= 24_000
    assert result["truncated"] is True
    assert all(len(project["name"]) <= 320 for project in result["projects"])
