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


def test_book_context_compiles_project_pipeline_and_publication_evidence(monkeypatch):
    calls = []
    payloads = {
        "/api/projects/jev": {
            "id": "jev", "name": "Jev", "kind": "book", "status": "active",
            "description": "A book about typed, uncertain decisions.",
            "current_milestone": "", "next_action": "", "tasks": [], "decisions": [],
            "outcomes": [{"id": "outcome-1", "name": "Publish Jev", "description": "Qualify the pipeline", "status": "open", "notes": "private"}],
            "related_projects": [{"id": "writer", "name": "Writer", "relationship": "depends_on", "status": "active"}],
            "repositories": [{"name": "jev", "path": "C:/books/jev", "role": "manuscript"}],
            "book": {"title": "Jev From First Principles", "status": "drafting", "manuscript_paths": ["C:/books/jev/content"]},
            "activity": [{"title": "Book record updated", "occurred_at": "2026-11-25T12:00:00Z"}],
        },
        "/api/pipeline/books/jev": {"known": True, "book_title": "Jev", "chapter_dir": "C:/books/jev/content",
                                     "registry_source": "writer/config/projects.yaml",
                                     "discovery": {"applicable_count": 36, "exclude_prefixes": ["00"], "skipped": ["_index.md (non-chapter stem)"]},
                                     "source_fingerprint": {"ok": True, "file_count": 37}},
        "/api/publication/publishing": {"project_id": "jev", "project_name": "Jev",
            "lifecycle": "development", "publication_gate": "mandatory evidence missing: prose_review",
            "observed_at": "2026-11-25T12:30:00Z", "evidence": [
                {"evidence_type": "writer_validation_run", "result": "unknown", "freshness": "unknown",
                 "diagnostics": "No stored validation output."}]},
        "/api/publication/assessments": [{"id": "assessment-1", "assessment_status": "NOT_YET_ASSESSED",
            "freshness": "CURRENT", "readiness_detail": "mandatory evidence missing: prose_review",
            "chapter_count": 36, "created_at": "2026-11-25T12:30:00Z",
            "gates": [{"gate_id": "prose_review", "result": "NOT_RUN", "blocking": False,
                       "detail": "1 of 36 discovered chapters current", "evidence_refs": ["quality:discovery"]}],
            "issues": []}],
    }

    def get(url, *, params=None, timeout=None):
        path = url.removeprefix("http://127.0.0.1:8790")
        calls.append((path, params))
        body = payloads[path]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    monkeypatch.setattr("sis.services.mission_project_context.httpx.get", get)
    compiler = MissionProjectContext({"cache_seconds": 0})
    result = compiler.get_project("jev")

    assert {path for path, _ in calls} == {
        "/api/projects/jev", "/api/pipeline/books/jev", "/api/publication/publishing",
        "/api/publication/assessments",
    }
    assert result["project"]["missing_information"] == [
        "Mission has no current milestone recorded",
        "Mission has no next action recorded",
        "Mission has no outstanding project tasks recorded",
    ]
    assert result["project"]["objectives"][0]["name"] == "Publish Jev"
    assert "notes" not in json.dumps(result["project"]["objectives"])
    assert result["project"]["related_projects"][0]["id"] == "writer"
    assert result["publication"]["publication_gate"].startswith("mandatory evidence missing")
    assert result["publication"]["assessment"]["assessment_status"] == "NOT_YET_ASSESSED"
    assert result["publication"]["assessment"]["gates"][0]["detail"] == "1 of 36 discovered chapters current"
    assert result["pipeline"]["chapter_count"] == 36
    assert result["pipeline"]["chapter_file_count"] == 37
    assert result["external_sections"]["repository_locations"]["writer_chapter_dir"] == "C:/books/jev/content"
    assert "chapter_dir" not in result["external_sections"]["publication_status"]["pipeline"]
    prompt_projection = compiler.model_projection(result)
    assert "external_sections" not in prompt_projection
    assert len(json.dumps(prompt_projection)) < len(json.dumps(result))
    assert result["sources"][-2]["source"] == "/api/publication/publishing"
    assert result["sources"][-1]["source"] == "/api/publication/assessments"
    assert result["estimated_context_chars"] < 24_000


def test_external_context_is_category_filtered_and_never_includes_full_documents():
    full = {
        "project": {"id": "jev", "name": "Jev", "description": "Book", "repositories": [{"path": "C:/books/jev"}]},
        "external_sections": {
            "ordinary_project_metadata": {"id": "jev", "name": "Jev"},
            "repository_locations": {"root": "C:/books/jev"},
            "publication_status": {"gate": "evidence missing"},
        },
        "sources": [{"source": "/api/projects/jev", "observed_at": "now"}],
        "observed_at": "now",
    }
    projection, excluded = MissionProjectContext.external_projection(
        full, {"ordinary_project_metadata": True, "repository_locations": False, "publication_status": False}
    )
    assert projection["included_categories"] == ["ordinary_project_metadata"]
    assert excluded == ["publication_status", "repository_locations"]
    assert "repositories" not in json.dumps(projection)
    assert "external_sections" not in projection["context"]


def test_allowlisted_programme_relationships_retain_line_provenance(tmp_path):
    (tmp_path / "ROADMAP.md").write_text("# Roadmap\n- Jev end-to-end work.\n- Wider portfolio reuses the qualified process.\n", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text("Jev follows the shared Publication architecture.\n", encoding="utf-8")
    (tmp_path / "unrelated.md").write_text("Jev should not be read from unlisted files.\n", encoding="utf-8")
    evidence = MissionProjectContext({"programme_docs_root": str(tmp_path)})._programme_relationship_projection("Jev")
    assert len(evidence) == 2
    assert evidence[0]["claim_status"] == "documented_programme_reference"
    assert evidence[0]["line"] == "2"
    assert "unrelated.md" not in json.dumps(evidence)
