from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
from threading import Thread

from stephanie.improvement.integrations import (
    as_evidence, compare_with_praxis, read_mission_episodes, read_worker_executions,
    read_worker_strategies,
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b'{"strategies":[{"id":"adaptive.v1"}]}' if self.path == "/v1/strategies"
                else b'{"executions":[{"request_id":"req-1","inference_executed":"UNKNOWN"}]}' if self.path == "/v1/executions"
                else b'{"episodes":[{"id":"ep-1","project_key":"p1"}]}')
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_):
        pass


def test_worker_and_mission_adapters_use_read_only_source_identities():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        strategies = read_worker_strategies(base)
        executions = read_worker_executions(base)
        mission = read_mission_episodes(base, limit=2)
    finally:
        server.shutdown()
    assert strategies.records[0]["id"] == "adaptive.v1"
    assert executions.records[0]["inference_executed"] == "UNKNOWN"
    assert mission.records[0]["id"] == "ep-1"
    mapped = as_evidence("worker", "req-1", executions.records[0])
    assert mapped.source_identity == "worker:req-1"
    assert mapped.evaluation_status == mapped.acceptance_status == "UNKNOWN"


def test_unavailable_worker_is_a_result_not_a_crash():
    result = read_worker_executions("http://127.0.0.1:1", timeout=0.1)
    assert result.state == "unavailable_service"
    assert result.records == ()


def test_operational_adapters_reject_non_loopback_urls():
    result = read_worker_strategies("https://example.com")
    assert result.state == "unavailable_service"
    assert "loopback" in result.detail


def test_praxis_adapter_calls_canonical_comparison_and_keeps_insufficient_result():
    praxis_root = Path(__file__).resolve().parents[2] / "praxis"
    sys.path.insert(0, str(praxis_root))
    result = compare_with_praxis([], [])
    assert result["mode"] == "existing_callable_implementation"
    assert result["comparison"]["decision"] == "insufficient_evidence"
    assert result["operational_evaluation"] is False


def test_worker_investigation_and_unapproved_proposal_persist_then_retrieve(tmp_path):
    from stephanie.improvement.engine import ImprovementEngine
    class CaseBookSpy:
        def __init__(self): self.entries = []
        def get_by_name(self, name): return None
        def create_casebook(self, *args, **kwargs): return {"name": args[0]}
        def get_id_by_name(self, name): return 7
        def add_case(self, *args, **kwargs): self.entries.append(kwargs)
        def list_case_summaries(self, book_id): return []
    book = CaseBookSpy()
    engine = ImprovementEngine(tmp_path, casebook_store=book)
    investigation = engine.investigate_worker_empty_output()
    proposal = engine.propose_worker_diagnostic()
    assert engine.retrieve_investigation(investigation["investigation_id"]) == investigation
    assert investigation["result"] == "REQUEST_ADDITIONAL_EVIDENCE"
    assert investigation["hypothesis"] is None
    assert proposal["approval_status"] == "NOT_SUBMITTED_NOT_AUTHORIZED"
    assert proposal["resource_limits"]["model_invocations"] == 0
    assert book.entries[0]["meta"]["reuse_eligible"] is False


def test_unsupported_ai_citation_preserves_available_provider_result(tmp_path, monkeypatch):
    import json
    from stephanie.improvement.engine import ImprovementEngine
    from ai.inference.providers import ollama
    from ai.inference.providers.fake import DeterministicFakeReasoningProvider

    engine = ImprovementEngine(tmp_path)
    source = json.loads((Path(__file__).parents[1] / "stephanie" / "improvement" / "harness_fixtures.json").read_text())[:2]
    engine.import_evidence(source)
    raw = json.dumps({"statement": "A sufficiently long unsupported hypothesis", "rationale": "This generated claim cites no supplied evidence", "uncertainty": "The cause remains uncertain", "evidence_ids": ["not-in-evidence"]})
    monkeypatch.setattr(ollama, "OllamaReasoningProvider", lambda **kwargs: DeterministicFakeReasoningProvider(response=raw))
    try:
        engine.investigate_with_ai("ollama", "fake")
    except ValueError as exc:
        assert "cited evidence" in str(exc)
    else:
        raise AssertionError("unsupported citation must be rejected")
    saved = engine._read(engine.ai_path, [])[-1]
    assert saved["raw_response"] == raw
    assert saved["result"]["request_id"]
    assert saved["qualification_eligible"] is False
