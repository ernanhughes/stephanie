import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from stephanie.improvement.evolution02q import ReadinessAnswer
from stephanie.improvement.worker_harness import WorkerHarnessClient, validate_worker_hypothesis


def test_worker_client_requires_loopback_and_bounds_resources():
    with pytest.raises(ValueError, match="loopback"):
        WorkerHarnessClient("http://example.com", model="m", policy="p")
    with pytest.raises(ValueError, match="bounded"):
        WorkerHarnessClient("http://127.0.0.1", model="m", policy="p", output_tokens=2049)


def test_worker_work_contract_and_independent_validation():
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"version":"test", "executors":[{"id":"model", "available":True}]}
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def do_POST(self):
            captured.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = {"request_id":"wrq_example", "status":"AWAITING_ACCEPTANCE", "output":"{\"findings\":[\"x\"],\"unknowns\":[\"unknown\"],\"next_step\":\"inspect next step\",\"evidence_ids\":[\"D1\"]}",
                "strategy_execution":{"strategy":{"digest":"abc"}, "validation":{"status":"valid"},
                    "invocations":[{"invocation_id":"call-1"}], "resources":{"invocation_count":1}}}
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
        def log_message(self, *_): pass

    server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        client = WorkerHarnessClient(f"http://127.0.0.1:{server.server_port}", model="local", policy="p")
        assert client.health()["version"] == "test"
        result = client.run(task_id="dev1", objective="public test", output_schema={"type":"object"})
        assert captured["execution"]["strategy_id"] == "adaptive.v1"
        assert captured["execution"]["normalization_policy"] == "json-fence.v1"
        assert captured["parameters"]["data_policy"] == "local_only"
        validated = validate_worker_hypothesis(result, ReadinessAnswer, allowed_evidence_ids={"D1"})
        assert validated["eligible"] is True
        assert validated["authority"] == "hypothesis_only"
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=1)


def test_worker_candidate_stays_ineligible_on_invalid_work_or_evidence():
    result = {"status":"FAILED", "output":"{}", "strategy_execution":{"validation":{"status":"invalid"},
        "invocations":[{"invocation_id":"call-1"}], "strategy":{"digest":"abc"}}}
    assert validate_worker_hypothesis(result, ReadinessAnswer, allowed_evidence_ids={"D1"})["eligible"] is False


def test_worker_rejects_fabricated_evidence_reference():
    result = {"status":"AWAITING_ACCEPTANCE", "output":"{\"findings\":[\"x\"],\"unknowns\":[\"unknown\"],\"next_step\":\"inspect next step\",\"evidence_ids\":[\"SECRET\"]}",
        "strategy_execution":{"validation":{"status":"valid"}, "invocations":[{"invocation_id":"call-1"}], "strategy":{"digest":"abc"}}}
    validated = validate_worker_hypothesis(result, ReadinessAnswer, allowed_evidence_ids={"D1"})
    assert validated["eligible"] is False
    assert validated["invalid_evidence_ids"] == ["SECRET"]


def test_readiness_schema_enforces_declared_bounds():
    with pytest.raises(ValueError):
        ReadinessAnswer(findings=["a", "b", "c"], unknowns=[], next_step="ok", evidence_ids=[])
    with pytest.raises(ValueError):
        ReadinessAnswer(findings=[], unknowns=[], next_step="x" * 301, evidence_ids=[])
    with pytest.raises(ValueError):
        ReadinessAnswer(findings=["finding"], unknowns=[], next_step="describe next step", evidence_ids=[])
    with pytest.raises(ValueError):
        ReadinessAnswer(findings=[""], unknowns=["unknown"], next_step="describe next step", evidence_ids=["D1"])
