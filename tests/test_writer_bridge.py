"""Tests for the Stephanie -> Writer runtime bridge."""

from __future__ import annotations

import json

from stephanie.integrations.writer_runtime import (
    WriterRuntimeBridge,
    WriterRuntimeConfig,
    WriterRuntimeDiscovery,
    WriterRuntimeService,
)


def test_config_defaults():
    cfg = WriterRuntimeConfig()
    assert cfg.timeout_seconds == 30
    assert cfg.enable_execution is False


def test_discovery_parses_json(monkeypatch, tmp_path):
    exe = tmp_path / "writer.exe"
    exe.write_text("")
    cfg = WriterRuntimeConfig(writer_executable=str(exe), project_root=str(tmp_path))
    discovery = WriterRuntimeDiscovery(cfg)

    captured = {}

    class _Completed:
        returncode = 0
        stdout = json.dumps({"query": "x", "results": [{"runtime_name": "editorial"}]})
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _Completed()

    import stephanie.integrations.writer_runtime as mod
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = discovery.search("x")
    assert result["ok"] is True
    assert result["results"][0]["runtime_name"] == "editorial"
    assert "--json" in captured["args"]


def test_discovery_handles_nonzero_exit(monkeypatch, tmp_path):
    exe = tmp_path / "writer.exe"
    exe.write_text("")
    cfg = WriterRuntimeConfig(writer_executable=str(exe))
    discovery = WriterRuntimeDiscovery(cfg)

    class _Completed:
        returncode = 2
        stdout = ""
        stderr = "boom"

    import stephanie.integrations.writer_runtime as mod
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Completed())

    result = discovery.describe("missing")
    assert result["ok"] is False
    assert result["stderr"] == "boom"


def test_bridge_status_without_execution():
    bridge = WriterRuntimeBridge(WriterRuntimeConfig(project_root=None))
    status = bridge.status()
    assert status["execution_enabled"] is False
    assert status["runtime_importable"] is None


def test_execution_disabled_returns_clear_error():
    from stephanie.integrations.writer_runtime import WriterRuntimeExecutor

    executor = WriterRuntimeExecutor(WriterRuntimeConfig(enable_execution=False))
    result = executor.invoke("editorial", "describe")
    assert result["ok"] is False
    assert "disabled" in result["error"]


def test_service_initializes_and_reports_health():
    service = WriterRuntimeService(cfg={"project_root": None})
    service.initialize(cfg={"project_root": None})
    health = service.health_check()
    assert health["status"] == "healthy"
    assert health["details"]["execution_enabled"] is False
    service.shutdown()
