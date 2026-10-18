"""Tests for environment overrides of the Writer runtime bridge config."""

from __future__ import annotations

from stephanie.integrations.writer_runtime import WriterRuntimeConfig


def test_writer_config_defaults_when_env_is_unset(monkeypatch):
    for name in (
        "WRITER_PROJECT_ROOT",
        "WRITER_EXECUTABLE",
        "WRITER_PYTHON",
        "WRITER_TIMEOUT_SECONDS",
        "WRITER_ENABLE_EXECUTION",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = WriterRuntimeConfig(project_root="C:/Projects/Writer", enable_execution=False)
    assert cfg.project_root == "C:/Projects/Writer"
    assert cfg.enable_execution is False
    assert cfg.timeout_seconds == 30


def test_writer_config_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("WRITER_PROJECT_ROOT", "D:/other/Writer")
    monkeypatch.setenv("WRITER_EXECUTABLE", "D:/other/Writer/writer.cmd")
    monkeypatch.setenv("WRITER_PYTHON", "D:/other/Writer/python.exe")
    monkeypatch.setenv("WRITER_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("WRITER_ENABLE_EXECUTION", "true")

    # Config values are provided but the environment wins.
    cfg = WriterRuntimeConfig(
        project_root="C:/Projects/Writer",
        writer_executable=None,
        python_executable=None,
        timeout_seconds=30,
        enable_execution=False,
    )
    assert cfg.project_root == "D:/other/Writer"
    assert cfg.writer_executable == "D:/other/Writer/writer.cmd"
    assert cfg.python_executable == "D:/other/Writer/python.exe"
    assert cfg.timeout_seconds == 120
    assert cfg.enable_execution is True
