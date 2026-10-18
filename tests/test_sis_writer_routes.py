"""Tests for SIS Writer discovery routes (read-only, no Writer required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sis.routes import writer as writer_routes
from sis.template_utils import SISTemplates


class _FakeDiscovery:
    def search(self, query):
        return {
            "ok": True,
            "results": [
                {"rank": 1, "runtime_name": "editorial", "score": 0.42,
                 "semantic_reasons": ["summary overlap"], "matched_capabilities": []}
            ],
        }

    def describe(self, name):
        return {
            "ok": True,
            "identity": {"title": "Editorial", "summary": "Book-level diagnosis"},
            "roles": ["diagnosis"],
            "capabilities": ["diagnose"],
            "scopes": ["book"],
            "stateful": False,
            "mutates_source": False,
        }

    def operations(self, name):
        return {"ok": True, "operations": [{"operation_name": "diagnose_book", "signature": "diagnose_book(target)"}]}


class _FakeBridge:
    def __init__(self):
        self.discovery = _FakeDiscovery()

    def status(self):
        return {
            "project_root": "C:/Projects/Writer",
            "execution_enabled": False,
            "writer_executable": "writer.exe",
        }


@pytest.fixture()
def client():
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    app.state.templates = SISTemplates(directory="sis/templates")
    app.mount("/static", StaticFiles(directory="sis/static"), name="static")
    app.include_router(writer_routes.router)
    return TestClient(app)


def test_writer_index_without_bridge(client):
    resp = client.get("/writer")
    assert resp.status_code == 200
    assert "Writer integration is not configured" in resp.text


def test_writer_search_with_bridge(client):
    client.app.state.writer_bridge = _FakeBridge()
    resp = client.get("/writer", params={"q": "repair a sentence"})
    assert resp.status_code == 200
    assert "editorial" in resp.text
    assert "discovery only" in resp.text


def test_writer_runtime_detail(client):
    client.app.state.writer_bridge = _FakeBridge()
    resp = client.get("/writer/runtime/editorial")
    assert resp.status_code == 200
    assert "diagnose_book" in resp.text
