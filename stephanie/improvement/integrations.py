"""Bounded read-only adapters for external evidence and Praxis comparison."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .models import Evidence


@dataclass(frozen=True)
class ReadResult:
    state: str  # verified_source_record | missing_record | unavailable_service | unknown
    source: str
    records: tuple[dict, ...] = ()
    detail: str | None = None


def _get_json(base_url: str, path: str, *, params: dict | None = None,
              timeout: float = 3.0) -> dict:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("integration adapters require an HTTP loopback base URL")
    url = base_url.rstrip("/") + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"GET returned HTTP {response.status}")
        data = json.loads(response.read(2_000_001))
    if len(json.dumps(data, ensure_ascii=False)) > 2_000_000:
        raise ValueError("response exceeded 2 MB read cap")
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def read_worker_strategies(base_url: str, *, timeout: float = 3.0) -> ReadResult:
    try:
        payload = _get_json(base_url, "/v1/strategies", timeout=timeout)
        records = tuple(payload.get("strategies", []))
        return ReadResult("verified_source_record" if records else "inconclusive", "worker:/v1/strategies",
                          records, None if records else "successful read, empty catalogue")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return ReadResult("unavailable_service", "worker:/v1/strategies", detail=str(exc)[:300])


def read_worker_executions(base_url: str, *, timeout: float = 3.0) -> ReadResult:
    try:
        payload = _get_json(base_url, "/v1/executions", timeout=timeout)
        records = tuple(payload.get("executions", []))
        return ReadResult("verified_source_record" if records else "inconclusive", "worker:/v1/executions",
                          records, "bounded source window; Worker ledger remains authoritative" if records else "successful read, no execution records in source window")
    except HTTPError as exc:
        state = "missing_record" if exc.code == 404 else "unavailable_service"
        return ReadResult(state, "worker:/v1/executions", detail=f"HTTP {exc.code}")
    except (URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return ReadResult("unavailable_service", "worker:/v1/executions", detail=str(exc)[:300])


def read_mission_episodes(base_url: str, *, limit: int = 20,
                               timeout: float = 3.0) -> ReadResult:
    limit = max(1, min(int(limit), 200))
    try:
        payload = _get_json(base_url, "/api/work-episodes", params={"limit": limit}, timeout=timeout)
        records = tuple(payload.get("episodes", []))
        return ReadResult("verified_source_record" if records else "inconclusive", "mission:/api/work-episodes",
                          records, "episode index only; no inferred Worker association" if records else "successful read, no episodes returned")
    except HTTPError as exc:
        state = "missing_record" if exc.code == 404 else "unavailable_service"
        return ReadResult(state, "mission:/api/work-episodes", detail=f"HTTP {exc.code}")
    except (URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return ReadResult("unavailable_service", "mission:/api/work-episodes", detail=str(exc)[:300])


def read_mission_signals(base_url: str, *, limit: int = 20,
                         timeout: float = 3.0) -> ReadResult:
    limit = max(1, min(int(limit), 500))
    try:
        payload = _get_json(base_url, "/api/work-observation/signals", params={"limit": limit}, timeout=timeout)
        records = tuple(payload.get("signals", []))
        return ReadResult("verified_source_record" if records else "inconclusive", "mission:/api/work-observation/signals",
                          records, "source identities preserved; signal is not an accepted Work Outcome" if records else "successful read, no signals returned")
    except HTTPError as exc:
        state = "missing_record" if exc.code == 404 else "unavailable_service"
        return ReadResult(state, "mission:/api/work-observation/signals", detail=f"HTTP {exc.code}")
    except (URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        return ReadResult("unavailable_service", "mission:/api/work-observation/signals", detail=str(exc)[:300])


def as_evidence(source_system: str, source_identity: str, record: dict, *,
                observed_facts: tuple[str, ...] = (),
                limitations: tuple[str, ...] = ()) -> Evidence:
    """Make a minimal source-bound Stephanie reference, retaining state unknowns."""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    source_digest = hashlib.sha256(payload.encode()).hexdigest()
    record_id = str(record.get("request_id") or record.get("id") or record.get("episode_id") or source_identity)
    ident = f"{source_system}:{record_id}"
    return Evidence(evidence_id=hashlib.sha256(ident.encode()).hexdigest()[:24],
                    source_system=source_system, source_identity=ident,
                    source_digest=source_digest,
                    observation_id=str(record.get("observation_id")) if record.get("observation_id") else None,
                    task_id=str(record.get("task_id")) if record.get("task_id") else None,
                    execution_id=str(record.get("execution_id") or record.get("request_id")) if record.get("execution_id") or record.get("request_id") else None,
                    strategy_id=str(record.get("strategy_id")) if record.get("strategy_id") else None,
                    evaluation_status="UNKNOWN", acceptance_status="UNKNOWN",
                    observed_facts=observed_facts, limitations=limitations + ("External source remains authoritative; this is a bounded reference copy.",))


def compare_with_praxis(baseline: list, candidate: list, *, praxis_module=None) -> dict:
    """Call Praxis's canonical comparer; accepts its record types without reimplementing."""
    if praxis_module is None:
        from praxis.effectiveness import compare_effectiveness
    else:
        compare_effectiveness = praxis_module.compare_effectiveness
    comparison = compare_effectiveness(baseline, candidate)
    return {"mode": "existing_callable_implementation", "comparison": comparison.to_dict(),
            "authority": "advisory", "operational_evaluation": False}
