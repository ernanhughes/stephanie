# stephanie/memcubes/memcube_client.py
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from sqlalchemy import text

from stephanie.memory.memcube_store import MemcubeStore
from stephanie.memcubes.memcube_factory import MemCubeFactory
from stephanie.scoring.scorable import Scorable, ScorableType

# --- simple domain heuristic (same as earlier stub) ---
_HISTORY = ("war", "empire", "treaty", "century", "king", "dynasty")
_GEO     = ("river", "border", "capital", "country", "latitude", "longitude")
_SCI     = ("cell", "atom", "neuron", "quantum", "protein", "enzyme")
_TECH    = ("api", "database", "algorithm", "model", "gpu", "cloud")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

def _sanitize(s: str) -> str:
    s = EMAIL_RE.sub("<EMAIL>", s or "")
    s = PHONE_RE.sub("<PHONE>", s)
    return s

def _guess_domain_heuristic(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in _HISTORY): return "history"
    if any(k in q for k in _GEO):     return "geography"
    if any(k in q for k in _SCI):     return "science"
    if any(k in q for k in _TECH):    return "tech"
    return "general"


class MemCubeClient:
    """
    Async client over MemcubeStore with a tiny API used by risk prediction:
      - query_calibration(kind, filters, sort, limit) Hi good night
      - store_calibration(kind, payload)
      - guess_domain(question)

    Also provides:
      - store(blob)   # persist arbitrary Q/A traces as MemCube versions
      - get(cube_id)  # fetch MemCube by id
    """

    def __init__(self, store: MemcubeStore, logger=None):
        self.store = store
        self.logger = logger

    # ------------- Calibration I/O -------------
    async def query_calibration(
        self,
        kind: str,
        filters: Dict[str, Any],
        sort: Optional[list] = None,
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up the latest calibration for a domain (and kind).
        Stored in memcubes with source='risk-calibration' and extra_data JSON.
        """
        domain = str(filters.get("domain", "general"))

        def _op(session):
            # NOTE: extra_data is stored as JSON text; we search with LIKE for portability.
            # If your DB column is JSON/JSONB, swap to proper JSON operators.
            sql = """
                SELECT id, content, extra_data, created_at
                FROM memcubes
                WHERE source = 'risk-calibration'
                  AND model = 'risk'
                  AND extra_data LIKE :domain_like
                  AND extra_data LIKE :kind_like
                ORDER BY created_at DESC
                LIMIT :lim
            """
            rows = session.execute(
                text(sql),
                {
                    "domain_like": f'%\"domain\":\"{domain}\"%',
                    "kind_like":   f'%\"calibration_kind\":\"{kind}\"%',
                    "lim": int(max(1, limit)),
                },
            ).fetchall()

            if not rows:
                return None

            # Use the first (newest)
            r = rows[0]
            try:
                data = json.loads(r.extra_data or "{}")
            except Exception:
                data = {}

            # Common fields we expect callers to want
            out = {
                "id": r.id,
                "domain": data.get("domain", domain),
                "low_threshold": data.get("low_threshold"),
                "high_threshold": data.get("high_threshold"),
                "calibration_kind": data.get("calibration_kind", kind),
                "created_at": str(r.created_at) if hasattr(r, "created_at") else None,
                "payload": data,
            }
            return out

        return await asyncio.to_thread(self.store._run, _op)

    async def store_calibration(self, kind: str, payload: Dict[str, Any]) -> bool:
        """
        Persist a calibration record as a MemCube row.
        We store the full payload in extra_data and minimal content for searchability.
        """
        domain = str(payload.get("domain", "general"))
        content = json.dumps(
            {
                "calibration_kind": kind,
                "domain": domain,
                "summary": {
                    "low_threshold": payload.get("low_threshold"),
                    "high_threshold": payload.get("high_threshold"),
                },
            },
            ensure_ascii=False,
        )

        # Make a stable-ish scorable id per (kind,domain) to get version bumping
        scorable = Scorable(
            id=hash((kind, domain)),
            text=content,
            target_type=ScorableType.DOCUMENT,
        )
        cube = MemCubeFactory.from_scorable(scorable, version="auto")
        cube.source = "risk-calibration"
        cube.model = "risk"
        cube.sensitivity = "internal"
        cube.extra_data.update(
            {
                "calibration_kind": kind,
                "domain": domain,
                **payload,  # keep everything
            }
        )

        def _save():
            self.store.save_memcube(cube)
            return True

        ok = await asyncio.to_thread(_save)
        if self.logger:
            try:
                self.logger.log(
                    "RiskCalibrationSaved",
                    {"kind": kind, "domain": domain, "low": payload.get("low_threshold"), "high": payload.get("high_threshold")},
                )
            except Exception:
                pass
        return ok

    # ------------- Domain helper -------------
    async def guess_domain(self, question: str) -> str:
        return _guess_domain_heuristic(question)

    # ------------- Convenience methods (match your usage) -------------
    async def store(self, blob: Dict[str, Any], *, source: str = "memcube.store", model: str = "unknown", sensitivity: str = "public") -> str:
        """
        Persist an arbitrary interaction blob, similar to your example:
            memory.memcube.store({...})
        """
        # We keep the raw payload in extra_data, and a minimal, sanitized text in content.
        content = json.dumps(
            {
                "summary": {
                    "query": _sanitize(str(blob.get("query", "")))[:512],
                    "answer_head": _sanitize(str(blob.get("answer", "")))[:512],
                }
            },
            ensure_ascii=False,
        )

        scorable = Scorable(
            id=hash(content),
            text=content,
            target_type=ScorableType.DOCUMENT,
        )
        cube = MemCubeFactory.from_scorable(scorable, version="auto")
        cube.source = source
        cube.model = model
        cube.sensitivity = sensitivity
        cube.extra_data.update(blob)

        def _save():
            return self.store.save_memcube(cube)

        cube_id = await asyncio.to_thread(_save)
        if self.logger:
            try:
                self.logger.log("MemCubeBlobSaved", {"id": cube_id, "source": source})
            except Exception:
                pass
        return cube_id

    async def get(self, cube_id: str):
        """Fetch a MemCube by id (thin passthrough)."""
        return await asyncio.to_thread(self.store.get_memcube, cube_id)
