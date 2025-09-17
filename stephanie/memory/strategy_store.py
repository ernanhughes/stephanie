# stephanie/memory/strategy_store.py Much better
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

# Reuse the dataclass already defined in your agent:
# NOTE: this import avoids circulars because the agent doesn't import this module.
from stephanie.agents.summary.knowledge_infused_summarizer import StrategyProfile
from stephanie.models.strategy import StrategyProfileORM
from stephanie.models.strategy import StrategyProfile

# ---------- Interface-ish ----------
class IStrategyStore:
    def load(self, *, agent_name: str, scope: str = "default") -> StrategyProfile: ...
    def save(self, *, agent_name: str, profile: StrategyProfile, scope: str = "default") -> StrategyProfile: ...
    # Optional helpers:
    def reset(self, *, agent_name: str, scope: str = "default") -> StrategyProfile: ...
    def peek(self, *, agent_name: str, scope: str = "default") -> Optional[StrategyProfile]: ...


# ---------- DB-backed store ----------
class DBStrategyStore(IStrategyStore):
    def __init__(self, session: Session):
        self.session = session

    def _to_dataclass(self, row: StrategyProfileORM) -> StrategyProfile:
        return StrategyProfile.from_dict({
            "verification_threshold": row.verification_threshold,
            "pacs_weights": row.pacs_weights or {},
            "strategy_version": row.strategy_version,
            "last_updated": row.last_updated.timestamp() if row.last_updated else datetime.now().timestamp(),
        })

    def load(self, *, agent_name: str, scope: str = "default") -> StrategyProfile:
        row = (
            self.session.query(StrategyProfileORM)
            .filter_by(agent_name=agent_name, scope=scope)
            .one_or_none()
        )
        if row:
            return self._to_dataclass(row)

        # create a default one on first load
        default = StrategyProfile()
        self.save(agent_name=agent_name, profile=default, scope=scope)
        return default

    def save(self, *, agent_name: str, profile: StrategyProfile, scope: str = "default") -> StrategyProfile:
        row = (
            self.session.query(StrategyProfileORM)
            .filter_by(agent_name=agent_name, scope=scope)
            .one_or_none()
        )
        if row is None:
            row = StrategyProfileORM(
                agent_name=agent_name,
                scope=scope,
                pacs_weights=dict(profile.pacs_weights),
                verification_threshold=float(profile.verification_threshold),
                strategy_version=int(profile.strategy_version),
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )
            self.session.add(row)
        else:
            row.pacs_weights = dict(profile.pacs_weights)
            row.verification_threshold = float(profile.verification_threshold)
            row.strategy_version = int(profile.strategy_version)
            row.last_updated = datetime.now()

        self.session.commit()
        return self._to_dataclass(row)

    def reset(self, *, agent_name: str, scope: str = "default") -> StrategyProfile:
        row = (
            self.session.query(StrategyProfileORM)
            .filter_by(agent_name=agent_name, scope=scope)
            .one_or_none()
        )
        if row:
            self.session.delete(row)
            self.session.commit()
        return self.load(agent_name=agent_name, scope=scope)

    def peek(self, *, agent_name: str, scope: str = "default") -> Optional[StrategyProfile]:
        row = (
            self.session.query(StrategyProfileORM)
            .filter_by(agent_name=agent_name, scope=scope)
            .one_or_none()
        )
        return self._to_dataclass(row) if row else None


# ---------- JSON fallback store ----------
class JsonStrategyStore(IStrategyStore):
    """
    Simple file store for strategy. Handy for local/dev or when DB isn't available.
    One file per (agent, scope).
    """
    def __init__(self, root: str = "./runs/strategy"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_name: str, scope: str) -> Path:
        safe_agent = "".join(c for c in agent_name if c.isalnum() or c in ("-", "_")).strip() or "agent"
        safe_scope = "".join(c for c in scope if c.isalnum() or c in ("-", "_")).strip() or "default"
        return self.root / f"{safe_agent}__{safe_scope}.json"

    def _to_profile(self, d: Dict[str, Any]) -> StrategyProfile:
        return StrategyProfile.from_dict(d or {})

    def load(self, *, agent_name: str, scope: str = "default") -> StrategyProfile:
        p = self._path(agent_name, scope)
        if p.exists():
            return self._to_profile(json.loads(p.read_text()))
        prof = StrategyProfile()
        self.save(agent_name=agent_name, profile=prof, scope=scope)
        return prof

    def save(self, *, agent_name: str, profile: StrategyProfile, scope: str = "default") -> StrategyProfile:
        p = self._path(agent_name, scope)
        d = profile.to_dict()
        # human-friendly + atomic write
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        os.replace(tmp, p)
        return self._to_profile(d)

    def reset(self, *, agent_name: str, scope: str = "default") -> StrategyProfile:
        p = self._path(agent_name, scope)
        if p.exists():
            p.unlink()
        return self.load(agent_name=agent_name, scope=scope)

    def peek(self, *, agent_name: str, scope: str = "default") -> Optional[StrategyProfile]:
        p = self._path(agent_name, scope)
        if not p.exists():
            return None
        return self._to_profile(json.loads(p.read_text()))
