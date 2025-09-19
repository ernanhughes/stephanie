from __future__ import annotations

import datetime
import os
import json
from typing import Dict, List, Mapping, Union

import numpy as np
from sqlalchemy import case, func
from sqlalchemy.exc import SQLAlchemyError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.calibration import CalibrationEventORM


def _as_str(x, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, np.generic):
        x = x.item()
    return x if isinstance(x, str) else str(x)


def _as_float(x, default: float = 0.0) -> float:
    if x is None:
        return default
    if isinstance(x, np.generic):
        x = x.item()
    try:
        return float(x)
    except Exception:
        return default


def _as_bool(x, default: bool = False) -> bool:
    if isinstance(x, np.bool_):
        return bool(x.item())
    try:
        return bool(x)
    except Exception:
        return default


class CalibrationEventStore(BaseSQLAlchemyStore):
    """
    Store for CalibrationEventORM.
    Provides persistence and query methods for calibration events.
    """
    orm_model = CalibrationEventORM
    default_order_by = CalibrationEventORM.timestamp  # use column for BaseSQLAlchemyStore

    def __init__(self, session_maker, logger=None, model_dir: str = "data/calibration"):
        super().__init__(session_maker, logger)
        self.name = "calibration_events"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def add(self, event: Union[CalibrationEventORM, Mapping]) -> CalibrationEventORM:
        def op(s):
            if not isinstance(event, CalibrationEventORM):
                payload = dict(event)
                domain = _as_str(payload.get("domain"), "general")
                query = _as_str(payload.get("query"), "")
                raw_similarity = _as_float(payload.get("raw_similarity"), 0.0)
                scorable_id = _as_str(payload.get("scorable_id"), "unknown")
                scorable_type = _as_str(payload.get("scorable_type"), "unknown")
                entity_type_raw = payload.get("entity_type")
                entity_type = _as_str(entity_type_raw) if entity_type_raw is not None else None
                is_relevant = _as_bool(payload.get("is_relevant"), False)
                timestamp = payload.get("timestamp") or datetime.datetime.now()
                features = payload.get("features")

                event_obj = CalibrationEventORM(
                    domain=domain,
                    query=query,
                    raw_similarity=raw_similarity,
                    scorable_id=scorable_id,
                    scorable_type=scorable_type,
                    entity_type=entity_type,
                    is_relevant=is_relevant,
                    timestamp=timestamp,
                    **({"features": features} if "features" in CalibrationEventORM.__table__.columns else {}),
                )
            else:
                event_obj = event

            if getattr(event_obj, "query", "") is None:
                event_obj.query = ""

            s.add(event_obj)
            return event_obj

        return self._run(op)

    def get_by_domain(self, domain: str) -> List[CalibrationEventORM]:
        def op(s):
            
                return (
                    s.query(CalibrationEventORM)
                    .filter(CalibrationEventORM.domain == domain)
                    .order_by(CalibrationEventORM.timestamp.asc())
                    .all()
                )
        return self._run(op)

    def count_by_domain(self, domain: str) -> dict:
        def op(s):
            
                pos_expr = func.coalesce(
                    func.sum(case((CalibrationEventORM.is_relevant.is_(True), 1), else_=0)), 0
                )
                neg_expr = func.coalesce(
                    func.sum(case((CalibrationEventORM.is_relevant.is_(False), 1), else_=0)), 0
                )
                pos, neg = (
                    s.query(pos_expr.label("pos"), neg_expr.label("neg"))
                    .filter(CalibrationEventORM.domain == domain)
                    .one()
                )
                pos, neg = int(pos or 0), int(neg or 0)
                return {"pos": pos, "neg": neg, "total": pos + neg}
        return self._run(op)

    def get_recent(self, domain: str, limit: int = 100) -> List[CalibrationEventORM]:
        def op(s):
            
                return (
                    s.query(CalibrationEventORM)
                    .filter(CalibrationEventORM.domain == domain)
                    .order_by(CalibrationEventORM.timestamp.desc())
                    .limit(limit)
                    .all()
                )
        return self._run(op)

    def get_recent_domains(self, since: datetime.datetime) -> List[str]:
        def op(s):
            
                rows = (
                    s.query(CalibrationEventORM.domain)
                    .filter(CalibrationEventORM.timestamp >= since)
                    .distinct()
                    .all()
                )
                return [r[0] for r in rows if r[0]]
        return self._run(op)

    def fetch_events(self, domain: str, limit: int = 10000, require_features: bool = False) -> List[Dict]:
        def op(s):
            
                rows = (
                    s.query(CalibrationEventORM)
                    .filter(CalibrationEventORM.domain == domain)
                    .order_by(CalibrationEventORM.timestamp.desc())
                    .limit(limit)
                    .all()
                )
                out: List[Dict] = []
                for r in rows:
                    feats = None
                    if hasattr(r, "features"):
                        feats = r.features
                        if isinstance(feats, str):
                            try:
                                feats = json.loads(feats)
                            except Exception:
                                feats = None
                    if not isinstance(feats, dict):
                        feats = {
                            "coverage": float(getattr(r, "coverage", 0.0) or 0.0),
                            "correctness": float(getattr(r, "correctness", 0.0) or 0.0),
                            "coherence": float(getattr(r, "coherence", 0.0) or 0.0),
                            "citation_support": float(getattr(r, "citation_support", 0.0) or 0.0),
                        }
                        if require_features and not any(feats.values()):
                            continue
                    out.append({
                        "domain": r.domain,
                        "query": r.query,
                        "raw_similarity": float(r.raw_similarity or 0.0),
                        "is_relevant": bool(r.is_relevant),
                        "scorable_id": r.scorable_id,
                        "scorable_type": r.scorable_type,
                        "entity_type": r.entity_type,
                        "timestamp": r.timestamp.isoformat() if getattr(r, "timestamp", None) else None,
                        "features": feats,
                    })
                return out
        return self._run(op)

    def fetch_counts_by_label(self, domain: str) -> Dict[str, int]:
        return self.count_by_domain(domain)

    # --- File-based calibrator handling stays the same ---
    def load_calibrator(self, domain: str) -> dict | None:
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            gen = os.path.join(self.model_dir, "general_calibration.json")
            if os.path.exists(gen):
                with open(gen, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            if self.logger:
                self.logger.log("CalibrationLoadFailed", {"domain": domain, "error": str(e)})
            return None

    def save_calibrator(self, domain: str, model: dict) -> None:
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        tmp = path + ".tmp"
        payload = dict(model)
        payload.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
            if self.logger:
                self.logger.log("CalibrationSaved", {"domain": domain, "path": path})
        except Exception as e:
            if self.logger:
                self.logger.log("CalibrationSaveFailed", {"domain": domain, "error": str(e)})

    def upsert_calibrator(self, domain: str, model: dict) -> None:
        self.save_calibrator(domain, model)

    def delete_calibrator(self, domain: str) -> None:
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
