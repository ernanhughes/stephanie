# stephanie/memory/calibration_event_store.py
from __future__ import annotations

import datetime
import os
from typing import Dict, List, Mapping, Union

import numpy as np
from sqlalchemy import case, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.calibration import CalibrationEventORM


def _as_str(x, default: str = "") -> str:
    if x is None:
        return default
    # unwrap numpy scalars
    if isinstance(x, np.generic):
        x = x.item()
    try:
        return x if isinstance(x, str) else str(x)
    except Exception:
        return default


def _as_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, np.generic):
        x = x.item()
    try:
        return float(x)
    except Exception:
        return float(default)


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
    default_order_by = CalibrationEventORM.timestamp.desc()

    def __init__(self, session: Session, logger=None, model_dir: str = "data/calibration"):
        super().__init__(session, logger)
        self.name = "calibration_events"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def name(self) -> str:
        return self.name    

    def add(self, event: Union[CalibrationEventORM, Mapping]) -> CalibrationEventORM:
        """
        Insert a calibration event.
        - Accepts either a CalibrationEventORM or a dict-like mapping.
        - Coerces numpy types and fills safe defaults (never NULL for NOT NULL cols).
        """
        try:
            if not isinstance(event, CalibrationEventORM):
                payload = dict(event)  # shallow copy

                # Provide safe, non-null defaults
                domain = _as_str(payload.get("domain"), "general")
                query = _as_str(payload.get("query"), "")  # NOT NULL in DB
                raw_similarity = _as_float(payload.get("raw_similarity"), 0.0)
                scorable_id = _as_str(payload.get("scorable_id"), "unknown")
                scorable_type = _as_str(payload.get("scorable_type"), "unknown")
                entity_type_raw = payload.get("entity_type")
                entity_type = (
                    _as_str(entity_type_raw) if entity_type_raw is not None else None
                )
                is_relevant = _as_bool(payload.get("is_relevant"), False)
                timestamp = payload.get("timestamp") or datetime.datetime.now()

                # Optional JSON column (only set if your ORM has it)
                features = payload.get("features", None)

                event = CalibrationEventORM(
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

            # Final safeguard: NOT NULL columns must not be None
            if getattr(event, "query", "") is None:
                event.query = ""

            self.session.add(event)
            self.session.commit()
            return event

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("CalibrationEventInsertFailed", {"error": str(e)})
            raise

    def get_by_domain(self, domain: str) -> List[CalibrationEventORM]:
        try:
            return (
                self.session.query(CalibrationEventORM)
                .filter(CalibrationEventORM.domain == domain)
                .order_by(CalibrationEventORM.timestamp.asc())
                .all()
            )
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventFetchFailed", {"error": str(e), "domain": domain})
            raise

    def count_by_domain(self, domain: str) -> dict:
        """
        Return {'pos': <int>, 'neg': <int>, 'total': <int>} for a given domain.
        Uses a single aggregation query with CASE for performance.
        """
        try:
            pos_expr = func.coalesce(
                func.sum(case((CalibrationEventORM.is_relevant.is_(True), 1), else_=0)), 0
            )
            neg_expr = func.coalesce(
                func.sum(case((CalibrationEventORM.is_relevant.is_(False), 1), else_=0)), 0
            )

            pos, neg = (
                self.session.query(pos_expr.label("pos"), neg_expr.label("neg"))
                .filter(CalibrationEventORM.domain == domain)
                .one()
            )

            pos = int(pos or 0)
            neg = int(neg or 0)
            return {"pos": pos, "neg": neg, "total": pos + neg}

        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log(
                    "CalibrationEventCountFailed", {"error": str(e), "domain": domain}
                )
            # Preserve previous behavior on error: raise, or return zeros if you prefer
            raise

    def get_recent(self, domain: str, limit: int = 100) -> List[CalibrationEventORM]:
        try:
            return (
                self.session.query(CalibrationEventORM)
                .filter(CalibrationEventORM.domain == domain)
                .order_by(CalibrationEventORM.timestamp.desc())
                .limit(limit)
                .all()
            )
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventRecentFailed", {"error": str(e), "domain": domain})
            raise

    def get_recent_domains(self, since: datetime.datetime) -> List[str]:
        rows = (
            self.session.query(CalibrationEventORM.domain)
            .filter(CalibrationEventORM.timestamp >= since)
            .distinct()
            .all()
        )
        return [r[0] for r in rows if r[0]]

    def fetch_events(self, domain: str, limit: int = 10000, require_features: bool = False) -> List[Dict]:
        """
        Return recent events for a domain as dicts with a 'features' payload.

        Shape:
        {
          "domain": str,
          "query": str,
          "raw_similarity": float,
          "is_relevant": bool,
          "scorable_id": str|None,
          "scorable_type": str|None,
          "entity_type": str|None,
          "timestamp": iso8601|None,
          "features": {
              "coverage": float,
              "correctness": float,
              "coherence": float,
              "citation_support": float
          }
        }
        """
        try:
            rows = (
                self.session.query(CalibrationEventORM)
                .filter(CalibrationEventORM.domain == domain)
                .order_by(CalibrationEventORM.timestamp.desc())
                .limit(limit)
                .all()
            )

            out: List[Dict] = []
            for r in rows:
                # Try to pull features from a JSON column if your ORM has one.
                feats = None
                if hasattr(r, "features"):
                    feats = r.features
                    if isinstance(feats, str):
                        try:
                            feats = json.loads(feats)
                        except Exception:
                            feats = None

                # Fallback: synthesize zeros (or skip if require_features=True)
                if not isinstance(feats, dict):
                    feats = {
                        "coverage": float(getattr(r, "coverage", 0.0) or 0.0),
                        "correctness": float(getattr(r, "correctness", 0.0) or 0.0),
                        "coherence": float(getattr(r, "coherence", 0.0) or 0.0),
                        "citation_support": float(getattr(r, "citation_support", 0.0) or 0.0),
                    }
                    if require_features and not any(feats.values()):
                        continue  # skip rows with no usable features

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

        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventFetchEventsFailed", {"error": str(e), "domain": domain})
            raise

    # Optional shim for older callers:
    def fetch_counts_by_label(self, domain: str) -> Dict[str, int]:
        """
        Backwards-compat wrapper. Returns {"pos": int, "neg": int, "total": int}.
        """
        return self.count_by_domain(domain)

    # ---- NEW ----
    def load_calibrator(self, domain: str) -> dict | None:
        """
        Load a calibrator blob for a domain.
        Falls back to general if domain file is missing.
        Returns None if nothing is available.
        """
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            # fallback to general
            gen = os.path.join(self.model_dir, "general_calibration.json")
            if os.path.exists(gen):
                with open(gen, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            if self.logger:
                self.logger.log("CalibrationLoadFailed", {"domain": domain, "error": str(e)})
            return None

    # ---- NEW ----
    def save_calibrator(self, domain: str, model: dict) -> None:
        """
        Persist a calibrator blob for a domain (atomic write).
        """
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        tmp = path + ".tmp"
        payload = dict(model)
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
            if self.logger:
                self.logger.log("CalibrationSaved", {"domain": domain, "path": path})
        except Exception as e:
            if self.logger:
                self.logger.log("CalibrationSaveFailed", {"domain": domain, "error": str(e)})

    # Optional helpers (aliases)
    def upsert_calibrator(self, domain: str, model: dict) -> None:
        self.save_calibrator(domain, model)

    def delete_calibrator(self, domain: str) -> None:
        path = os.path.join(self.model_dir, f"{domain}_calibration.json")
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
