# stephanie/memory/calibration_event_store.py
import datetime
import os
import pickle
from typing import List, Optional, Dict, Any

from sqlalchemy import func, case
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from stephanie.models.calibration import CalibrationEventORM, CalibrationModelORM


class CalibrationEventStore:
    """
    Store for CalibrationEventORM.
    Provides persistence and query methods for calibration events.
    """

    def __init__(self, session: Session, logger=None, data_dir: str = "data/calibration"):
        self.session = session
        self.logger = logger
        self.name = "calibration_events"
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------ write ------------------

    def add(self, event: CalibrationEventORM) -> CalibrationEventORM:
        """Insert a calibration event into the database (NumPy-safe)."""
        try:
            # --- sanitize event fields to pure Python/JSON-safe types ---
            def _as_native(x):
                try:
                    import numpy as np
                except Exception:
                    np = None

                if np is not None:
                    if isinstance(x, np.ndarray):
                        return [_as_native(v) for v in x.tolist()]
                    if isinstance(x, (np.float16, np.float32, np.float64)):
                        return float(x)
                    if isinstance(x, (np.int8, np.int16, np.int32, np.int64, np.int_)):
                        return int(x)
                    if isinstance(x, (np.bool_,)):
                        return bool(x)

                if isinstance(x, dict):
                    return {str(_as_native(k)): _as_native(v) for k, v in x.items()}
                if isinstance(x, (list, tuple, set)):
                    return [_as_native(v) for v in x]
                return x

            # Cast scalar columns to safe types if they exist
            if hasattr(event, "raw_similarity") and event.raw_similarity is not None:
                event.raw_similarity = float(_as_native(event.raw_similarity))

            if hasattr(event, "is_relevant") and event.is_relevant is not None:
                event.is_relevant = bool(_as_native(event.is_relevant))

            if hasattr(event, "query") and event.query is not None and not isinstance(event.query, str):
                q = _as_native(event.query)
                event.query = q if isinstance(q, str) else str(q)[:2000]

            for attr in ("scorable_id", "scorable_type", "entity_type", "domain"):
                if hasattr(event, attr) and getattr(event, attr) is not None:
                    setattr(event, attr, str(_as_native(getattr(event, attr))))

            # JSON-ish payloads (e.g., .features) → sanitize recursively if present
            if hasattr(event, "features") and event.features is not None:
                event.features = _as_native(event.features)

            self.session.add(event)
            self.session.commit()
            return event
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("CalibrationEventInsertFailed", {"error": str(e)})
            raise

    # ------------------ read: basic ------------------

    def get_by_domain(self, domain: str) -> List[CalibrationEventORM]:
        """Fetch all calibration events for a given domain."""
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

    def count_by_domain(self, domain: str) -> int:
        """Count calibration events for a given domain."""
        try:
            return (
                self.session.query(CalibrationEventORM)
                .filter(CalibrationEventORM.domain == domain)
                .count()
            )
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventCountFailed", {"error": str(e), "domain": domain})
            raise

    def get_recent(self, domain: str, limit: int = 100) -> List[CalibrationEventORM]:
        """Fetch recent calibration events for monitoring/training."""
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
        """Return unique domains with calibration activity since the given time."""
        rows = (
            self.session.query(CalibrationEventORM.domain)
            .filter(CalibrationEventORM.timestamp >= since)
            .distinct()
            .all()
        )
        return [r[0] for r in rows if r[0]]

    # ------------------ read: expected by CalibrationManager ------------------

    def fetch_counts_by_label(self, domain: str) -> Dict[str, int]:
        """
        Return {"pos": n1, "neg": n0} for a domain.
        """
        try:
            q = (
                self.session.query(
                    func.sum(case((CalibrationEventORM.is_relevant == True, 1), else_=0)).label("pos"),  # noqa: E712
                    func.sum(case((CalibrationEventORM.is_relevant == False, 1), else_=0)).label("neg"),  # noqa: E712
                )
                .filter(CalibrationEventORM.domain == domain)
            )
            row = q.one()
            return {"pos": int(row.pos or 0), "neg": int(row.neg or 0)}
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventCountsFailed", {"error": str(e), "domain": domain})
            # Be resilient
            return {"pos": 0, "neg": 0}


    # Helper for manager._load_training()
    def fetch_events(self, domain: str, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        Return dicts with raw_similarity and is_relevant for a domain (newest first).
        """
        try:
            rows = (
                self.session.query(CalibrationEventORM)
                .filter(CalibrationEventORM.domain == domain)
                .order_by(CalibrationEventORM.timestamp.desc())
                .limit(limit)
                .all()
            )
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = r.to_dict() if hasattr(r, "to_dict") else {}
                if not d:
                    # manual mapping fallback
                    d = {
                        "raw_similarity": float(getattr(r, "raw_similarity", 0.0) or 0.0),
                        "is_relevant": bool(getattr(r, "is_relevant", False)),
                    }
                out.append({
                    "raw_similarity": float(d.get("raw_similarity", 0.0) or 0.0),
                    "is_relevant": bool(d.get("is_relevant", False)),
                })
            return out
        except SQLAlchemyError as e:
            if self.logger:
                self.logger.log("CalibrationEventFetchEventsFailed", {"error": str(e), "domain": domain})
            raise

    # -------------------- Calibrators (models) --------------------

    def persist_calibrator(self, domain: str, calibrator: object, kind: str = "unknown", threshold: float = 0.5) -> None:
        """
        Pickle and upsert the calibrator for a domain.
        """
        try:
            blob = pickle.dumps(calibrator, protocol=pickle.HIGHEST_PROTOCOL)
            existing = (
                self.session.query(CalibrationModelORM)
                .filter(CalibrationModelORM.domain == domain)
                .one_or_none()
            )
            if existing:
                existing.kind = kind
                existing.threshold = float(threshold)
                existing.payload = blob
                existing.updated_at = datetime.datetime.now()
            else:
                row = CalibrationModelORM(
                    domain=domain,
                    kind=kind,
                    threshold=float(threshold),
                    payload=blob,
                )
                self.session.add(row)

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("CalibrationPersistFailed", {"error": str(e), "domain": domain})
            raise

    def load_calibrator(self, domain: str):
        """
        Load and unpickle the calibrator for a domain.
        Returns (calibrator, meta_dict) or None if missing.
        """
        try:
            row = (
                self.session.query(CalibrationModelORM)
                .filter(CalibrationModelORM.domain == domain)
                .one_or_none()
            )
            if not row:
                return None
            calibrator = pickle.loads(row.payload)
            meta = {"kind": row.kind, "threshold": float(row.threshold), "updated_at": row.updated_at}
            return calibrator, meta
        except Exception as e:
            if self.logger:
                self.logger.log("CalibrationLoadFailed", {"error": str(e), "domain": domain})
            # Do not raise — manager will fall back gracefully
            return None
