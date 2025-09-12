# stephanie/memory/calibration_event_store.py
import datetime
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from stephanie.models.calibration import CalibrationEventORM


class CalibrationEventStore:
    """
    Store for CalibrationEventORM.
    Provides persistence and query methods for calibration events.
    """

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "calibration_events"

    def add(self, event: CalibrationEventORM) -> CalibrationEventORM:
        """Insert a calibration event into the database."""
        try:
            self.session.add(event)
            self.session.commit()
            return event
        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("CalibrationEventInsertFailed", {"error": str(e)})
            raise

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

    def get_recent_domains(self, since: datetime) -> List[str]:
        """Return unique domains with calibration activity since the given time."""
        rows = (
            self.session.query(CalibrationEventORM.domain)
            .filter(CalibrationEventORM.timestamp >= since)
            .distinct()
            .all()
        )
        return [r[0] for r in rows if r[0]]