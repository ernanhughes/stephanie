# stephanie/memory/scorable_domain_store.py
from __future__ import annotations

import logging
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.scorable_domain import ScorableDomainORM

logger = logging.getLogger(__name__)


class ScorableDomainStore(BaseSQLAlchemyStore):
    """
    Store for domain classifications linked to any Scorable
    (documents, plan_traces, prompts, etc.)
    """
    orm_model = ScorableDomainORM
    default_order_by = ScorableDomainORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "scorable_domains"

    def name(self) -> str:
        return self.name

    def insert(self, data: dict) -> ScorableDomainORM:
        """
        Insert or update a domain classification entry.

        Expected dict keys:
            - scorable_id (str)
            - scorable_type (str)
            - domain (str)
            - score (float)
        """
        def op(s):
            stmt = (
                pg_insert(ScorableDomainORM)
                .values(**data)
                .on_conflict_do_update(
                    index_elements=["scorable_id", "scorable_type", "domain"],
                    set_={"score": data["score"]},
                )
                .returning(ScorableDomainORM.id)
            )
            
            inserted_id = s.execute(stmt).scalar()
            if self.logger and inserted_id:
                self.logger.log("DomainUpserted", data)
            return (
                s.query(ScorableDomainORM)
                .filter_by(
                    scorable_id=data["scorable_id"],
                    scorable_type=data["scorable_type"],
                    domain=data["domain"],
                )
                .first()
            )
        return self._run(op)

    def get_domains(self, scorable_id: str, scorable_type: str) -> List[ScorableDomainORM]:
        """Get all domain classifications for a scorable."""
        def op(s):
            return (
                s.query(ScorableDomainORM)
                .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
                .order_by(ScorableDomainORM.score.desc())
                .all()
            )
        return self._run(op)

    def delete_domains(self, scorable_id: str, scorable_type: str) -> None:
        """Delete all domain classifications for a scorable."""
        def op(s):
            
            s.query(ScorableDomainORM).filter_by(
                scorable_id=scorable_id, scorable_type=scorable_type
            ).delete()
            if self.logger:
                self.logger.log(
                    "DomainsDeleted",
                    {"scorable_id": scorable_id, "scorable_type": scorable_type},
                )
        self._run(op)

    def set_domains(
        self, scorable_id: str, scorable_type: str, domains: list[tuple[str, float]]
    ) -> None:
        """
        Clear and re-add domains for the scorable.

        :param domains: list of (domain, score) pairs
        """
        self.delete_domains(scorable_id, scorable_type)
        for domain, score in domains:
            self.insert(
                {
                    "scorable_id": scorable_id,
                    "scorable_type": scorable_type,
                    "domain": domain,
                    "score": float(score),
                }
            )

    def has_domains(self, scorable_id: str, scorable_type: str) -> bool:
        """Check if the given scorable has any domain classifications."""
        def op(s):
            return (
                s.query(ScorableDomainORM.id)
                .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
                .first()
                is not None
            )
        return self._run(op)

    def get_by_scorable(self, scorable_id: str, scorable_type: str) -> List[ScorableDomainORM]:
        """Get all domain classifications for a given scorable."""
        def op(s):
            result = (
                s.query(ScorableDomainORM)
                .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
                .all()
            )
            logger.debug(
                f"FetchedScorableDomain: scorable_id={scorable_id}, "
                f"scorable_type={scorable_type}, result_count={len(result)}"
            )
            return result
        return self._run(op)
