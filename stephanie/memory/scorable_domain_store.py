from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from stephanie.models.scorable_domain import ScorableDomainORM


class ScorableDomainStore:
    """
    Store for domain classifications linked to any Scorable
    (documents, plan_traces, prompts, etc.)
    """

    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scorable_domains"

    def insert(self, data: dict) -> ScorableDomainORM:
        """
        Insert or update a domain classification entry.

        Expected dict keys:
            - scorable_id (str)
            - scorable_type (str)
            - domain (str)
            - score (float)
        """
        stmt = (
            pg_insert(ScorableDomainORM)
            .values(**data)
            .on_conflict_do_update(
                index_elements=["scorable_id", "scorable_type", "domain"],
                set_={"score": data["score"]},
            )
            .returning(ScorableDomainORM.id)
        )

        result = self.session.execute(stmt)
        inserted_id = result.scalar()
        self.session.commit()

        if inserted_id:
            self.logger.log("DomainUpserted", data)

        return (
            self.session.query(ScorableDomainORM)
            .filter_by(
                scorable_id=data["scorable_id"],
                scorable_type=data["scorable_type"],
                domain=data["domain"],
            )
            .first()
        )

    def get_domains(self, scorable_id: str, scorable_type: str) -> list[ScorableDomainORM]:
        """
        Get all domain classifications for a scorable.
        """
        return (
            self.session.query(ScorableDomainORM)
            .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
            .order_by(ScorableDomainORM.score.desc())
            .all()
        )

    def delete_domains(self, scorable_id: str, scorable_type: str):
        """
        Delete all domain classifications for a scorable.
        """
        self.session.query(ScorableDomainORM).filter_by(
            scorable_id=scorable_id, scorable_type=scorable_type
        ).delete()
        self.session.commit()

        if self.logger:
            self.logger.log("DomainsDeleted", {
                "scorable_id": scorable_id,
                "scorable_type": scorable_type
            })

    def set_domains(self, scorable_id: str, scorable_type: str, domains: list[tuple[str, float]]):
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
        """
        Check if the given scorable has any domain classifications.
        """
        exists = (
            self.session.query(ScorableDomainORM.id)
            .filter_by(scorable_id=scorable_id, scorable_type=scorable_type)
            .first()
        )
        return exists is not None


    def get_by_scorable(self, scorable_id: str, scorable_type: str) -> List[ScorableDomainORM]:
        """
        Check if the given scorable has any domain classifications.
        """
        result = self.session.query(ScorableDomainORM) \
            .filter_by(scorable_id=scorable_id, scorable_type=scorable_type).all()
        self.logger.log("FetchedScorableDomain", {"scorable_id": scorable_id, "scorable_type": scorable_type, "result": result})
        return result
