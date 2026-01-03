from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.exc import IntegrityError

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.scorable_identifier import IdentifierORM, ScorableIdentifierORM


class ScorableIdentifierStore(BaseSQLAlchemyStore):
    """
    One-stop store:
      - creates/gets IdentifierORM (type,value)
      - links scorable_type+scorable_id -> identifier_id
      - lists identifiers/links for a scorable

    Requires DB constraints:
      - identifiers: UNIQUE(identifier_type, identifier_value)
      - scorable_identifiers: UNIQUE(scorable_type, scorable_id, identifier_id)
    """

    # BaseSQLAlchemyStore expects an orm_model, but here we manage two.
    # Set it to the link table since that's the "primary" thing we query.
    orm_model = ScorableIdentifierORM
    default_order_by = ScorableIdentifierORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "scorable_identifiers"

    # -------------------------
    # Internal helpers (same session)
    # -------------------------

    def _get_identifier(self, s, *, identifier_type: str, identifier_value: str) -> Optional[ScorableIdentifierORM]:
        return (
            s.query(ScorableIdentifierORM)
            .filter_by(identifier_type=identifier_type, identifier_value=identifier_value)
            .first()
        )

    def _get_link(self, s, *, scorable_type: str, scorable_id: str, identifier_id: int) -> Optional[ScorableIdentifierORM]:
        return (
            s.query(ScorableIdentifierORM)
            .filter_by(
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                identifier_id=identifier_id,
            )
            .first()
        )

    # -------------------------
    # Public API
    # -------------------------

    def list_identifiers(self, *, scorable_type: str, scorable_id: str):
        def op(s):
            return (
                s.query(IdentifierORM)
                 .join(ScorableIdentifierORM, ScorableIdentifierORM.identifier_id == IdentifierORM.id)
                 .filter(
                    ScorableIdentifierORM.scorable_type == scorable_type,
                    ScorableIdentifierORM.scorable_id == str(scorable_id),
                 )
                 .all()
            )
        return self._run(op)

    def get_or_create_identifier_id(
        self,
        *,
        identifier_type: str,
        identifier_value: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Returns IdentifierORM.id for (identifier_type, identifier_value),
        creating if missing.
        """
        def op(s):
            existing = self._get_identifier(
                s,
                identifier_type=identifier_type,
                identifier_value=identifier_value,
            )
            if existing:
                return existing.id

            obj = ScorableIdentifierORM(
                identifier_type=identifier_type,
                identifier_value=identifier_value,
                name=name,
                description=description,
            )
            s.add(obj)
            s.flush()  # populate obj.id
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            # race: someone inserted first
            def op_retry(s):
                existing = self._get_identifier(
                    s,
                    identifier_type=identifier_type,
                    identifier_value=identifier_value,
                )
                return existing.id if existing else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def link_identifier_id(
        self,
        *,
        scorable_type: str,
        scorable_id: str,
        identifier_id: int,
    ) -> int:
        """
        Returns ScorableIdentifierORM.id (link row id) for (scorable, identifier_id),
        creating if missing.
        """
        def op(s):
            existing = self._get_link(
                s,
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                identifier_id=identifier_id,
            )
            if existing:
                return existing.id

            obj = ScorableIdentifierORM(
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                identifier_id=identifier_id,
            )
            s.add(obj)
            s.flush()
            return obj.id

        try:
            return self._run(op)
        except IntegrityError:
            # race: someone linked first
            def op_retry(s):
                existing = self._get_link(
                    s,
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    identifier_id=identifier_id,
                )
                return existing.id if existing else None

            out = self._run(op_retry)
            if out is None:
                raise
            return out

    def get_or_create_link(
        self,
        *,
        scorable_type: str,
        scorable_id: str,
        identifier_type: str,
        identifier_value: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience: create/get identifier + create/get link in ONE call.

        Returns:
            {
              "identifier_id": ...,
              "link_id": ...
            }
        """
        # simplest: two calls (still safe) — but we can do it in one transaction too.
        # We'll do it in ONE _run for better atomicity and less session churn.
        def op(s):
            ident = self._get_identifier(
                s,
                identifier_type=identifier_type,
                identifier_value=identifier_value,
            )
            if not ident:
                ident = ScorableIdentifierORM(
                    identifier_type=identifier_type,
                    identifier_value=identifier_value,
                    name=name,
                    description=description,
                )
                s.add(ident)
                s.flush()

            link = self._get_link(
                s,
                scorable_type=scorable_type,
                scorable_id=str(scorable_id),
                identifier_id=ident.id,
            )
            if not link:
                link = ScorableIdentifierORM(
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    identifier_id=ident.id,
                )
                s.add(link)
                s.flush()

            return {"identifier_id": ident.id, "link_id": link.id}

        try:
            return self._run(op)
        except IntegrityError:
            # race: either identifier or link was created concurrently.
            # retry by reading both.
            def op_retry(s):
                ident = self._get_identifier(
                    s,
                    identifier_type=identifier_type,
                    identifier_value=identifier_value,
                )
                if not ident:
                    return None

                link = self._get_link(
                    s,
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                    identifier_id=ident.id,
                )
                return {"identifier_id": ident.id, "link_id": (link.id if link else None)}

            out = self._run(op_retry)
            if not out or out.get("link_id") is None:
                raise
            return out

    def list_links_for_scorable(self, *, scorable_type: str, scorable_id: str) -> List[ScorableIdentifierORM]:
        def op(s):
            return (
                s.query(ScorableIdentifierORM)
                .filter_by(scorable_type=scorable_type, scorable_id=str(scorable_id))
                .all()
            )
        return self._run(op)

    def list_identifiers_for_scorable(self, *, scorable_type: str, scorable_id: str) -> List[ScorableIdentifierORM]:
        """
        Returns IdentifierORM rows for a given scorable.
        """
        def op(s):
            return (
                s.query(ScorableIdentifierORM)
                .join(ScorableIdentifierORM, ScorableIdentifierORM.identifier_id == ScorableIdentifierORM.id)
                .filter(
                    ScorableIdentifierORM.scorable_type == scorable_type,
                    ScorableIdentifierORM.scorable_id == str(scorable_id),
                )
                .all()
            )
        return self._run(op)
