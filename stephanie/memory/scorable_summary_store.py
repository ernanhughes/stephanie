from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.scorable_summary import ScorableSummaryORM


class ScorableSummaryStore(BaseSQLAlchemyStore):
    """
    Store for ScorableSummaryORM.

    Generic summaries for any scorable:
      - scorable_type: "document", "section", "chat_turn", ...
      - scorable_id:   string identifier for that scorable

    Supports:
      - insert: always create a new summary row
      - upsert: update or insert based on (scorable_type, scorable_id, tool_name,
                model_name, summary_kind)
      - upsert_many: batch upsert
      - list_for_scorable: all summaries for a given scorable
      - get_latest: latest summary for a given scorable
    """

    orm_model = ScorableSummaryORM
    default_order_by = "id"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "scorable_summaries"

    # ------------------------------------------------------------------
    # Create / update
    # ------------------------------------------------------------------

    def insert(self, summary_dict: Dict) -> ScorableSummaryORM:
        """
        Always insert a new ScorableSummary row.
        """
        def op(s):
            obj = ScorableSummaryORM(**summary_dict)
            s.add(obj)
            s.flush()  # get auto id

            if self.logger:
                self.logger.log(
                    "ScorableSummaryInserted",
                    {
                        "id": obj.id,
                        "scorable_type": obj.scorable_type,
                        "scorable_id": obj.scorable_id,
                        "tool_name": obj.tool_name,
                        "model_name": obj.model_name,
                        "summary_kind": obj.summary_kind,
                    },
                )
            return obj

        return self._run(op)

    def upsert(self, summary_dict: Dict) -> ScorableSummaryORM:
        """
        Update or insert a summary based on
        (scorable_type, scorable_id, tool_name, model_name, summary_kind).

        This gives you "one active summary per (tool, model, kind, scorable)".
        """
        def op(s):
            stype = summary_dict["scorable_type"]
            sid = summary_dict["scorable_id"]
            tool = summary_dict["tool_name"]
            model = summary_dict["model_name"]
            skind = summary_dict.get("summary_kind")

            q = (
                s.query(ScorableSummaryORM)
                .filter_by(
                    scorable_type=stype,
                    scorable_id=str(sid),
                    tool_name=tool,
                    model_name=model,
                )
            )
            if skind is not None:
                q = q.filter(ScorableSummaryORM.summary_kind == skind)

            existing = q.first()

            if existing:
                for key, value in summary_dict.items():
                    setattr(existing, key, value)
                obj = existing
                action = "ScorableSummaryUpdated"
            else:
                obj = ScorableSummaryORM(**summary_dict)
                s.add(obj)
                action = "ScorableSummaryInserted"

            s.flush()

            if self.logger:
                self.logger.log(
                    action,
                    {
                        "id": obj.id,
                        "scorable_type": obj.scorable_type,
                        "scorable_id": obj.scorable_id,
                        "tool_name": obj.tool_name,
                        "model_name": obj.model_name,
                        "summary_kind": obj.summary_kind,
                    },
                )

            return obj

        return self._run(op)

    def upsert_many(self, rows: Iterable[Dict]) -> List[ScorableSummaryORM]:
        """
        Batch upsert; same key as upsert().
        """
        def op(s):
            out: List[ScorableSummaryORM] = []
            for row in rows:
                stype = row["scorable_type"]
                sid = row["scorable_id"]
                tool = row["tool_name"]
                model = row["model_name"]
                skind = row.get("summary_kind")

                q = (
                    s.query(ScorableSummaryORM)
                    .filter_by(
                        scorable_type=stype,
                        scorable_id=str(sid),
                        tool_name=tool,
                        model_name=model,
                    )
                )
                if skind is not None:
                    q = q.filter(ScorableSummaryORM.summary_kind == skind)

                existing = q.first()

                if existing:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    obj = existing
                else:
                    obj = ScorableSummaryORM(**row)
                    s.add(obj)

                out.append(obj)

            s.flush()
            return out

        return self._run(op)

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def list_for_scorable(
        self,
        scorable_type: str,
        scorable_id: str,
        *,
        tool_name: Optional[str] = None,
        summary_kind: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[ScorableSummaryORM]:
        """
        List summaries for a given scorable; optionally filter by tool/kind.
        """
        def op(s):
            q = (
                s.query(ScorableSummaryORM)
                .filter_by(
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                )
                .order_by(
                    ScorableSummaryORM.created_at.desc().nulls_last(),
                    ScorableSummaryORM.id.desc(),
                )
            )
            if tool_name:
                q = q.filter(ScorableSummaryORM.tool_name == tool_name)
            if summary_kind:
                q = q.filter(ScorableSummaryORM.summary_kind == summary_kind)

            return q.offset(offset).limit(limit).all()

        return self._run(op)

    def get_latest(
        self,
        scorable_type: str,
        scorable_id: str,
        *,
        tool_name: Optional[str] = None,
        summary_kind: Optional[str] = None,
    ) -> Optional[ScorableSummaryORM]:
        """
        Get the most recent summary for a given scorable (optionally by tool/kind).
        """
        def op(s):
            q = (
                s.query(ScorableSummaryORM)
                .filter_by(
                    scorable_type=scorable_type,
                    scorable_id=str(scorable_id),
                )
                .order_by(
                    ScorableSummaryORM.created_at.desc().nulls_last(),
                    ScorableSummaryORM.id.desc(),
                )
            )
            if tool_name:
                q = q.filter(ScorableSummaryORM.tool_name == tool_name)
            if summary_kind:
                q = q.filter(ScorableSummaryORM.summary_kind == summary_kind)
            return q.first()

        return self._run(op)

    def list_recent(self, limit: int = 100) -> List[ScorableSummaryORM]:
        """
        Utility: see the most recent summaries across everything.
        """
        def op(s):
            q = (
                s.query(ScorableSummaryORM)
                .order_by(
                    ScorableSummaryORM.created_at.desc().nulls_last(),
                    ScorableSummaryORM.id.desc(),
                )
            )
            return q.limit(limit).all()

        return self._run(op)
