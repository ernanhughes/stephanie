# stephanie/memory/idea_store.py
from __future__ import annotations

from typing import Any, Iterable, List

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.idea import IdeaORM


class IdeaStore(BaseSQLAlchemyStore):
    orm_model = IdeaORM
    default_order_by = IdeaORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "ideas"

    # -------------------
    # Insert / Update
    # -------------------
    def add_idea(self, idea_data: dict) -> IdeaORM:
        """Add a single idea to the database."""
        def op(s):
            idea = IdeaORM(**idea_data)
            s.add(idea)
            s.flush()
            if self.logger:
                self.logger.log("IdeaInserted", idea.to_dict())
            return idea
        return self._run(op)

    def bulk_add_ideas(self, ideas_data: List[dict]) -> List[IdeaORM]:
        """Add multiple ideas at once."""
        def op(s):
            ideas = [IdeaORM(**data) for data in ideas_data]
            s.add_all(ideas)
            if self.logger:
                self.logger.log("IdeasBulkInserted", {"count": len(ideas)})
            return ideas
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_by_goal_id(self, goal_id: int) -> List[IdeaORM]:
        """Retrieve all ideas associated with a specific goal."""
        def op(s):
            return s.query(IdeaORM).filter(IdeaORM.goal_id == goal_id).all()
        return self._run(op)

    def get_top_ranked_ideas(self, limit: int = 5) -> List[IdeaORM]:
        """
        Get top-ranked ideas based on novelty + feasibility scores.
        Assumes scores are stored in extra_data JSON.
        """
        def op(s):
            return (
                s.query(IdeaORM)
                .order_by(
                    IdeaORM.extra_data["novelty_score"].desc(),
                    IdeaORM.extra_data["feasibility_score"].desc(),
                )
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_by_focus_area_and_strategy(self, focus_area: str, strategy: str) -> List[IdeaORM]:
        """Retrieve ideas filtered by domain and strategy."""
        def op(s):
            return (
                s.query(IdeaORM)
                .filter(IdeaORM.focus_area == focus_area, IdeaORM.strategy == strategy)
                .all()
            )
        return self._run(op)

    def get_by_source(self, source: str) -> List[IdeaORM]:
        """Retrieve ideas by their origin (e.g., 'llm', 'survey_agent', 'evolved')."""
        def op(s):
            return s.query(IdeaORM).filter(IdeaORM.source == source).all()
        return self._run(op)

    # -------------------
    # Delete / Clear
    # -------------------
    def delete_by_goal_id(self, goal_id: int) -> None:
        """Delete all ideas linked to a given goal."""
        def op(s):
            deleted = s.query(IdeaORM).filter(IdeaORM.goal_id == goal_id).delete()
            if self.logger:
                self.logger.log("IdeasDeletedByGoal", {"goal_id": goal_id, "count": deleted})
        self._run(op)

    def clear_all(self) -> None:
        """Clear all ideas (useful for testing)."""
        def op(s):
            deleted = s.query(IdeaORM).delete()
            if self.logger:
                self.logger.log("IdeasCleared", {"count": deleted})
        self._run(op)


    def upsert_ideas(self, ideas: Iterable[Any]) -> List[IdeaORM]:
        """
        Upsert a batch of ideas.

        Accepts:
          - Pydantic Idea models (with .dict())
          - Plain dicts
          - IdeaORM instances

        Uses `id` as the primary key:
          - If an IdeaORM with that id exists -> update its fields
          - Otherwise -> insert a new row
        """
        def op(s):
            col_names = {c.name for c in IdeaORM.__table__.columns}
            upserted: List[IdeaORM] = []
            inserted = 0
            updated = 0

            for obj in ideas:
                # Normalise to a dict or an IdeaORM instance
                if isinstance(obj, IdeaORM):
                    # Merge directly into the session and treat as updated/inserted
                    instance = s.merge(obj)
                    upserted.append(instance)
                    updated += 1
                    continue

                if hasattr(obj, "dict"):
                    # Pydantic model (e.g., Idea)
                    data = obj.dict()
                elif isinstance(obj, dict):
                    data = obj
                else:
                    # Best-effort fallback: use __dict__
                    data = {
                        k: v for k, v in vars(obj).items()
                        if not k.startswith("_")
                    }

                idea_id = data.get("id")

                # Try to find existing row
                existing = None
                if idea_id is not None:
                    # `query().get` works across SQLAlchemy 1.4/2.0
                    existing = s.query(IdeaORM).get(idea_id)

                if existing is not None:
                    # Update only known columns (except id)
                    for k, v in data.items():
                        if k in col_names and k != "id":
                            setattr(existing, k, v)
                    upserted.append(existing)
                    updated += 1
                else:
                    # Insert new row
                    filtered = {k: v for k, v in data.items() if k in col_names}
                    instance = IdeaORM(**filtered)
                    s.add(instance)
                    upserted.append(instance)
                    inserted += 1

            if self.logger:
                self.logger.log(
                    "IdeasUpserted",
                    {
                        "count": len(upserted),
                        "inserted": inserted,
                        "updated": updated,
                    },
                )
            return upserted

        return self._run(op)
