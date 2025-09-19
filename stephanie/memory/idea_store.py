# stephanie/memory/idea_store.py
from __future__ import annotations

from typing import List

from sqlalchemy import desc

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.idea import IdeaORM


class IdeaStore(BaseSQLAlchemyStore):
    orm_model = IdeaORM
    default_order_by = IdeaORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "ideas"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert / Update
    # -------------------
    def add_idea(self, idea_data: dict) -> IdeaORM:
        """Add a single idea to the database."""
        def op():
            s = self._scope()
            idea = IdeaORM(**idea_data)
            s.add(idea)
            s.flush()
            if self.logger:
                self.logger.log("IdeaInserted", idea.to_dict())
            return idea
        return self._run(op)

    def bulk_add_ideas(self, ideas_data: List[dict]) -> List[IdeaORM]:
        """Add multiple ideas at once."""
        def op():
            s = self._scope()
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
        def op():
            return self._scope().query(IdeaORM).filter(IdeaORM.goal_id == goal_id).all()
        return self._run(op)

    def get_top_ranked_ideas(self, limit: int = 5) -> List[IdeaORM]:
        """
        Get top-ranked ideas based on novelty + feasibility scores.
        Assumes scores are stored in extra_data JSON.
        """
        def op():
            return (
                self._scope().query(IdeaORM)
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
        def op():
            return (
                self._scope().query(IdeaORM)
                .filter(IdeaORM.focus_area == focus_area, IdeaORM.strategy == strategy)
                .all()
            )
        return self._run(op)

    def get_by_source(self, source: str) -> List[IdeaORM]:
        """Retrieve ideas by their origin (e.g., 'llm', 'survey_agent', 'evolved')."""
        def op():
            return self._scope().query(IdeaORM).filter(IdeaORM.source == source).all()
        return self._run(op)

    # -------------------
    # Delete / Clear
    # -------------------
    def delete_by_goal_id(self, goal_id: int) -> None:
        """Delete all ideas linked to a given goal."""
        def op():
            deleted = self._scope().query(IdeaORM).filter(IdeaORM.goal_id == goal_id).delete()
            if self.logger:
                self.logger.log("IdeasDeletedByGoal", {"goal_id": goal_id, "count": deleted})
        self._run(op)

    def clear_all(self) -> None:
        """Clear all ideas (useful for testing)."""
        def op():
            deleted = self._scope().query(IdeaORM).delete()
            if self.logger:
                self.logger.log("IdeasCleared", {"count": deleted})
        self._run(op)
