from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.reflection import ReflectionORM


class ReflectionStore(BaseSQLAlchemyStore):
    """
    SQLAlchemy-backed store for reflections.

    Intended usage:
      - save_micro_reflection(...) from InformationReflectionAgent
      - get_for_task(...) when doing topic-level analysis
      - list_recent(...) for global / macro reflection or analysis
    """

    def __init__(self, session_maker):
        super().__init__(session_maker=session_maker)

    # ------------------------------------------------------------------ #
    # Write API
    # ------------------------------------------------------------------ #

    def save_micro_reflection(
        self,
        *,
        task_id: str,
        trace_id: int,
        draft_text: str,
        reference_text: Optional[str],
        score: Optional[int],
        problems: List[Dict[str, Any]],
        action_plan: List[str],
        raw_text: str,
        level: str = "micro",
    ) -> ReflectionORM:
        """
        Create and persist a single micro-level reflection.
        """
        def op(session):
            obj = ReflectionORM.from_micro(
                task_id=task_id,
                trace_id=trace_id,
                level=level,
                draft_text=draft_text,
                reference_text=reference_text,
                score=score,
                problems=problems,
                action_plan=action_plan,
                raw_text=raw_text,
            )
            session.add(obj)
            session.flush()  # populate obj.id
            return obj

        return self._run(op)

    # ------------------------------------------------------------------ #
    # Read API
    # ------------------------------------------------------------------ #

    def get_by_id(self, reflection_id: int) -> Optional[ReflectionORM]:
        def op(session):
            return session.get(ReflectionORM, reflection_id)
        return self._run(op)

    def get_for_task(
        self,
        task_id: str,
        *,
        limit: int = 50,
        level: Optional[str] = None,
    ) -> List[ReflectionORM]:
        """
        Fetch reflections for a given task (most recent first).
        """
        def op(session):
            q = session.query(ReflectionORM).filter(
                ReflectionORM.task_id == task_id
            )
            if level:
                q = q.filter(ReflectionORM.level == level)
            q = q.order_by(ReflectionORM.created_at.desc()).limit(limit)
            return q.all()

        return self._run(op)

    def list_recent(
        self,
        *,
        limit: int = 50,
        level: Optional[str] = None,
    ) -> List[ReflectionORM]:
        """
        Fetch most recent reflections globally, optionally filtered by level.
        """
        def op(session):
            q = session.query(ReflectionORM)
            if level:
                q = q.filter(ReflectionORM.level == level)
            q = q.order_by(ReflectionORM.created_at.desc()).limit(limit)
            return q.all()

        return self._run(op)
