# stephanie/memory/casebook_store.py
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Sequence

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Query

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.case_goal_state import CaseGoalStateORM
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.dynamic_scorable import DynamicScorableORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.scorable_factory import TargetType

_logger = logging.getLogger(__name__)


def _trunc(s: str | None, n: int = 200) -> str | None:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + "…"


class CaseBookStore(BaseSQLAlchemyStore):
    orm_model = CaseBookORM
    default_order_by = CaseBookORM.id.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "casebooks"

    def name(self) -> str:
        return self.name

    # ------------------------
    # CaseBook methods
    # ------------------------

    def get_by_name(self, name: str) -> CaseBookORM | None:
        def op(s):
            return s.query(CaseBookORM).filter_by(name=name).first()
        return self._run(op)

    def ensure_casebook(self, name: str, description: str = "", tag: str = "", meta: dict = None) -> CaseBookORM:
        def op(s):
            cb = s.query(CaseBookORM).filter_by(name=name).first()
            if cb:
                return cb
            cb = CaseBookORM(name=name, description=description, tag=tag, meta=meta)
            s.add(cb)
            return cb
        return self._run(op)

    def create_casebook(self, name, description="", tag="", meta=None):
        def op(s):
            
                cb = CaseBookORM(name=name, description=description, tag=tag, meta=meta)
                s.add(cb)
                return cb
        return self._run(op)

    def get_all_casebooks(self, limit: int = 100) -> List[CaseBookORM]:
        def op(s):
            return s.query(CaseBookORM).limit(limit).all()
        return self._run(op)

    def list_casebooks(
        self,
        *,
        agent_name: Optional[str] = None,
        tag: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[CaseBookORM]:
        def op(s):
            
                q = s.query(CaseBookORM)
                if agent_name is not None:
                    q = q.filter(CaseBookORM.agent_name == agent_name)
                if tag is not None:
                    q = q.filter(CaseBookORM.tag == tag)
                if pipeline_run_id is not None:
                    q = q.filter(CaseBookORM.pipeline_run_id == pipeline_run_id)
                return q.order_by(CaseBookORM.id.desc()).limit(limit).all()
        return self._run(op)

    def get_casebook(self, casebook_id: int) -> Optional[CaseBookORM]:
        def op(s):
            return s.get(CaseBookORM, casebook_id)
        return self._run(op)

    def get_casebooks(self) -> List[CaseBookORM]:
        def op(s):
            return s.query(CaseBookORM).all()
        return self._run(op)

    def count_cases(self, casebook_id: int) -> int:
        def op(s):
            return s.query(func.count(CaseORM.id)).filter_by(casebook_id=casebook_id).scalar() or 0

        return self._run(op)

    def get_for_run_id(self, run_id: int):
        def op(s):
            return s.query(CaseBookORM).filter_by(pipeline_run_id=run_id).first()
        return self._run(op)

    # ------------------------
    # Cases
    # ------------------------

    def get_cases_for_goal(self, goal_id):
        def op(s):
            return s.query(CaseORM).filter_by(goal_id=goal_id).all()
        return self._run(op)

    def get_cases_for_agent(self, agent_name):
        def op(s):
            return s.query(CaseORM).filter_by(agent_name=agent_name).all()
        return self._run(op)

    def get_cases_for_casebook(self, casebook_id: int):
        def op(s):
            return s.query(CaseORM).filter_by(casebook_id=casebook_id).all()
        return self._run(op)

    def get_case_by_id(self, case_id: int) -> Optional[CaseORM]:
        def op(s):
            return s.get(CaseORM, case_id)
        return self._run(op)

    def list_cases(
        self,
        *,
        casebook_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        goal_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[CaseORM]:
        def op(s):
            
                q = s.query(CaseORM)
                if casebook_id is not None:
                    q = q.filter(CaseORM.casebook_id == casebook_id)
                if agent_name is not None:
                    q = q.filter(CaseORM.agent_name == agent_name)
                if goal_id is not None:
                    q = q.filter(CaseORM.goal_id == goal_id)
                return q.order_by(CaseORM.id.desc()).limit(limit).all()
        return self._run(op)

    # ------------------------
    # Goal state
    # ------------------------

    def get_goal_state(self, casebook_id: int, goal_id: str):
        def op(s):
            return s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
        return self._run(op)

    def ensure_goal_state(self, casebook_id: int, goal_id: str, *, case_id: Optional[int] = None, quality: Optional[float] = None) -> CaseGoalStateORM:
        def op(s):
            
                state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                if state is None:
                    state = CaseGoalStateORM(
                        casebook_id=casebook_id,
                        goal_id=goal_id,
                        champion_case_id=case_id,
                        champion_quality=float(quality or 0.0),
                    )
                    s.add(state)
                return state
        return self._run(op)

    def upsert_goal_state(
        self,
        casebook_id: int,
        goal_id: str,
        case_id: Optional[int] = None,
        quality: Optional[float] = None,
        *,
        only_if_better: bool = False,
        improved: Optional[bool] = None,
        delta: Optional[float] = None,
        ema_alpha: float = 0.2,
    ) -> CaseGoalStateORM:
        def op(s):
            
                state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                if state is None:
                    state = CaseGoalStateORM(casebook_id=casebook_id, goal_id=goal_id)
                    s.add(state)
                if case_id is not None:
                    if only_if_better and (quality is not None):
                        if float(quality) > float(state.champion_quality or 0.0):
                            state.champion_case_id = case_id
                            state.champion_quality = float(quality)
                    else:
                        state.champion_case_id = case_id
                        if quality is not None:
                            state.champion_quality = float(quality)
                if improved is not None and delta is not None:
                    state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
                return state
        return self._run(op)

    def record_ab_result(self, casebook_id: int, goal_id: str, *, improved: bool, delta: float, ema_alpha: float = 0.2) -> CaseGoalStateORM:
        def op(s):
            
                state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                if not state:
                    state = CaseGoalStateORM(casebook_id=casebook_id, goal_id=goal_id)
                    s.add(state)
                state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
                return state
        return self._run(op)

    # ------------------------
    # Scorables
    # ------------------------

    def get_case_scorable_by_id(self, case_scorable_id: int) -> Optional[CaseScorableORM]:
        def op(s):
            return s.get(CaseScorableORM, case_scorable_id)
        return self._run(op)

    def list_scorables(self, case_id: int, role: str = None):
        def op(s):
            
                q = s.query(CaseScorableORM).filter_by(case_id=case_id)
                if role:
                    q = q.filter(CaseScorableORM.role == role)
                return q.all()
        return self._run(op)

    def add_scorable(
        self,
        case_id: int,
        pipeline_run_id: int,
        text: str,
        scorable_type: str = TargetType.DYNAMIC,
        meta: Optional[dict] = None,
        role: Optional[str] = None,
    ) -> DynamicScorableORM:
        def op(s):
            
                orm = DynamicScorableORM(
                    case_id=case_id,
                    pipeline_run_id=pipeline_run_id,
                    scorable_type=scorable_type,
                    text=text,
                    meta=meta or {},
                    role=role,
                )
                s.add(orm)
                return orm
        return self._run(op)

    # ------------------------
    # Helpers
    # ------------------------

    @staticmethod
    def _make_scorable_id(s: dict, case_id: int, idx: int) -> str:
        sid = s.get("id") or s.get("scorable_id")
        if sid:
            return str(sid)
        text = ""
        meta = s.get("meta") or {}
        if isinstance(meta, dict):
            text = meta.get("text", "") or meta.get("content", "")
        text = text or s.get("text", "") or s.get("content", "")
        if text:
            base = f"{case_id}:{idx}:{text}"
            return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
        return uuid.uuid4().hex
