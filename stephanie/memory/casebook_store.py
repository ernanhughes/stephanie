# stephanie/memory/casebook_store.py
from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Query

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.case_goal_state import CaseGoalStateORM
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM, CaseAttributeORM
from stephanie.models.dynamic_scorable import DynamicScorableORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.scorable import ScorableType

_logger = logging.getLogger(__name__)

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
            s.flush()
            return cb
        return self._run(op)

    def create_casebook(self, name, description: str = "", tag: str = "", meta: dict | None = None):
        def op(s):
            cb = CaseBookORM(name=name, description=description, tag=tag, meta=meta)
            s.add(cb)
            s.flush()
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

            order_col = getattr(CaseBookORM, "created_at", None)
            q = q.order_by(order_col.desc() if order_col is not None else CaseBookORM.id.desc())
            return q.limit(limit).all()
        return self._run(op)

    def get_casebook(self, casebook_id: int) -> Optional[CaseBookORM]:
        def op(s):
            return s.get(CaseBookORM, casebook_id)
        return self._run(op)

    def get_casebooks(self) -> List[CaseBookORM]:
        def op(s):
            return s.query(CaseBookORM).all()
        return self._run(op)

    def get_casebooks_by_tag(
        self,
        tag: str,
        *,
        limit: int = 200,
        agent_name: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
    ):
        """Return CaseBookORM rows filtered by the simple `tag` column."""
        def op(s):
            q = s.query(CaseBookORM).filter(CaseBookORM.tag == tag)
            if agent_name is not None:
                q = q.filter(CaseBookORM.agent_name == agent_name)
            if pipeline_run_id is not None:
                q = q.filter(CaseBookORM.pipeline_run_id == pipeline_run_id)

            order_col = getattr(CaseBookORM, "created_at", None)
            q = q.order_by(order_col.desc() if order_col is not None else CaseBookORM.id.desc())
            return q.limit(limit).all()
        return self._run(op)

    def count_cases(self, casebook_id: int) -> int:
        def op(s):
            return s.query(func.count(CaseORM.id)).filter_by(casebook_id=casebook_id).scalar() or 0
        return self._run(op)

    def get_for_run_id(self, run_id: int):
        def op(s):
            return s.query(CaseBookORM).filter_by(pipeline_run_id=run_id).first()
        return self._run(op)

    def get_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default") -> Optional[CaseBookORM]:
        def op(s):
            return (
                s.query(CaseBookORM)
                 .filter_by(pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag)
                 .first()
            )
        return self._run(op)

    def ensure_casebook_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default") -> CaseBookORM:
        def op(s):
            row = (
                s.query(CaseBookORM)
                 .filter_by(pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag)
                 .first()
            )
            if row:
                return row
            name = f"cb:{agent_name or 'all'}:{pipeline_run_id or 'all'}:{tag}"
            cb = CaseBookORM(
                name=name,
                description="Scoped casebook",
                pipeline_run_id=pipeline_run_id,
                agent_name=agent_name,
                tag=tag,
            )
            s.add(cb)
            s.flush()
            return cb
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

            order_col = getattr(CaseORM, "created_at", None)
            q = q.order_by(order_col.desc() if order_col is not None else CaseORM.id.desc())
            return q.limit(limit).all()
        return self._run(op)

    def ensure_case(self, casebook_id: int, goal_text: str, agent_name: str) -> CaseORM:
        def op(s):
            goal = s.query(GoalORM).filter_by(goal_text=goal_text).one_or_none()
            goal_id = goal.id if goal else None
            if goal is None:
                g = GoalORM(goal_text=goal_text, description=f"Auto-created for casebook {casebook_id}")
                s.add(g) 
                s.flush()
                goal_id = g.id

            existing = (s.query(CaseORM)
                        .filter_by(casebook_id=casebook_id, goal_id=goal_id, agent_name=agent_name)
                        .order_by(CaseORM.created_at.desc())
                        .first())
            if existing:
                return existing

            case = CaseORM(casebook_id=casebook_id, goal_id=goal_id, agent_name=agent_name)
            s.add(case)
            s.flush()
            return case
        return self._run(op)

    # Keep a single copy of these helpers
    def ensure_goal_state_for_case(self, casebook_id: int, goal_text: str, goal_id: str) -> CaseGoalStateORM:
        def op(s):
            state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id).one_or_none()
            if state is None:
                if not goal_id:
                    g = s.query(GoalORM).filter_by(goal_text=goal_text).first()
                    goal_id_local = g.id if g else None
                else:
                    goal_id_local = goal_id
                state = CaseGoalStateORM(casebook_id=casebook_id, goal_id=goal_id_local)
                s.add(state)
                s.flush()
            return state
        return self._run(op)

    # ------------------------
    # Goal state
    # ------------------------

    def get_goal_state(self, casebook_id: int, goal_id: str):
        def op(s):
            return s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
        return self._run(op)

    def ensure_goal_state(
        self,
        casebook_id: int,
        goal_id: str,
        *,
        case_id: Optional[int] = None,
        quality: Optional[float] = None,
    ) -> CaseGoalStateORM:
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
                s.flush()
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
                # Prefer ORM helper if present
                try:
                    state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
                except Exception:
                    prev_run_ix = int(getattr(state, "run_ix", 0) or 0)
                    state.run_ix = prev_run_ix + 1
                    if improved:
                        state.wins = (state.wins or 0) + 1
                    else:
                        state.losses = (state.losses or 0) + 1
                    prev = float(state.avg_delta or 0.0)
                    alpha = float(ema_alpha)
                    state.avg_delta = (1.0 - alpha) * prev + alpha * float(delta)
                    v = state.avg_delta
                    state.trust = max(-1.0, min(1.0, v))
            s.flush()
            return state
        return self._run(op)

    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int:
        def op(s):
            state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
            if state is None:
                state = CaseGoalStateORM(casebook_id=casebook_id, goal_id=goal_id)
                s.add(state)
                s.flush()
            before = int(getattr(state, "run_ix", 0) or 0)
            state.run_ix = before + 1
            s.flush()
            return state.run_ix
        return self._run(op)

    def record_ab_result(self, casebook_id: int, goal_id: str, *, improved: bool, delta: float, ema_alpha: float = 0.2) -> CaseGoalStateORM:
        def op(s):
            state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
            if not state:
                state = CaseGoalStateORM(casebook_id=casebook_id, goal_id=goal_id)
                s.add(state)
            try:
                state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
            except Exception:
                prev_run_ix = int(getattr(state, "run_ix", 0) or 0)
                state.run_ix = prev_run_ix + 1
                if improved:
                    state.wins = (state.wins or 0) + 1
                else:
                    state.losses = (state.losses or 0) + 1
                prev = float(state.avg_delta or 0.0)
                alpha = float(ema_alpha)
                state.avg_delta = (1.0 - alpha) * prev + alpha * float(delta)
                v = state.avg_delta
                state.trust = max(-1.0, min(1.0, v))
            s.flush()
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
        scorable_type: str = ScorableType.DYNAMIC,
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
            s.flush()
            return orm
        return self._run(op)

    def add_case(
        self,
        casebook_id: int,
        goal_id: str,
        agent_name: str,
        goal_text: Optional[str] = None,
        mars_summary: Optional[str] = None,
        scores: Optional[dict] = None,
        scorables: Optional[list[dict]] = None,
        prompt_text: Optional[str] = None,
        meta: Optional[dict] = None,
        response_texts: Optional[list[str]] = None,
    ) -> CaseORM:
        def op(s):
            case = CaseORM(
                casebook_id=casebook_id,
                goal_id=goal_id,
                agent_name=agent_name,
                prompt_text=prompt_text,
                meta=meta or {},
            )
            s.add(case)
            s.flush()  # need case.id for scorable ids

            # 1) Explicit scorables
            added = 0
            if scorables:
                for idx, item in enumerate(scorables):
                    safe_sid = self._make_scorable_id(item, case.id, idx)
                    cs = CaseScorableORM(
                        case_id=case.id,
                        scorable_id=safe_sid,
                        scorable_type=item.get("type") or item.get("target_type") or "document",
                        role=(item.get("role") or "output"),
                        rank=item.get("rank"),
                        meta=item.get("meta") or {},
                    )
                    s.add(cs)
                    added += 1

            # 2) Assistant response scorables
            if response_texts:
                for idx, resp in enumerate(response_texts):
                    if not resp or not resp.strip():
                        continue
                    sc_dict = {"text": resp, "role": "assistant"}
                    safe_sid = self._make_scorable_id(sc_dict, case.id, idx)
                    cs = CaseScorableORM(
                        case_id=case.id,
                        scorable_id=safe_sid,
                        scorable_type="document",
                        role="assistant",
                        rank=idx,
                        meta={"text": resp},
                    )
                    s.add(cs)
                    added += 1

            s.flush()
            return case
        return self._run(op)

    # ------------------------
    # Retrieval helpers w/ meta filtering
    # ------------------------

    def _apply_json_meta_filter(self, q: Query, column, meta_filter: Optional[Dict]) -> Tuple[Query, bool]:
        """
        Try to filter JSON(meta) at the DB level.
        Returns (query, db_filtered). If db_filtered is False, caller should Python-filter rows.
        """
        if not meta_filter:
            return q, True

        # Use the session from the query to inspect dialect
        bind = q.session.get_bind()
        dialect = (bind.dialect.name if bind is not None else "").lower()

        if dialect == "postgresql":
            # JSONB containment: column @> meta_filter
            try:
                q = q.filter(column.contains(meta_filter))
                return q, True
            except Exception:
                return q, False

        if dialect == "sqlite":
            # emulate via json_extract
            try:
                conds = [func.json_extract(column, f'$.{k}') == v for k, v in meta_filter.items()]
                if conds:
                    q = q.filter(and_(*conds))
                    return q, True
            except Exception:
                pass
            return q, False

        return q, False

    def get_by_case(
        self,
        *,
        casebook_name: str,
        case_id: int,
        role: Optional[str] = None,
        scorable_type: Optional[str] = None,
        meta_filter: Optional[dict] = None,
        limit: int = 50,
    ):
        """
        Return scorable rows for a given (casebook_name, case_id), newest first.
        Prefer DynamicScorableORM; fallback to CaseScorableORM if none.
        """
        def op(s):
            cb = s.query(CaseBookORM).filter_by(name=casebook_name).first()
            if not cb:
                return []

            # DynamicScorable path (direct text)
            q1 = (
                s.query(DynamicScorableORM)
                 .join(CaseORM, DynamicScorableORM.case_id == CaseORM.id)
                 .filter(CaseORM.casebook_id == cb.id, DynamicScorableORM.case_id == case_id)
            )
            if role:
                q1 = q1.filter(DynamicScorableORM.role == role)
            if scorable_type:
                q1 = q1.filter(DynamicScorableORM.scorable_type == scorable_type)
            order_col = getattr(DynamicScorableORM, "created_at", None)
            q1 = q1.order_by(desc(order_col) if order_col is not None else DynamicScorableORM.id.desc())
            dyn_rows = q1.limit(limit).all()

            # Optional Python-side meta filter (DynamicScorable.meta)
            if meta_filter:
                dyn_rows = [
                    r for r in dyn_rows
                    if isinstance(r.meta, dict) and all(r.meta.get(k) == v for k, v in meta_filter.items())
                ]

            if dyn_rows:
                return dyn_rows

            # Legacy CaseScorable fallback (meta["text"])
            q2 = (
                s.query(CaseScorableORM)
                 .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
                 .filter(CaseORM.casebook_id == cb.id, CaseScorableORM.case_id == case_id)
            )
            if role:
                q2 = q2.filter(CaseScorableORM.role == role)
            if scorable_type:
                q2 = q2.filter(CaseScorableORM.scorable_type == scorable_type)

            q2, db_filtered = self._apply_json_meta_filter(q2, CaseScorableORM.meta, meta_filter)
            order_col2 = getattr(CaseScorableORM, "created_at", None)
            q2 = q2.order_by(desc(order_col2) if order_col2 is not None else CaseScorableORM.id.desc())
            rows = q2.limit(limit).all()
            return rows
        return self._run(op)

    # ------------------------
    # Convenience retrieval
    # ------------------------

    def get_recent_cases(
        self,
        casebook_id: int,
        goal_id: str,
        *,
        limit: int = 10,
        only_accepted: bool = False,
        include_champion: bool = True,
        min_quality: Optional[float] = None,
    ) -> List[CaseORM]:
        def op(s):
            q = (
                s.query(CaseORM)
                 .filter(CaseORM.casebook_id == casebook_id, CaseORM.goal_id == goal_id)
            )
            order_col = getattr(CaseORM, "created_at", None)
            q = q.order_by(order_col.desc() if order_col is not None else CaseORM.id.desc())
            recent_all = q.limit(max(limit * 3, limit)).all()

            champion_case = None
            champion_id = None
            try:
                state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                champion_id = getattr(state, "champion_case_id", None)
                if include_champion and champion_id:
                    champion_case = next((c for c in recent_all if c.id == champion_id), None)
                    if champion_case is None:
                        champion_case = s.get(CaseORM, champion_id)
            except Exception:
                pass

            def is_accepted(case: CaseORM) -> bool:
                if champion_id and case.id == champion_id:
                    return True
                meta = getattr(case, "meta", {}) or {}
                if meta.get("accepted") is True:
                    return True
                if min_quality is not None:
                    try:
                        qv = float(meta.get("quality", float("-inf")))
                        if qv >= float(min_quality):
                            return True
                    except Exception:
                        pass
                return False

            accepted = [c for c in recent_all if is_accepted(c)] if only_accepted else []
            result: List[CaseORM] = []
            if include_champion and champion_case:
                result.append(champion_case)

            if only_accepted:
                for c in accepted:
                    if champion_case and c.id == getattr(champion_case, "id", None):
                        continue
                    result.append(c)
                if len(result) < limit:
                    for c in recent_all:
                        if (champion_case and c.id == getattr(champion_case, "id", None)) or c in result:
                            continue
                        result.append(c)
                        if len(result) >= limit:
                            break
            else:
                for c in recent_all:
                    if champion_case and c.id == getattr(champion_case, "id", None):
                        continue
                    result.append(c)
                    if len(result) >= limit + (1 if champion_case else 0):
                        break

            return result[:limit]
        return self._run(op)

    def list_scorables_by_role(self, case_id: int, role: str, limit: int = 500):
        """Convenience for SIS to fetch items by role."""
        def op(s):
            return (s.query(CaseScorableORM)
                      .filter(CaseScorableORM.case_id == case_id,
                              CaseScorableORM.role == role)
                      .order_by(CaseScorableORM.id.desc())
                      .limit(limit)
                      .all())
        return self._run(op)

    def get_pool_for_goal(
        self,
        casebook_id: int,
        goal_id: str,
        exclude_ids: Optional[Sequence[Optional[int]]] = None,
        *,
        limit: int = 200,
        include_champion: bool = False,
        only_accepted: bool = False,
        min_quality: Optional[float] = None,
    ) -> List[CaseORM]:
        def op(s):
            excl = {int(x) for x in (exclude_ids or []) if x is not None}

            q = (
                s.query(CaseORM)
                 .filter(CaseORM.casebook_id != casebook_id, CaseORM.goal_id == goal_id)
            )
            if excl:
                q = q.filter(~CaseORM.id.in_(excl))

            if not include_champion:
                try:
                    state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                    champ_id = getattr(state, "champion_case_id", None)
                    if champ_id:
                        q = q.filter(CaseORM.id != champ_id)
                except Exception:
                    pass

            order_col = getattr(CaseORM, "created_at", None)
            q = q.order_by(order_col.desc() if order_col is not None else CaseORM.id.desc())
            candidates = q.limit(max(limit * 3, limit)).all()

            if only_accepted or (min_quality is not None):
                def is_accepted(case: CaseORM) -> bool:
                    try:
                        state = s.query(CaseGoalStateORM).filter_by(casebook_id=casebook_id, goal_id=goal_id).one_or_none()
                        if state and state.champion_case_id and case.id == state.champion_case_id:
                            return True
                    except Exception:
                        pass
                    meta = getattr(case, "meta", {}) or {}
                    if only_accepted and meta.get("accepted") is True:
                        return True
                    if min_quality is not None:
                        try:
                            return float(meta.get("quality", float("-inf"))) >= float(min_quality)
                        except Exception:
                            return False
                    return not only_accepted

                candidates = [c for c in candidates if is_accepted(c)]

            return candidates[:limit]
        return self._run(op)

    def update_scorable_meta(self, case_scorable_id: int, patch: dict) -> Optional[CaseScorableORM]:
        """Shallow-merge fields into CaseScorableORM.meta and return the row."""
        def op(s):
            row = s.get(CaseScorableORM, case_scorable_id)
            if not row:
                return None
            meta = dict(row.meta or {})
            meta.update(patch or {})
            row.meta = meta
            s.add(row)
            return row
        return self._run(op)


    def set_case_attr(self, case_id: int, key: str, *, 
                    value_text: str | None = None,
                    value_num: float | None = None,
                    value_bool: bool | None = None,
                    value_json: dict | list | None = None):
        def op(s):
            row = (s.query(CaseAttributeORM)
                    .filter(CaseAttributeORM.case_id == case_id, CaseAttributeORM.key == key)
                    .one_or_none())
            if row is None:
                row = CaseAttributeORM(case_id=case_id, key=key)
                s.add(row)
            row.value_text = value_text
            row.value_num  = value_num
            row.value_bool = value_bool
            row.value_json = value_json
            s.flush()
            return row
        return self._run(op)

    def get_best_by_attrs(
        self,
        casebook_id: int,
        group_keys: list[str],     # e.g. ["paper_id","section_name","case_kind"]
        score_weights: dict[str,float] = {"knowledge_score":0.7, "verification_score":0.3},
        role: str | None = None,   # filter the scorable role if you want
        limit_per_group: int = 1,
        group_filter: dict[str, str] | None = None,  # optional WHERE on attributes
    ):
        """
        Generic 'best per group' using arbitrary attribute keys.
        Returns rows with: case_id, group_attrs(json), scores, composite_score.
        """
        def op(s):
            # Build a pivot of attributes per case via subquery
            # (Using text for brevity; you can write it with SA core if you prefer.)
            keys_sql = ",".join([f"MAX(CASE WHEN a.key='{k}' THEN a.value_text END) AS {k}" for k in group_keys])
            where_filter = ""
            if group_filter:
                conds = []
                for k,v in group_filter.items():
                    conds.append(f"MAX(CASE WHEN a.key='{k}' THEN a.value_text END) = :gf_{k}")
                if conds:
                    where_filter = "HAVING " + " AND ".join(conds)

            sql = f"""
            WITH attrs AS (
            SELECT c.id AS case_id, c.casebook_id,
                    {keys_sql}
            FROM cases c
            JOIN case_attributes a ON a.case_id = c.id
            WHERE c.casebook_id = :cbid
            GROUP BY c.id, c.casebook_id
            {where_filter}
            ),
            scored AS (
            SELECT
                attrs.*,
                ds.id AS ds_id,
                COALESCE((ds.meta->>'knowledge_score')::float, 0) AS knowledge_score,
                COALESCE((ds.meta->>'verification_score')::float, 0) AS verification_score,
                ds.created_at
            FROM attrs
            JOIN dynamic_scorables ds ON ds.case_id = attrs.case_id
            { "WHERE ds.role = :role" if role else "" }
            ),
            ranked AS (
            SELECT *,
                ({score_weights.get("knowledge_score",0)} * knowledge_score
            + {score_weights.get("verification_score",0)} * verification_score) AS composite_score,
                ROW_NUMBER() OVER (
                PARTITION BY {", ".join(group_keys)}
                ORDER BY ({score_weights.get("knowledge_score",0)} * knowledge_score
                        + {score_weights.get("verification_score",0)} * verification_score) DESC,
                        created_at DESC
                ) AS rn
            FROM scored
            )
            SELECT *
            FROM ranked
            WHERE rn <= :n
            ORDER BY {", ".join(group_keys)}, composite_score DESC, created_at DESC
            """

            params = {"cbid": casebook_id, "n": limit_per_group}
            if role:
                params["role"] = role
            if group_filter:
                for k,v in group_filter.items():
                    params[f"gf_{k}"] = v

            return list(s.execute(sql, params))
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


