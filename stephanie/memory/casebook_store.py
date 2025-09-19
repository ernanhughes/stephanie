# stephanie/memory/casebook_store.py
from __future__ import annotations

import hashlib
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Query, Session

from stephanie.memory.sqlalchemy_store import BaseSQLAlchemyStore
from stephanie.models.case_goal_state import CaseGoalStateORM
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.dynamic_scorable import DynamicScorableORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.scorable_factory import TargetType

import logging
_logger = logging.getLogger(__name__)


def _trunc(s: str | None, n: int = 200) -> str | None:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + "…"


class CaseBookStore(BaseSQLAlchemyStore):
    orm_model = CaseBookORM
    default_order_by = CaseBookORM.id.desc()
    
    def __init__(self, session: Session, logger=None):
        super().__init__(session, logger)
        self.name = "casebooks"
        _logger.info("CaseBookStore.__init__(name=%s)", self.name)

    def name(self) -> str:
        return self.name
    
    def get_by_name(self, name: str):
        _logger.info("get_by_name(name=%s) -> querying", name)
        row = self.session.query(CaseBookORM).filter_by(name=name).first()
        _logger.info("get_by_name(name=%s) -> %s", name, f"id={row.id}" if row else "None")
        return row

    def ensure_casebook(self, name: str, description: str = "", tag: str = "", meta: dict = None) -> CaseBookORM:
        _logger.info("ensure_casebook(name=%s, tag=%s)", name, tag)
        cb = self.get_by_name(name)
        if cb:
            _logger.info("ensure_casebook -> exists id=%s", cb.id)
            return cb
        cb = CaseBookORM(name=name, description=description, tag=tag, meta=meta)
        self.session.add(cb)
        self.session.commit()
        _logger.info("ensure_casebook -> created id=%s", cb.id)
        return cb

    def create_casebook(self, name, description="", tag="", meta=None):
        _logger.info("create_casebook(name=%s, tag=%s)", name, tag)
        cb = CaseBookORM(name=name, description=description, tag=tag, meta=meta)
        self.session.add(cb)
        self.session.commit()
        _logger.info("create_casebook -> id=%s", cb.id)
        return cb

    def count_cases(self, casebook_id: int) -> int:
        _logger.info("count_cases(casebook_id=%s)", casebook_id)
        cnt = self.session.query(func.count(CaseORM.id)).filter_by(casebook_id=casebook_id).scalar() or 0
        _logger.info("count_cases -> %s", cnt)
        return cnt

    def _apply_json_meta_filter(self, q: Query, column, meta_filter: Optional[Dict]) -> Tuple[Query, bool]:
        """
        Try to filter JSON meta in the database.
        Returns (query, db_filtered). If db_filtered is False, caller should Python-filter.
        """
        if not meta_filter:
            return q, True

        dialect = (self.session.bind.dialect.name if self.session.bind is not None else "").lower()
        _logger.info("_apply_json_meta_filter(dialect=%s, keys=%s)", dialect, list(meta_filter.keys()))

        if dialect == "postgresql":
            q = q.filter(column.contains(meta_filter))
            _logger.info("_apply_json_meta_filter -> using JSONB containment")
            return q, True

        if dialect == "sqlite":
            conds = [func.json_extract(column, f'$.{k}') == v for k, v in meta_filter.items()]
            if conds:
                q = q.filter(and_(*conds))
                _logger.info("_apply_json_meta_filter -> using sqlite json_extract")
                return q, True
            _logger.info("_apply_json_meta_filter -> sqlite no conds, fallback to Python filter")
            return q, False

        _logger.info("_apply_json_meta_filter -> unsupported dialect, Python filter")
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
        Prefers DynamicScorableORM; falls back to CaseScorableORM if none.
        """
        _logger.info("get_by_case(casebook_name=%s, case_id=%s, role=%s, scorable_type=%s, limit=%s)",
                     casebook_name, case_id, role, scorable_type, limit)
        cb = self.get_by_name(casebook_name)
        if not cb:
            _logger.warning("get_by_case -> casebook not found: %s", casebook_name)
            return []

        # ---- DynamicScorableORM path (has .text directly) ----
        q = (
            self.session.query(DynamicScorableORM)
            .join(CaseORM, DynamicScorableORM.case_id == CaseORM.id)
            .filter(CaseORM.casebook_id == cb.id, DynamicScorableORM.case_id == case_id)
        )
        if role:
            q = q.filter(DynamicScorableORM.role == role)
        if scorable_type:
            q = q.filter(DynamicScorableORM.scorable_type == scorable_type)

        order_col = getattr(DynamicScorableORM, "created_at", None)
        q = q.order_by(desc(order_col) if order_col is not None else DynamicScorableORM.id.desc())
        dyn_rows = q.limit(limit).all()
        _logger.info("get_by_case -> DynamicScorableORM fetched=%s", len(dyn_rows))

        if meta_filter:
            before = len(dyn_rows)
            dyn_rows = [
                r for r in dyn_rows
                if isinstance(r.meta, dict) and all(r.meta.get(k) == v for k, v in meta_filter.items())
            ]
            _logger.info("get_by_case -> meta Python-filtered Dynamic rows %s -> %s", before, len(dyn_rows))

        if dyn_rows:
            _logger.info("get_by_case -> returning %s DynamicScorableORM rows", len(dyn_rows))
            return dyn_rows

        # ---- Legacy CaseScorableORM fallback (text usually in meta['text']) ----
        _logger.info("get_by_case -> falling back to CaseScorableORM")
        q2 = (
            self.session.query(CaseScorableORM)
            .join(CaseORM, CaseScorableORM.case_id == CaseORM.id)
            .filter(CaseORM.casebook_id == cb.id, CaseScorableORM.case_id == case_id)
        )
        if role:
            q2 = q2.filter(CaseScorableORM.role == role)
        if scorable_type:
            q2 = q2.filter(CaseScorableORM.scorable_type == scorable_type)

        q2, db_filtered_legacy = self._apply_json_meta_filter(q2, CaseScorableORM.meta, meta_filter)

        order_col2 = getattr(CaseScorableORM, "created_at", None)
        q2 = q2.order_by(desc(order_col2) if order_col2 is not None else CaseScorableORM.id.desc())
        rows = q2.limit(limit).all()
        _logger.info("get_by_case -> CaseScorableORM fetched=%s (db_filtered=%s)", len(rows), db_filtered_legacy)

        return rows

    def get_all_casebooks(self, limit: int = 100) -> List[CaseBookORM]:
        _logger.info("get_all_casebooks(limit=%s)", limit)
        rows = self.session.query(CaseBookORM).limit(limit).all()
        _logger.info("get_all_casebooks -> %s rows", len(rows))
        return rows

    def get_cases_for_goal(self, goal_id):
        _logger.info("get_cases_for_goal(goal_id=%s)", goal_id)
        rows = self.session.query(CaseORM).filter_by(goal_id=goal_id).all()
        _logger.info("get_cases_for_goal -> %s rows", len(rows))
        return rows

    def get_cases_for_agent(self, agent_name):
        _logger.info("get_cases_for_agent(agent_name=%s)", agent_name)
        rows = self.session.query(CaseORM).filter_by(agent_name=agent_name).all()
        _logger.info("get_cases_for_agent -> %s rows", len(rows))
        return rows

    def get_for_run_id(self, run_id: int):
        _logger.info("get_for_run_id(run_id=%s)", run_id)
        row = self.session.query(CaseBookORM).filter_by(pipeline_run_id=run_id).first()
        _logger.info("get_for_run_id -> %s", f"id={row.id}" if row else "None")
        return row

    def get_cases_for_casebook(self, casebook_id: int):
        _logger.info("get_cases_for_casebook(casebook_id=%s)", casebook_id)
        rows = self.session.query(CaseORM).filter_by(casebook_id=casebook_id).all()
        _logger.info("get_cases_for_casebook -> %s rows", len(rows))
        return rows

    # in CaseBookStore
    def get_goal_state(self, casebook_id: int, goal_id: str):
        _logger.info("get_goal_state(casebook_id=%s, goal_id=%s)", casebook_id, goal_id)
        row = (self.session.query(CaseGoalStateORM)
               .filter_by(casebook_id=casebook_id, goal_id=goal_id)
               .one_or_none())
        _logger.info("get_goal_state -> %s", "found" if row else "None")
        return row

    def get_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default"):
        _logger.info("get_scope(run_id=%s, agent=%s, tag=%s)", pipeline_run_id, agent_name, tag)
        q = self.session.query(CaseBookORM).filter_by(
            pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag
        )
        row = q.first()
        _logger.info("get_scope -> %s", f"id={row.id}" if row else "None")
        return row

    def ensure_casebook_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default"):
        _logger.info("ensure_casebook_scope(run_id=%s, agent=%s, tag=%s)", pipeline_run_id, agent_name, tag)
        cb = self.get_scope(pipeline_run_id, agent_name, tag)
        if cb: 
            _logger.info("ensure_casebook_scope -> exists id=%s", cb.id)
            return cb
        name = f"cb:{agent_name or 'all'}:{pipeline_run_id or 'all'}:{tag}"
        cb = CaseBookORM(name=name, description="Scoped casebook",
                         pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag)
        self.session.add(cb)
        self.session.commit()
        _logger.info("ensure_casebook_scope -> created id=%s", cb.id)
        return cb

    # Scoped retrieval (strict)
    def get_cases_for_goal_in_casebook(self, casebook_id: int, goal_id: str):
        _logger.info("get_cases_for_goal_in_casebook(casebook_id=%s, goal_id=%s)", casebook_id, goal_id)
        rows = (self.session.query(CaseORM)
                .filter_by(casebook_id=casebook_id, goal_id=goal_id).all())
        _logger.info("get_cases_for_goal_in_casebook -> %s rows", len(rows))
        return rows

    # Flexible retrieval (union of scopes, ordered by specificity)
    def get_cases_for_goal_scoped(self, goal_id: str,
                                  scopes: list[tuple[str|None, str|None, str]]):
        _logger.info("get_cases_for_goal_scoped(goal_id=%s, scopes=%s)", goal_id, scopes)
        casebook_ids = []
        for p,a,t in scopes:
            cb = self.get_scope(p, a, t)
            if cb: 
                casebook_ids.append(cb.id)
        if not casebook_ids: 
            _logger.info("get_cases_for_goal_scoped -> no scoped casebooks")
            return []
        rows = (self.session.query(CaseORM)
                .filter(CaseORM.goal_id==goal_id, CaseORM.casebook_id.in_(casebook_ids))
                .all())
        _logger.info("get_cases_for_goal_scoped -> %s rows", len(rows))
        return rows

    def ensure_goal_state(
        self,
        casebook_id: int,
        goal_id: str,
        *,
        case_id: Optional[int] = None,
        quality: Optional[float] = None,
    ) -> CaseGoalStateORM:
        _logger.info("ensure_goal_state(casebook_id=%s, goal_id=%s, case_id=%s, quality=%s)",
                     casebook_id, goal_id, case_id, quality)
        state = self.get_goal_state(casebook_id, goal_id)
        if state is None:
            state = CaseGoalStateORM(
                casebook_id=casebook_id,
                goal_id=goal_id,
                champion_case_id=case_id,
                champion_quality=float(quality or 0.0),
            )
            self.session.add(state)
            self.session.commit()
            _logger.info("ensure_goal_state -> created")
        else:
            _logger.info("ensure_goal_state -> exists (champion_case_id=%s, quality=%s)",
                          state.champion_case_id, state.champion_quality)
        return state

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
        _logger.info(("upsert_goal_state(cb_id=%s, goal_id=%s, case_id=%s, quality=%s, "
                      "only_if_better=%s, improved=%s, delta=%s, ema_alpha=%.3f)"),
                     casebook_id, goal_id, case_id, quality, only_if_better, improved, delta, ema_alpha)
        state = self.get_goal_state(casebook_id, goal_id)
        if state is None:
            state = CaseGoalStateORM(
                casebook_id=casebook_id,
                goal_id=goal_id,
                champion_case_id=case_id,
                champion_quality=float(quality or 0.0),
            )
            self.session.add(state)
            _logger.info("upsert_goal_state -> created new state")
        else:
            if case_id is not None:
                if only_if_better and (quality is not None):
                    if float(quality) > float(state.champion_quality or 0.0):
                        _logger.info("upsert_goal_state -> updating champion (better quality %.4f > %.4f)",
                                      float(quality), float(state.champion_quality or 0.0))
                        state.champion_case_id = case_id
                        state.champion_quality = float(quality)
                    else:
                        _logger.info("upsert_goal_state -> skipping champion update (quality not better)")
                else:
                    _logger.info("upsert_goal_state -> unconditional champion update to case_id=%s", case_id)
                    state.champion_case_id = case_id
                    if quality is not None:
                        state.champion_quality = float(quality)

        if improved is not None and delta is not None:
            try:
                state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
                _logger.info("upsert_goal_state -> update_ab_stats(improved=%s, delta=%.4f)", improved, float(delta))
            except Exception:
                prev_run_ix = int(state.run_ix or 0)
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
                _logger.info("upsert_goal_state -> fallback A/B stats (run_ix %s->%s, trust=%.4f)",
                              prev_run_ix, state.run_ix, state.trust)

        self.session.commit()
        _logger.info("upsert_goal_state -> committed (champion_case_id=%s, quality=%.4f, trust=%s)",
                     state.champion_case_id, float(state.champion_quality or 0.0), getattr(state, "trust", None))
        return state


    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int:
        _logger.info("bump_run_ix(cb_id=%s, goal_id=%s)", casebook_id, goal_id)
        state = self.ensure_goal_state(casebook_id, goal_id)
        before = int(state.run_ix or 0)
        state.run_ix = before + 1
        self.session.commit()
        _logger.info("bump_run_ix -> %s -> %s", before, state.run_ix)
        return state.run_ix


    def record_ab_result(
        self,
        casebook_id: int,
        goal_id: str,
        *,
        improved: bool,
        delta: float,
        ema_alpha: float = 0.2,
    ) -> CaseGoalStateORM:
        _logger.info("record_ab_result(cb_id=%s, goal_id=%s, improved=%s, delta=%.4f, alpha=%.3f)",
                     casebook_id, goal_id, improved, delta, ema_alpha)
        state = self.ensure_goal_state(casebook_id, goal_id)
        try:
            state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
            _logger.info("record_ab_result -> update_ab_stats applied")
        except Exception:
            prev_run_ix = int(state.run_ix or 0)
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
            _logger.info("record_ab_result -> fallback stats (run_ix %s->%s, trust=%.4f)",
                          prev_run_ix, state.run_ix, state.trust)
        self.session.commit()
        _logger.info("record_ab_result -> committed (wins=%s, losses=%s, trust=%s)",
                     getattr(state, "wins", None), getattr(state, "losses", None), getattr(state, "trust", None))
        return state


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
        _logger.info(("get_recent_cases(cb_id=%s, goal_id=%s, limit=%s, only_accepted=%s, "
                      "include_champion=%s, min_quality=%s)"),
                     casebook_id, goal_id, limit, only_accepted, include_champion, min_quality)
        q = (self.session.query(CaseORM)
            .filter(CaseORM.casebook_id == casebook_id,
                    CaseORM.goal_id == goal_id)
            .order_by(getattr(CaseORM, "created_at", CaseORM.id).desc()))
        recent_all = q.limit(max(limit * 3, limit)).all()
        _logger.info("get_recent_cases -> overfetched=%s", len(recent_all))

        champion_case = None
        champion_id = None
        try:
            state = self.get_goal_state(casebook_id, goal_id)
            champion_id = getattr(state, "champion_case_id", None)
            if include_champion and champion_id:
                champion_case = next((c for c in recent_all if c.id == champion_id), None)
                if champion_case is None:
                    champion_case = self.session.get(CaseORM, champion_id)
            _logger.info("get_recent_cases -> champion_id=%s, included=%s", champion_id, include_champion)
        except Exception as e:
            _logger.info("get_recent_cases -> champion lookup failed: %s", e)

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
                if champion_case and c.id == champion_case.id:
                    continue
                result.append(c)
            if len(result) < limit:
                for c in recent_all:
                    if (champion_case and c.id == champion_case.id) or c in result:
                        continue
                    result.append(c)
                    if len(result) >= limit:
                        break
        else:
            for c in recent_all:
                if champion_case and c.id == champion_case.id:
                    continue
                result.append(c)
                if len(result) >= limit + (1 if champion_case else 0):
                    break

        out = result[:limit]
        _logger.info("get_recent_cases -> returning %s cases (champion_included=%s)", len(out), bool(champion_case))
        return out


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
        _logger.info(("get_pool_for_goal(cb_id=%s, goal_id=%s, exclude=%s, limit=%s, "
                      "include_champion=%s, only_accepted=%s, min_quality=%s)"),
                     casebook_id, goal_id, list(exclude_ids or []), limit, include_champion, only_accepted, min_quality)
        excl = {int(x) for x in (exclude_ids or []) if x is not None}

        q = (
            self.session.query(CaseORM)
            .filter(CaseORM.casebook_id != casebook_id, CaseORM.goal_id == goal_id)
        )

        if excl:
            q = q.filter(~CaseORM.id.in_(excl))

        if not include_champion:
            try:
                state = self.get_goal_state(casebook_id, goal_id)
                champ_id = getattr(state, "champion_case_id", None)
                if champ_id:
                    q = q.filter(CaseORM.id != champ_id)
            except Exception:
                pass

        order_col = getattr(CaseORM, "created_at", None)
        if order_col is not None:
            q = q.order_by(order_col.desc())
        else:
            q = q.order_by(CaseORM.id.desc())

        candidates = q.limit(max(limit * 3, limit)).all()
        _logger.info("get_pool_for_goal -> overfetched=%s", len(candidates))

        if only_accepted or (min_quality is not None):
            def is_accepted(case: CaseORM) -> bool:
                try:
                    state = self.get_goal_state(casebook_id, goal_id)
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

            before = len(candidates)
            candidates = [c for c in candidates if is_accepted(c)]
            _logger.info("get_pool_for_goal -> acceptance filtered %s -> %s", before, len(candidates))

        out = candidates[:limit]
        _logger.info("get_pool_for_goal -> returning %s candidates", len(out))
        return out




    # --- helper to guarantee a scorable_id ---
    @staticmethod
    def _make_scorable_id(s: dict, case_id: int, idx: int) -> str:
        """
        Ensure a non-empty scorable_id. Prefer passed id; otherwise derive a stable
        hash from text; if no text, use a UUID.
        """
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
            h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
            _logger.info("_make_scorable_id -> sha1[:16]=%s (text_len=%s)", h, len(text))
            return h

        rid = uuid.uuid4().hex
        _logger.info("_make_scorable_id -> uuid=%s", rid)
        return rid

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
    ):
        _logger.info(("add_case(cb_id=%s, goal_id=%s, agent=%s, prompt_len=%s, "
                      "scorables=%s, responses=%s)"),
                     casebook_id, goal_id, agent_name,
                     len(prompt_text or "") if prompt_text else 0,
                     len(scorables or []), len(response_texts or []))
        case = CaseORM(
            casebook_id=casebook_id,
            goal_id=goal_id,
            agent_name=agent_name, 
            prompt_text=prompt_text,
            meta=meta or {},
        )
        self.session.add(case)
        self.session.flush()  # need case.id for scorable ids
        _logger.info("add_case -> new CaseORM id=%s", case.id)

        added_scorables = 0

        # --- 1. Add scorables from explicit dicts ---
        if scorables:
            for idx, s in enumerate(scorables):
                safe_sid = self._make_scorable_id(s, case.id, idx)
                meta = s.get("meta") or {}
                cs = CaseScorableORM(
                    case_id=case.id,
                    scorable_id=safe_sid,
                    scorable_type=s.get("type") or s.get("target_type") or "document",
                    role=(s.get("role") or "output"),
                    rank=s.get("rank"),
                    meta=meta,
                )
                self.session.add(cs)
                added_scorables += 1
            _logger.info("add_case -> added explicit scorables=%s", added_scorables)

        # --- 2. Add scorables for assistant responses ---
        if response_texts:
            for idx, resp in enumerate(response_texts):
                if not resp or not resp.strip():
                    continue
                sc_dict = {"text": resp, "role": "assistant"}
                safe_sid = self._make_scorable_id(sc_dict, case.id, idx)
                cs = CaseScorableORM(
                    case_id=case.id,
                    scorable_id=safe_sid,
                    scorable_type="document",  # or "response"
                    role="assistant",
                    rank=idx,
                    meta={"text": resp},
                )
                self.session.add(cs)
                added_scorables += 1
            _logger.info("add_case -> added response scorables=%s (total=%s)", len(response_texts), added_scorables)

        self.session.commit()
        _logger.info("add_case -> committed case_id=%s (scorables_total=%s)", case.id, added_scorables)
        return case

    def list_casebooks(
        self,
        *,
        agent_name: Optional[str] = None,
        tag: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[CaseBookORM]:
        _logger.info("list_casebooks(agent=%s, tag=%s, run_id=%s, limit=%s)",
                      agent_name, tag, pipeline_run_id, limit)
        q = self.session.query(CaseBookORM)

        if agent_name is not None:
            q = q.filter(CaseBookORM.agent_name == agent_name)

        if tag is not None:
            q = q.filter(CaseBookORM.tag == tag)

        if pipeline_run_id is not None:
            q = q.filter(CaseBookORM.pipeline_run_id == pipeline_run_id)

        order_col = getattr(CaseBookORM, "created_at", None)
        if order_col is not None:
            q = q.order_by(order_col.desc())
        else:
            q = q.order_by(CaseBookORM.id.desc())

        rows = q.limit(limit).all()
        _logger.info("list_casebooks -> %s rows", len(rows))
        return rows

    def get_casebook(self, casebook_id: int) -> Optional[CaseBookORM]:
        _logger.info("get_casebook(id=%s)", casebook_id)
        row = self.session.get(CaseBookORM, casebook_id)
        _logger.info("get_casebook -> %s", "found" if row else "None")
        return row

    def get_casebooks(self) -> List[CaseBookORM]:
        """Return all casebooks (fixed from previous incorrect implementation)."""
        _logger.info("get_casebooks()")
        rows = self.session.query(CaseBookORM).all()
        _logger.info("get_casebooks -> %s rows", len(rows))
        return rows

    def list_cases(
        self,
        *,
        casebook_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        goal_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[CaseORM]:
        _logger.info("list_cases(cb_id=%s, agent=%s, goal_id=%s, limit=%s)",
                      casebook_id, agent_name, goal_id, limit)
        q = self.session.query(CaseORM)
        if casebook_id is not None:
            q = q.filter(CaseORM.casebook_id == casebook_id)
        if agent_name is not None:
            q = q.filter(CaseORM.agent_name == agent_name)
        if goal_id is not None:
            q = q.filter(CaseORM.goal_id == goal_id)

        order_col = getattr(CaseORM, "created_at", None)
        if order_col is not None:
            q = q.order_by(order_col.desc())
        else:
            q = q.order_by(CaseORM.id.desc())

        rows = q.limit(limit).all()
        _logger.info("list_cases -> %s rows", len(rows))
        return rows

    def get_case_by_id(self, case_id: int) -> Optional[CaseORM]:
        _logger.info("get_case_by_id(id=%s)", case_id)
        row = self.session.get(CaseORM, case_id)
        _logger.info("get_case_by_id -> %s", "found" if row else "None")
        return row


    def ensure_case(
        self,
        casebook_id: int,
        goal_text: str,
        agent_name: str,
    ) -> CaseORM:
        _logger.info("ensure_case(cb_id=%s, goal_text=%s, agent=%s)",
                     casebook_id, _trunc(goal_text), agent_name)
        q = (self.session.query(CaseORM)
             .filter_by(casebook_id=casebook_id,
                        target_type=TargetType.GOAL,
                        target_id=None))
        goal_case = q.one_or_none()

        if goal_case is None:
            goal = self.session.query(GoalORM).filter_by(goal_text=goal_text).first()
            goal_case = CaseORM(
                casebook_id=casebook_id,
                goal_id=goal.id if goal else None,
                agent_name=agent_name,
            )
            self.session.add(goal_case)
            self.session.flush()
            _logger.info("ensure_case -> created goal case id=%s (goal_id=%s)", goal_case.id, goal_case.goal_id)
        else:
            _logger.info("ensure_case -> exists goal case id=%s", goal_case.id)
        return goal_case

    def ensure_goal_state_for_case(
        self,
        casebook_id: int,
        goal_text: str,
        goal_id: str,
    ) -> CaseGoalStateORM:
        _logger.info("ensure_goal_state_for_case(cb_id=%s, goal_text=%s, goal_id=%s)",
                     casebook_id, _trunc(goal_text), goal_id)
        state = (self.session.query(CaseGoalStateORM)
                 .filter_by(casebook_id=casebook_id)
                 .one_or_none())
        if state is None:
            if not goal_id:
                goal_id = self.session.query(GoalORM).filter_by(goal_text=goal_text).first().id    
            state = CaseGoalStateORM(
                casebook_id=casebook_id,
                goal_id=goal_id,
            )
            self.session.add(state)
            self.session.commit()
            _logger.info("ensure_goal_state_for_case -> created")
        else:
            _logger.info("ensure_goal_state_for_case -> exists")
        return state

    def get_case_scorable_by_id(self, case_scorable_id: int) -> Optional[CaseScorableORM]:
        _logger.info("get_case_scorable_by_id(id=%s)", case_scorable_id)
        row = self.session.get(CaseScorableORM, case_scorable_id)
        _logger.info("get_case_scorable_by_id -> %s", "found" if row else "None")
        return row
    

    def list_scorables(self, case_id: int, role: str = None):
        _logger.info("list_scorables(case_id=%s, role=%s)", case_id, role)
        q = self.session.query(CaseScorableORM).filter_by(case_id=case_id)
        if role:
            q = q.filter(CaseScorableORM.role == role)
        rows = q.all()
        _logger.info("list_scorables -> %s rows", len(rows))
        return rows


    def add_scorable(
        self,
        case_id: int,
        pipeline_run_id: int,
        text: str,
        scorable_type: str = TargetType.DYNAMIC,
        meta: Optional[dict] = None,
        role: Optional[str] = None,
    ) -> DynamicScorableORM:
        _logger.info(("add_scorable(case_id=%s, run_id=%s, type=%s, role=%s, "
                      "text_len=%s, meta_keys=%s)"),
                     case_id, pipeline_run_id, scorable_type, role,
                     len(text or "") if text else 0,
                     list((meta or {}).keys()))
        orm = DynamicScorableORM(
            case_id=case_id,
            pipeline_run_id=pipeline_run_id,
            scorable_type=scorable_type,
            text=text,
            meta=meta or {},
            role=role,
        )
        self.session.add(orm)
        self.session.commit()
        _logger.info("add_scorable -> committed id=%s (case_id=%s)", orm.id, case_id)
        return orm
