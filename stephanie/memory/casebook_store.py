# stephanie/memory/casebook_store.py
import hashlib
# stephanie/memory/casebook_store.py
import uuid
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from stephanie.models.case_goal_state import CaseGoalStateORM
from stephanie.models.casebook import CaseBookORM, CaseORM, CaseScorableORM
from stephanie.models.goal import GoalORM
from stephanie.scoring.scorable_factory import TargetType


class CaseBookStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "casebooks"

    def get_by_name(self, name: str):
        return self.session.query(CaseBookORM).filter_by(name=name).one_or_none()

    def ensure_casebook(self, name: str, description: str = "") -> CaseBookORM:
        cb = self.get_by_name(name)
        if cb:
            return cb
        cb = CaseBookORM(name=name, description=description)
        self.session.add(cb)
        self.session.commit()
        return cb

    def create_casebook(self, name, description=""):
        cb = CaseBookORM(name=name, description=description)
        self.session.add(cb)
        self.session.commit()
        return cb

    def get_all_casebooks(self, limit: int = 100) -> List[CaseBookORM]:
        return self.session.query(CaseBookORM).limit(limit).all()

    def get_cases_for_goal(self, goal_id):
        return self.session.query(CaseORM).filter_by(goal_id=goal_id).all()

    def get_cases_for_agent(self, agent_name):
        return self.session.query(CaseORM).filter_by(agent_name=agent_name).all()

    def get_for_run_id(self, run_id: int):
        return self.session.query(CaseBookORM).filter_by(pipeline_run_id=run_id).first()

    def get_cases_for_casebook(self, casebook_id: int):
        return self.session.query(CaseORM).filter_by(casebook_id=casebook_id).all()

    # in CaseBookStore
    def get_goal_state(self, casebook_id: int, goal_id: str):
        return (self.session.query(CaseGoalStateORM)
                .filter_by(casebook_id=casebook_id, goal_id=goal_id)
                .one_or_none())

    def get_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default"):
        q = self.session.query(CaseBookORM).filter_by(
            pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag
        )
        return q.one_or_none()

    def ensure_casebook_scope(self, pipeline_run_id: int | None, agent_name: str | None, tag: str = "default"):
        cb = self.get_scope(pipeline_run_id, agent_name, tag)
        if cb: 
            return cb
        name = f"cb:{agent_name or 'all'}:{pipeline_run_id or 'all'}:{tag}"
        cb = CaseBookORM(name=name, description="Scoped casebook",
                         pipeline_run_id=pipeline_run_id, agent_name=agent_name, tag=tag)
        self.session.add(cb)
        self.session.commit()
        return cb

    # Scoped retrieval (strict)
    def get_cases_for_goal_in_casebook(self, casebook_id: int, goal_id: str):
        return (self.session.query(CaseORM)
                .filter_by(casebook_id=casebook_id, goal_id=goal_id).all())

    # Flexible retrieval (union of scopes, ordered by specificity)
    def get_cases_for_goal_scoped(self, goal_id: str,
                                  scopes: list[tuple[str|None, str|None, str]]):
        casebook_ids = []
        for p,a,t in scopes:
            cb = self.get_scope(p, a, t)
            if cb: 
                casebook_ids.append(cb.id)
        if not casebook_ids: 
            return []
        return (self.session.query(CaseORM)
                .filter(CaseORM.goal_id==goal_id, CaseORM.casebook_id.in_(casebook_ids))
                .all())

    def ensure_goal_state(
        self,
        casebook_id: int,
        goal_id: str,
        *,
        case_id: Optional[int] = None,
        quality: Optional[float] = None,
    ) -> CaseGoalStateORM:
        """
        Ensure a CaseGoalState row exists; optionally seed champion fields.
        """
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
        """
        Create or update the goal state for (casebook_id, goal_id).

        - If no row exists, it is created and optionally seeded with (case_id, quality).
        - If a row exists and case_id is provided:
            * if only_if_better=True, champion is updated only when `quality > current`.
            * otherwise champion is updated unconditionally (use when you've already decided it's improved).
        - If `improved` and `delta` are supplied, A/B counters (wins/losses, avg_delta, trust) are updated.

        Returns the up-to-date CaseGoalStateORM row.
        """
        state = self.get_goal_state(casebook_id, goal_id)
        if state is None:
            state = CaseGoalStateORM(
                casebook_id=casebook_id,
                goal_id=goal_id,
                champion_case_id=case_id,
                champion_quality=float(quality or 0.0),
            )
            self.session.add(state)
        else:
            if case_id is not None:
                if only_if_better and (quality is not None):
                    if float(quality) > float(state.champion_quality or 0.0):
                        state.champion_case_id = case_id
                        state.champion_quality = float(quality)
                else:
                    # Unconditional champion update
                    state.champion_case_id = case_id
                    if quality is not None:
                        state.champion_quality = float(quality)

        # Optional A/B bookkeeping
        if improved is not None and delta is not None:
            try:
                state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
            except Exception:
                # If the ORM doesn't have update_ab_stats yet, fall back to minimal counters
                state.run_ix = (state.run_ix or 0) + 1
                if improved:
                    state.wins = (state.wins or 0) + 1
                else:
                    state.losses = (state.losses or 0) + 1
                # simple EMA
                prev = float(state.avg_delta or 0.0)
                alpha = float(ema_alpha)
                state.avg_delta = (1.0 - alpha) * prev + alpha * float(delta)
                # clamp trust to [-1, 1]
                v = state.avg_delta
                state.trust = max(-1.0, min(1.0, v))

        self.session.commit()
        return state


    def bump_run_ix(self, casebook_id: int, goal_id: str, *, amount: int = 1) -> int:
        """
        Increment and return the run counter for A/B scheduling.
        Creates the state row if needed.
        """
        state = self.ensure_goal_state(casebook_id, goal_id)
        state.run_ix = int(state.run_ix or 0) + int(amount)
        self.session.commit()
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
        """
        Convenience wrapper to log an A/B outcome (CBR vs baseline) without changing the champion.
        """
        state = self.ensure_goal_state(casebook_id, goal_id)
        try:
            state.update_ab_stats(bool(improved), float(delta), alpha=float(ema_alpha))
        except Exception:
            state.run_ix = (state.run_ix or 0) + 1
            if improved:
                state.wins = (state.wins or 0) + 1
            else:
                state.losses = (state.losses or 0) + 1
            prev = float(state.avg_delta or 0.0)
            alpha = float(ema_alpha)
            state.avg_delta = (1.0 - alpha) * prev + alpha * float(delta)
            v = state.avg_delta
            state.trust = max(-1.0, min(1.0, v))
        self.session.commit()
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
        """
        Return recent cases for (casebook_id, goal_id), most-recent first.

        Args:
            casebook_id: scoped casebook id
            goal_id:     goal identifier
            limit:       max number of cases to return
            only_accepted:
                If True, prefer cases marked accepted.
                Acceptance is defined as:
                - champion case (if any), OR
                - case.meta['accepted'] is True, OR
                - (optional) case.meta['quality'] >= min_quality (if provided)
                If not enough accepted cases exist to reach `limit`,
                the result is filled with recent cases.
            include_champion:
                If True, the current champion (if any) is put first.
            min_quality:
                Optional float threshold to treat a case as accepted if its
                meta["quality"] >= min_quality.

        Returns:
            List[CaseORM] ordered by:
            - champion first (if include_champion and present),
            - then by created_at DESC.
        """
        # 1) Base recent query
        q = (self.session.query(CaseORM)
            .filter(CaseORM.casebook_id == casebook_id,
                    CaseORM.goal_id == goal_id)
            .order_by(getattr(CaseORM, "created_at", CaseORM.id).desc()))
        recent_all = q.limit(max(limit * 3, limit)).all()  # overfetch to allow filtering

        # 2) Champion (optional)
        champion_case = None
        champion_id = None
        try:
            state = self.get_goal_state(casebook_id, goal_id)
            champion_id = getattr(state, "champion_case_id", None)
            if include_champion and champion_id:
                champion_case = next((c for c in recent_all if c.id == champion_id), None)
                # If champion not in the overfetch window, pull it explicitly
                if champion_case is None:
                    champion_case = self.session.get(CaseORM, champion_id)
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

        # 3) Filter/sort
        accepted = [c for c in recent_all if is_accepted(c)] if only_accepted else []
        result: List[CaseORM] = []

        # champion first
        if include_champion and champion_case:
            result.append(champion_case)

        if only_accepted:
            # Add accepted (excluding champion to avoid dup)
            for c in accepted:
                if champion_case and c.id == champion_case.id:
                    continue
                result.append(c)
            # Top up with recents if needed
            if len(result) < limit:
                for c in recent_all:
                    if (champion_case and c.id == champion_case.id) or c in result:
                        continue
                    result.append(c)
                    if len(result) >= limit:
                        break
        else:
            # Not enforcing acceptance; just fill by recency after champion
            for c in recent_all:
                if champion_case and c.id == champion_case.id:
                    continue
                result.append(c)
                if len(result) >= limit + (1 if champion_case else 0):
                    break

        # 4) Cap to limit
        # If champion included, allow it to consume one of the slots.
        return result[:limit]


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
        """
        Return a pool of candidate cases for (casebook_id, goal_id), excluding the
        provided IDs and (by default) the champion. Results are ordered by recency.
        The caller can randomize/shuffle afterwards.

        Args:
            casebook_id: Scoped casebook id.
            goal_id:     Goal identifier (string).
            exclude_ids: Iterable of CaseORM ids to exclude.
            limit:       Max number of cases to return (after Python-side filtering).
            include_champion: If False (default), exclude the current champion case.
            only_accepted: If True, filter to cases marked accepted (or champion).
            min_quality: Optional quality threshold (checks case.meta["quality"]).

        Returns:
            List[CaseORM] (most-recent first), size ≤ limit.
        """
        # Sanitize excludes: remove None and cast to ints
        excl = {int(x) for x in (exclude_ids or []) if x is not None}

        # Base query for this scoped goal
        q = (
            self.session.query(CaseORM)
            .filter(CaseORM.casebook_id != casebook_id, CaseORM.goal_id == goal_id)
        )

        if excl:
            q = q.filter(~CaseORM.id.in_(excl))

        # Exclude champion unless explicitly included
        if not include_champion:
            try:
                state = self.get_goal_state(casebook_id, goal_id)
                champ_id = getattr(state, "champion_case_id", None)
                if champ_id:
                    q = q.filter(CaseORM.id != champ_id)
            except Exception:
                # goal state table not present or other issue — ignore
                pass

        # Order by recency (prefer created_at if present, else id)
        order_col = getattr(CaseORM, "created_at", None)
        if order_col is not None:
            q = q.order_by(order_col.desc())
        else:
            q = q.order_by(CaseORM.id.desc())

        # Overfetch to allow Python-side acceptance/quality filtering
        candidates = q.limit(max(limit * 3, limit)).all()

        if only_accepted or (min_quality is not None):
            def is_accepted(case: CaseORM) -> bool:
                # Champion counts as accepted even if excluded above (in case include_champion=True)
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
                # If only_accepted=True and no flags matched, it's not accepted
                return not only_accepted

            candidates = [c for c in candidates if is_accepted(c)]

        # Cap to limit; caller will shuffle or do diversity selection
        return candidates[:limit]




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

        # try to get text from meta or direct field
        text = ""
        meta = s.get("meta") or {}
        if isinstance(meta, dict):
            text = meta.get("text", "") or meta.get("content", "")
        text = text or s.get("text", "") or s.get("content", "")

        if text:
            base = f"{case_id}:{idx}:{text}"
            return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

        # absolute fallback
        return uuid.uuid4().hex

    def add_case(
        self,
        casebook_id: int,
        goal_id: str,
        agent_name: str,
        goal_text: Optional[str] = None,
        mars_summary: Optional[str] = None,
        scores: Optional[dict] = None,
        metadata: Optional[dict] = None,
        scorables: Optional[list[dict]] = None,
        prompt_text: Optional[str] = None,
        response_texts: Optional[list[str]] = None,
    ):
        """
        Add a case to a casebook with prompt + responses.

        Args:
            casebook_id: ID of the parent casebook
            goal_id:     Related goal id
            goal_text:   Related goal text
            agent_name:  Which agent created the case
            mars_summary: Optional reasoning summary
            scores:     Optional dict of scores
            metadata:   Optional dict of metadata
            scorables:  Explicit scorable dicts (legacy path)
            prompt_text: Text of the prompt (user input)
            response_texts: List of assistant responses to attach as scorables

        Returns:
            CaseORM instance
        """
        case = CaseORM(
            casebook_id=casebook_id,
            goal_id=goal_id,
            agent_name=agent_name, 
            prompt_text=prompt_text,
        )
        self.session.add(case)
        self.session.flush()  # need case.id for scorable ids

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

        self.session.commit()
        return case

    def list_casebooks(
        self,
        *,
        agent_name: Optional[str] = None,
        tag: Optional[str] = None,
        pipeline_run_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[CaseBookORM]:
        """
        Filterable list of casebooks, newest first.
        Any filter left as None is ignored.
        All right"""
        q = self.session.query(CaseBookORM)

        if agent_name is not None:
            q = q.filter(CaseBookORM.agent_name == agent_name)

        if tag is not None:
            q = q.filter(CaseBookORM.tag == tag)

        if pipeline_run_id is not None:
            q = q.filter(CaseBookORM.pipeline_run_id == pipeline_run_id)

        # Prefer newest first
        order_col = getattr(CaseBookORM, "created_at", None)
        if order_col is not None:
            q = q.order_by(order_col.desc())
        else:
            q = q.order_by(CaseBookORM.id.desc())

        return q.limit(limit).all()

    def get_casebook(self, casebook_id: int) -> Optional[CaseBookORM]:
        """Load a casebook by its primary key."""
        return self.session.get(CaseBookORM, casebook_id)


    def get_casebooks(self) -> Optional[CaseBookORM]:
        """Load a casebook by its primary key."""
        return self.session.get(CaseBookORM).all()

    def list_cases(
        self,
        *,
        casebook_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        goal_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[CaseORM]:
        """List recent cases with optional filters, newest first."""
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

        return q.limit(limit).all()

    def get_case_by_id(self, case_id: int) -> Optional[CaseORM]:
        """Load a single case with its relationships."""
        return self.session.get(CaseORM, case_id)


    def ensure_case(
        self,
        casebook_id: int,
        goal_text: str,
        agent_name: str,
    ) -> CaseORM:
        """
        Ensure there is a parent 'goal' case row for this casebook.
        Creates one if not present.

        Args:
            casebook_id: CaseBookORM.id
            goal_text:   text of the goal
            agent_name:  name of the agent creating the case

        Returns:
            CaseORM instance
        """
        q = (self.session.query(CaseORM)
             .filter_by(casebook_id=casebook_id,
                        target_type=TargetType.GOAL,
                        target_id=None))
        goal_case = q.one_or_none()


        if goal_case is None:
            goal = self.session.query(GoalORM).filter_by(goal_text=goal_text).first()
            goal_case = CaseORM(
                casebook_id=casebook_id,
                goal_id=goal.id,
                agent_name=agent_name,
            )
            self.session.add(goal_case)
            self.session.flush()
        return goal_case

    def ensure_goal_state_for_case(
        self,
        casebook_id: int,
        goal_text: str,
        goal_id: str,
    ) -> CaseGoalStateORM:
        """
        Ensure CaseGoalState row exists for the given casebook/goal.
        """
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
        return state

    def get_case_scorable_by_id(self, case_scorable_id: int) -> Optional[CaseScorableORM]:
        """Load a single CaseScorable by its primary key."""
        return self.session.get(CaseScorableORM, case_scorable_id)