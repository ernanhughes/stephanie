# stephanie/memory/reasoning_sample_store.py
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from sqlalchemy import desc, select
from sqlalchemy.orm import aliased

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm.chat import ChatMessageORM, ChatTurnORM
from stephanie.orm.evaluation import EvaluationORM
from stephanie.orm.goal import GoalORM
from stephanie.orm.reasoning_sample import ReasoningSampleORM
from stephanie.orm.score import ScoreORM
from stephanie.scoring.scorable import ScorableType


def _iter_dimension_scores(
    sample: ReasoningSampleORM,
) -> Iterable[Tuple[str, float]]:
    """
    Yield (dimension, score) pairs from ReasoningSampleORM.scores JSON.
    Each item in `scores` should look like: {"dimension": "...", "score": <number>, ...}.
    Skips non-numeric scores or missing dimensions.
    """
    items = getattr(sample, "scores", None) or []
    for it in items:
        dim = it.get("dimension")
        val = it.get("score")
        if not dim:
            continue
        try:
            yield dim, float(val)
        except Exception:
            continue


class ReasoningSampleStore(BaseSQLAlchemyStore):
    """
    Read-only store for reasoning_samples_view.
    Used by data loaders (TinyRecursion, SICQL, etc.)
    to fetch structured reasoning examples.
    """

    orm_model = ReasoningSampleORM
    default_order_by = ReasoningSampleORM.created_at.desc()

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "reasoning_samples"

    def get_all(self, limit: int = 1000) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )

        return self._run(op)

    def get_by_target_type(
        self, target_type: str, limit: int = 100
    ) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.scorable_type == target_type)
                .order_by(ReasoningSampleORM.created_at.desc())
                .limit(limit)
                .all()
            )

        return self._run(op)

    def get_by_goal(
        self, goal_text: str, limit: int = 50
    ) -> List[ReasoningSampleORM]:
        def op(s):
            return (
                s.query(ReasoningSampleORM)
                .filter(ReasoningSampleORM.goal_text.ilike(f"%{goal_text}%"))
                .limit(limit)
                .all()
            )

        return self._run(op)

    def get_training_pairs_by_dimension(
        self,
        goal: Optional[str] = None,
        limit: int = 100,
        dim: Optional[List[str]] = None,
        target_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        """
        Build preference pairs (TOP vs BOTTOM) per (dimension × scorable_id),
        mirroring the old SQL:
          - Partition by (dimension, doc_id) and pick rank_high=1 and rank_low=1
          - Keep non-empty text
          - Return pairs grouped by dimension

        Args:
          goal: optional substring filter on goal_text (ILIKE)
          limit: total pairs to return (best-effort across all dimensions)
          dim: optional list of dimension names to include
          target_type: optional filter on scorable_type (e.g., "document")

        Returns:
          { dimension: [ {title, output_a, output_b, value_a, value_b}, ... ] }
        """
        # 1) Fetch a generous slice of recent samples
        fetch_n = max(limit * 20, 500)

        if goal and target_type:
            samples = self.get_by_target_type(target_type, limit=fetch_n)
            g = goal.lower()
            samples = [
                s
                for s in samples
                if (getattr(s, "goal_text", "") or "").lower().find(g) >= 0
            ][:fetch_n]
        elif target_type:
            samples = self.get_by_target_type(target_type, limit=fetch_n)
        elif goal:
            samples = self.get_by_goal(goal, limit=fetch_n)
        else:
            samples = self.get_all(limit=fetch_n)

        # 2) Bucket by (dimension, scorable_id) collecting candidate rows with scores
        # We store: for each key, a list of tuples (score, text, title, sample_ref)
        by_key: Dict[
            Tuple[str, str], List[Tuple[float, str, str, ReasoningSampleORM]]
        ] = defaultdict(list)

        for s in samples:
            text = (getattr(s, "scorable_text", None) or "").strip()
            title = getattr(s, "goal_text", None) or ""
            scorable_id = getattr(s, "scorable_id", None)
            if not scorable_id:
                continue  # must have an id to emulate doc_id partitioning

            for dname, score in _iter_dimension_scores(s):
                if dim and dname not in dim:
                    continue
                # Note: allow empty text for bottom side? Original SQL required non-empty for 'top' only.
                by_key[(dname, scorable_id)].append((score, text, title, s))

        # 3) For each (dimension, scorable_id), pick highest and lowest scored sample
        results_by_dimension: Dict[str, List[Dict[str, object]]] = defaultdict(
            list
        )
        remaining = limit

        for (dname, _doc_id), rows in by_key.items():
            if not rows:
                continue

            # rank_high = 1 (max score), rank_low = 1 (min score)
            rows_sorted = sorted(rows, key=lambda r: r[0])
            low = rows_sorted[0]
            high = rows_sorted[-1]

            score_hi, text_hi, title_hi, _ = high
            score_lo, text_lo, title_lo, _ = low

            # Emulate original filter: top.text must be non-empty
            if not text_hi:
                continue

            # Build pair (top vs bottom). The original code allowed bottom to be empty;
            # we keep it as-is to mirror behavior.
            results_by_dimension[dname].append(
                {
                    "title": title_hi or title_lo or dname,
                    "output_a": text_hi,
                    "output_b": text_lo,
                    "value_a": float(score_hi),
                    "value_b": float(score_lo),
                }
            )
            remaining -= 1
            if remaining <= 0:
                break

        return dict(results_by_dimension)

    def get_eval_pairs_by_dimension(
        self,
        target_type: str,                  # e.g., 'conversation_turn'
        dimension: str,                    # e.g., 'reasoning'
        *,
        limit: int = 1000,
        max_recent: int = 50_000,          # look over this many recent evals
        # >>> ABSOLUTE selection (robust to skew) <<<
        top_k: int = 250,                  # take this many highest-scored items
        bottom_k: int = 250,               # and this many lowest-scored items
        # pairing mode
        pairing: Literal["zip", "cartesian"] = "cartesian",
        # hygiene
        require_nonempty_top: bool = True,
        allow_empty_bottom: bool = True,
        dedup_equal_texts: bool = True,
        # randomness
        shuffle_source_sets: bool = True,  # shuffle highs/lows before pairing (tie-breaking)
        final_shuffle: bool = True,        # shuffle final pairs list
        rng_seed: Optional[int] = 42,      # deterministic if set
    ) -> Dict[str, List[Dict[str, object]]]:
        """
        Build preference pairs for a single dimension using ABSOLUTE top_k/bottom_k
        (much more stable under highly skewed score distributions).
        Returns: { <dimension>: [ {title, output_a, output_b, value_a, value_b}, ... ] }
        """

        if rng_seed is not None:
            random.seed(int(rng_seed))

        def op(session):
            # 1) Fetch recent evaluations for the target scorable_type
            eval_rows = session.execute(
                select(
                    EvaluationORM.id,
                    EvaluationORM.scorable_type,
                    EvaluationORM.scorable_id,
                    EvaluationORM.created_at,
                    EvaluationORM.goal_id,
                )
                .where(EvaluationORM.scorable_type == target_type)
                .order_by(desc(EvaluationORM.created_at))
                .limit(max_recent)
            ).all()
            if not eval_rows:
                return {dimension: []}

            eval_ids = [r.id for r in eval_rows]
            scorable_ids = {r.scorable_id for r in eval_rows}
            goal_ids = {r.goal_id for r in eval_rows if r.goal_id is not None}

            # 2) Scores for this dimension
            score_rows = session.execute(
                select(ScoreORM.evaluation_id, ScoreORM.score)
                .where(ScoreORM.evaluation_id.in_(eval_ids))
                .where(ScoreORM.dimension == dimension)
            ).all()
            if not score_rows:
                return {dimension: []}

            # 3) Batch resolve text for conversation_turn; titles from goals
            text_by_scorable: Dict[str, str] = {}
            if target_type == ScorableType.CONVERSATION_TURN:
                for mid, t in session.query(ChatMessageORM.id, ChatMessageORM.text).filter(
                    ChatMessageORM.id.in_(scorable_ids)
                ):
                    text_by_scorable[str(mid)] = (t or "").strip()

            title_by_goal: Dict[int, str] = {}
            if goal_ids:
                for gid, gt in session.query(GoalORM.id, GoalORM.goal_text).filter(
                    GoalORM.id.in_(goal_ids)
                ):
                    title_by_goal[int(gid)] = (gt or "").strip()

            meta_by_eval: Dict[int, Tuple[str, str, str]] = {}  # eval_id -> (title, text, scorable_id)
            for r in eval_rows:
                title = title_by_goal.get(r.goal_id, target_type)
                text = text_by_scorable.get(str(r.scorable_id), "") if r.scorable_type == ScorableType.CONVERSATION_TURN else ""
                meta_by_eval[int(r.id)] = (title, text, str(r.scorable_id))

            # 4) Assemble (score, text, title)
            items: List[Tuple[float, str, str]] = []
            for eid, sval in score_rows:
                try:
                    s = float(sval) if sval is not None else None
                except Exception:
                    s = None
                if s is None:
                    continue
                title, text, _sid = meta_by_eval.get(int(eid), (target_type, "", ""))
                items.append((s, text, title))

            if not items:
                return {dimension: []}

            # 5) Sort ascending by score; pick absolute bottom_k + top_k with tie-aware sampling
            items.sort(key=lambda t: t[0])
            n = len(items)

            # Bottom slice (lowest scores)
            bottom_slice = items[:max(1, min(bottom_k, n))]
            # Top slice (highest scores)
            top_slice = items[-max(1, min(top_k, n)):] if n > 0 else []

            # If there are giant ties at the edges, randomize selection within those slices
            if shuffle_source_sets:
                random.shuffle(bottom_slice)
                random.shuffle(top_slice)

            lows = bottom_slice[: min(bottom_k, len(bottom_slice))]
            highs = list(reversed(top_slice))[: min(top_k, len(top_slice))]  # highest first

            # 6) Build pairs
            pairs: List[Dict[str, object]] = []

            def push_pair(hi, lo):
                s_hi, t_hi, title_hi = hi
                s_lo, t_lo, title_lo = lo
                if require_nonempty_top and not t_hi:
                    return
                if not allow_empty_bottom and not t_lo:
                    return
                if dedup_equal_texts and t_hi == t_lo:
                    return
                pairs.append({
                    "title": title_hi or title_lo or dimension,
                    "output_a": t_hi,          # top
                    "output_b": t_lo,          # bottom
                    "value_a": float(s_hi),
                    "value_b": float(s_lo),
                })

            if pairing == "zip":
                for hi, lo in zip(highs, lows):
                    if len(pairs) >= limit:
                        break
                    push_pair(hi, lo)
            elif pairing == "cartesian":
                for hi in highs:
                    if len(pairs) >= limit:
                        break
                    for lo in lows:
                        if len(pairs) >= limit:
                            break
                        push_pair(hi, lo)
            else:
                raise ValueError(f"Unknown pairing mode: {pairing}")

            if final_shuffle and len(pairs) > 1:
                random.shuffle(pairs)

            print(
                f"[pairs.abs] dim={dimension} n_items={n} "
                f"top_k={len(highs)}/{top_k} bottom_k={len(lows)}/{bottom_k} "
                f"mode={pairing} produced={len(pairs)} (limit={limit})"
            )
            # Optional quick distribution peek (first/last few scores)
            if highs:
                print(f"[pairs.abs] top head scores: {[round(x[0],2) for x in highs[:6]]}")
            if lows:
                print(f"[pairs.abs] low head scores: {[round(x[0],2) for x in lows[:6]]}")

            return {dimension: pairs}

        return self._run(op)

    def get_text(self, scorable_type: str, scorable_id: int) -> str:
        """
        Return the assistant's text for a given scorable.
        - For conversation turns, scorable_id == ChatTurn.id
          -> resolve ChatTurn.assistant_message_id -> ChatMessage.text
        - Otherwise, try treating scorable_id as a ChatMessage id.
        """

        def op(s):
            try:
                turn_id = int(scorable_id)
            except Exception:
                turn_id = None

            # Conversation turn: resolve via ChatTurn → assistant_message_id → ChatMessage
            if (
                scorable_type
                in (ScorableType.CONVERSATION_TURN, "conversation_turn")
                and turn_id is not None
            ):
                A = aliased(ChatMessageORM)
                row = (
                    s.query(A.text)
                    .select_from(ChatTurnORM)
                    .join(A, ChatTurnORM.assistant_message_id == A.id)
                    .filter(ChatTurnORM.id == turn_id)
                    .first()
                )
                return (
                    row[0].strip() if row and row[0] else ""
                )  # assistant_text

            # Fallback: treat id as a ChatMessage id (covers other types)
            msg = (
                s.query(ChatMessageORM.text)
                .filter(ChatMessageORM.id == scorable_id)
                .first()
            )
            return msg[0].strip() if msg and msg[0] else ""

        return self._run(op)
    


    def get_eval_pairs_global(
        self,
        target_type: str,           # 'conversation_turn'
        dimension: str,             # e.g. 'reasoning'
        limit: int = 10_000,
        max_recent: int = 50_000,
        top_frac: float = 0.10,
        bottom_frac: float = 0.10,
        require_nonempty_top: bool = True,
    ) -> Dict[str, List[dict]]:
        """
        Build MANY pairs by taking global top decile vs bottom decile across all
        scorable_ids for (target_type, dimension). No per-doc cap.
        """
        # 1) Fetch recent evaluations
        def op(s):
            eval_rows = (
                s.execute(
                    select(
                        EvaluationORM.id,
                        EvaluationORM.scorable_type,
                        EvaluationORM.scorable_id,
                        EvaluationORM.created_at,
                        EvaluationORM.goal_id,
                    )
                    .where(EvaluationORM.scorable_type == target_type)
                    .order_by(desc(EvaluationORM.created_at))
                    .limit(max_recent)
                )
                .all()
            )
            if not eval_rows:
                return {dimension: []}
            eval_map = {r.id: r for r in eval_rows}
            eval_ids = [r.id for r in eval_rows]

            # 2) Scores for that dimension
            score_rows = (
                s.execute(
                    select(ScoreORM.evaluation_id, ScoreORM.score)
                    .where(ScoreORM.evaluation_id.in_(eval_ids))
                    .where(ScoreORM.dimension == dimension)
                )
                .all()
            )
            if not score_rows:
                return {dimension: []}

            # 3) Resolve (score, text, title)
            items = []
            for eid, sval in score_rows:
                try:
                    sc = float(sval) if sval is not None else None
                except Exception:
                    sc = None
                if sc is None:
                    continue
                meta = eval_map.get(eid)
                if not meta:
                    continue

                # Assistant text for conversation_turn
                text = ""
                if meta.scorable_type == "conversation_turn":
                    msg = s.query(ChatMessageORM).filter(ChatMessageORM.id == meta.scorable_id).first()
                    text = (getattr(msg, "text", "") or "").strip()

                title = ""
                if meta.goal_id:
                    g = s.query(GoalORM).filter(GoalORM.id == meta.goal_id).first()
                    title = (getattr(g, "goal_text", "") or "").strip()

                items.append((sc, text, title))

            if not items:
                return {dimension: []}

            # 4) Global highs/lows
            items.sort(key=lambda t: t[0])
            n = len(items)
            k_top = max(1, int(n * top_frac))
            k_bot = max(1, int(n * bottom_frac))
            lows  = items[:k_bot]
            highs = items[-k_top:][::-1]

            # 5) Pair greedily (top i vs bottom i)
            pairs = []
            for (s_hi, t_hi, title_hi), (s_lo, t_lo, title_lo) in zip(highs, lows):
                if require_nonempty_top and not t_hi:
                    continue
                if t_hi == t_lo:
                    continue
                pairs.append({
                    "title": title_hi or title_lo or dimension,
                    "output_a": t_hi,
                    "output_b": t_lo,
                    "value_a": float(s_hi),
                    "value_b": float(s_lo),
                })
                if len(pairs) >= limit:
                    break

            return {dimension: pairs}
        return self._run(op)


    def debug_eval_pair_yield(
        self, target_type: str, dimension: str, max_recent: int = 50000
    ):
        """
        Print why we're not getting thousands of pairs.
        """

        def op(s):
            # 1) Fetch recent evaluations
            eval_rows = s.execute(
                select(
                    EvaluationORM.id,
                    EvaluationORM.scorable_type,
                    EvaluationORM.scorable_id,
                    EvaluationORM.created_at,
                    EvaluationORM.goal_id,
                )
                .where(EvaluationORM.scorable_type == target_type)
                .order_by(desc(EvaluationORM.created_at))
                .limit(max_recent)
            ).all()
            eval_ids = [r.id for r in eval_rows]
            print(
                f"[debug] evaluations fetched: {len(eval_ids)} (target_type={target_type}, max_recent={max_recent})"
            )

            # 2) Fetch scores for the requested dimension
            score_rows = s.execute(
                select(
                    ScoreORM.evaluation_id, ScoreORM.dimension, ScoreORM.score
                )
                .where(ScoreORM.evaluation_id.in_(eval_ids))
                .where(ScoreORM.dimension == dimension)
            ).all()
            print(f"[debug] scores in dim='{dimension}': {len(score_rows)}")

            # 3) Map eval_id -> text resolvable?
            #    (Fast approximate: just count how many will have *non-empty* assistant text)
            U = aliased(ChatMessageORM)
            A = aliased(ChatMessageORM)
            # build a set of scorable ids -> text presence
            scorable_ids = set(str(r.scorable_id) for r in eval_rows)
            # sample-resolve a subset to estimate empties (or resolve all if ok)
            has_text = 0
            total_checked = 0
            missing_text_ids = 0
            # OPTIONAL: resolve all; if too slow, limit or implement a vectorized join
            for r in eval_rows:
                if r.scorable_type == "conversation_turn":
                    msg = (
                        s.query(ChatMessageORM)
                        .filter(ChatMessageORM.id == r.scorable_id)
                        .first()
                    )
                    txt = (getattr(msg, "text", "") or "").strip()
                    has_text += 1 if txt else 0
                    total_checked += 1
            print(
                f"[debug] scorable text present (approx): {has_text}/{total_checked}"
            )

            # 4) Group by scorable_id to see the hard cap
            by_scorable = defaultdict(list)
            for eid, dname, sval in score_rows:
                by_scorable[eid].append((dname, sval))
            print(
                f"[debug] score rows grouped by evaluation_id: {len(by_scorable)} (cap if 1 pair per doc)"
            )

            # 5) Simple upper bounds
            print("[debug] theoretical caps:")
            print(f"       - num evals: {len(eval_rows)}")
            print(f"       - num scores in dim: {len(score_rows)}")
            print(
                f"       - num unique eval_ids with that dim: {len(set([eid for (eid, _, _) in score_rows]))}"
            )
            print(
                "If you pair *per scorable_id*, your cap is roughly the number of distinct scorable_ids with that dim & text."
            )

        return self._run(op)
