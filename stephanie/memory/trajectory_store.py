from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.fragment import FragmentORM
from stephanie.models.transition import TransitionORM

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _stable_json(data: Dict[str, Any]) -> str:
    # stable string for dedup fingerprints
    return json.dumps(data or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

class AgentTrajectoryStore(BaseSQLAlchemyStore):
    """
    Unified store for agent trajectories (transitions) and extracted fragments.
    - Emits structured (state, action, reward) transitions
    - Records fragments produced/selected during runs
    - Provides convenient queries for SIS and trainers
    """

    orm_model = TransitionORM
    default_order_by = TransitionORM.created_at.desc()


    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "trajectory"

    # ---------- Transition helpers ----------

    def _fp_transition(
        self, *, run_id: str, step_idx: int, agent: str, state: Dict[str, Any], action: Dict[str, Any]
    ) -> str:
        base = f"{run_id}||{step_idx}||{agent}||{_stable_json(state)}||{_stable_json(action)}"
        return _sha1(base)

    def emit_transition(
        self,
        *,
        run_id: str,
        step_idx: int,
        agent: str,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward_air: float = 0.0,
        rewards_vec: Optional[Dict[str, float]] = None,
        dedup: bool = False,
    ) -> int:
        """
        Add a single transition row. Set dedup=True to avoid duplicates for the same (run_id, step_idx, payload).
        Returns TransitionORM.id
        """
        def op(s):
            fp = None
            if dedup:
                fp = self._fp_transition(run_id=run_id, step_idx=step_idx, agent=agent, state=state, action=action)
                # soft dedup: search by (run_id, step_idx) first; fall back to fp scan if needed
                existing = (
                    s.query(TransitionORM)
                    .filter(TransitionORM.run_id == run_id, TransitionORM.step_idx == step_idx)
                    .first()
                )
                if existing:
                    return existing.id

            row = TransitionORM(
                run_id=run_id,
                step_idx=step_idx,
                agent=agent,
                state=state or {},
                action=action or {},
                reward_air=reward_air,
                rewards_vec=rewards_vec or {},
            )
            # NB: if you choose to add a 'fp' column later, set it here
            # row.fp = fp
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("TransitionEmitted", {"id": row.id, "run_id": run_id, "step_idx": step_idx})
            return row.id

        return self._run(op)

    def bulk_emit_transitions(
        self, items: Iterable[Dict[str, Any]], dedup: bool = False
    ) -> List[int]:
        """
        items: dicts with keys matching emit_transition (run_id, step_idx, agent, state, action, reward_air, rewards_vec)
        Returns list of inserted ids.
        """
        ids: List[int] = []

        def op(s):
            nonlocal ids
            rows = []
            seen_keys: set[Tuple[str, int]] = set()
            for it in items:
                run_id = it["run_id"]
                step_idx = int(it["step_idx"])
                agent = it.get("agent", "UnknownAgent")
                state = it.get("state") or {}
                action = it.get("action") or {}
                reward_air = float(it.get("reward_air", 0.0) or 0.0)
                rewards_vec = it.get("rewards_vec") or {}

                if dedup:
                    key = (run_id, step_idx)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                rows.append(
                    TransitionORM(
                        run_id=run_id,
                        step_idx=step_idx,
                        agent=agent,
                        state=state,
                        action=action,
                        reward_air=reward_air,
                        rewards_vec=rewards_vec,
                    )
                )
            if not rows:
                ids = []
                return []
            s.add_all(rows)
            s.flush()
            ids = [r.id for r in rows]
            if self.logger:
                self.logger.log("TransitionsBulkEmitted", {"count": len(ids)})
            return ids

        return self._run(op)

    # ---------- Fragment helpers ----------

    def add_fragment(
        self,
        *,
        case_id: int,
        source_type: str,
        text: str,
        section: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        scores: Optional[Dict[str, float]] = None,
        uncertainty: Optional[float] = None,
        dedup_on_text: bool = True,
    ) -> int:
        """
        Insert a fragment row. For dedup, we soft-check same (case_id, section, text[:256]) to avoid exact repeats.
        Returns FragmentORM.id
        """
        def op(s):
            if dedup_on_text and text:
                probe = (
                    s.query(FragmentORM.id)
                    .filter(
                        FragmentORM.case_id == case_id,
                        FragmentORM.section == section,
                        FragmentORM.text == text,
                    )
                    .first()
                )
                if probe:
                    return probe[0]

            row = FragmentORM(
                case_id=case_id,
                source_type=source_type,
                section=section,
                text=text,
                attrs=attrs or {},
                scores=scores or {},
                uncertainty=uncertainty,
            )
            s.add(row)
            s.flush()
            if self.logger:
                self.logger.log("FragmentInserted", {"id": row.id, "case_id": case_id, "section": section})
            return row.id

        return self._run(op)

    def bulk_add_fragments(self, items: Iterable[Dict[str, Any]], dedup_on_text: bool = True) -> List[int]:
        """
        items: dicts matching add_fragment kwargs.
        Returns list of inserted IDs (skips exact dupes if enabled).
        """
        ids: List[int] = []

        def op(s):
            nonlocal ids
            rows: List[FragmentORM] = []
            for it in items:
                case_id = int(it["case_id"])
                source_type = it["source_type"]
                text = it["text"]
                section = it.get("section")
                attrs = it.get("attrs") or {}
                scores = it.get("scores") or {}
                uncertainty = it.get("uncertainty")

                if dedup_on_text and text:
                    probe = (
                        s.query(FragmentORM.id)
                        .filter(
                            FragmentORM.case_id == case_id,
                            FragmentORM.section == section,
                            FragmentORM.text == text,
                        )
                        .first()
                    )
                    if probe:
                        continue

                rows.append(
                    FragmentORM(
                        case_id=case_id,
                        source_type=source_type,
                        section=section,
                        text=text,
                        attrs=attrs,
                        scores=scores,
                        uncertainty=uncertainty,
                    )
                )
            if not rows:
                ids = []
                return []
            s.add_all(rows)
            s.flush()
            ids = [r.id for r in rows]
            if self.logger:
                self.logger.log("FragmentsBulkInserted", {"count": len(ids)})
            return ids

        return self._run(op)

    # ---------- Credit + updates ----------

    def finalize_run_credit(self, run_id: str, alpha: float = 0.4) -> float:
        """
        Assign final credited rewards to transitions of a run using your credit rule.
        Returns the computed terminal scalar for the run.
        """
        from sqlalchemy import asc
        def op(s):
            rows = (
                s.query(TransitionORM)
                .filter(TransitionORM.run_id == run_id)
                .order_by(asc(TransitionORM.step_idx))
                .all()
            )
            if not rows:
                return 0.0
            rvec = rows[-1].rewards_vec or {}
            final = (
                0.5 * float(rvec.get("mrq_correct", 0.0))
                + 0.3 * float(rvec.get("hrm_epistemic", 0.0))
                + 0.2 * float(rvec.get("sicql_adv_norm", 0.0))
            )
            n = len(rows)
            for r in rows:
                share = (1.0 - alpha) * (final / n)
                air = float(r.reward_air or 0.0)
                r.reward_final = air * alpha + share
            if self.logger:
                self.logger.log("RunCreditAssigned", {"run_id": run_id, "final": final, "steps": n})
            return final

        return self._run(op)

    # ---------- Queries (SIS/dashboard friendly) ----------

    def get_transitions_by_run(self, run_id: str) -> List[TransitionORM]:
        def op(s):
            q = s.query(TransitionORM).filter(TransitionORM.run_id == run_id)
            q = q.order_by(TransitionORM.step_idx.asc())
            return q.all()
        return self._run(op)

    def get_fragments_by_case(self, case_id: int, limit: int = 200) -> List[FragmentORM]:
        def op(s):
            q = s.query(FragmentORM).filter(FragmentORM.case_id == case_id)
            q = q.order_by(FragmentORM.created_at.desc()).limit(limit)
            return q.all()
        return self._run(op)

    def recent_runs(self, limit: int = 50) -> List[str]:
        """Distinct run_ids seen recently in transitions."""
        from sqlalchemy import desc
        def op(s):
            q = s.query(TransitionORM.run_id).order_by(desc(TransitionORM.created_at)).distinct()
            return [r[0] for r in q.limit(limit).all()]
        return self._run(op)

    def strategy_attribution(self, run_id: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Mean credited reward by strategy label found in action/state.
        """
        def extract_strategy(a: Dict[str, Any], st: Dict[str, Any]) -> Optional[str]:
            return (a or {}).get("strategy") or (st or {}).get("strategy")

        def op(s):
            q = s.query(TransitionORM)
            if run_id:
                q = q.filter(TransitionORM.run_id == run_id)
            q = q.order_by(TransitionORM.created_at.desc()).limit(limit)
            rows = q.all()
            bucket: Dict[str, List[float]] = {}
            for r in rows:
                strat = extract_strategy(r.action, r.state)
                if not strat:
                    continue
                val = float(r.reward_final or 0.0)
                bucket.setdefault(strat, []).append(val)
            out = []
            for k, arr in bucket.items():
                if not arr:
                    continue
                avg = sum(arr) / len(arr)
                out.append({"strategy": k, "mean_credit": avg, "n": len(arr)})
            out.sort(key=lambda x: x["mean_credit"], reverse=True)
            return out

        return self._run(op)

    # ---------- Maintenance ----------

    def delete_run(self, run_id: str) -> int:
        """Hard-delete all transitions for a run. Returns count."""
        def op(s):
            rows = s.query(TransitionORM).filter(TransitionORM.run_id == run_id).all()
            n = len(rows)
            for r in rows:
                s.delete(r)
            if self.logger:
                self.logger.log("RunDeleted", {"run_id": run_id, "count": n})
            return n
        return self._run(op)
