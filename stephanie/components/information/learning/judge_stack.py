# stephanie/components/information/learning/judge_stack.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

# Types for the tools we expect to receive.
PairRMCallable = Callable[[str, List[str]], Awaitable[Dict[str, float]]]
RubricCallable = Callable[[str, str], Awaitable[Dict[str, Any]]]
FactualityCallable = Callable[[str, str], Awaitable[Dict[str, Any]]]


@dataclass
class JudgeStageResult:
    """
    One stage of the judge stack for a single candidate.
    """
    stage_name: str                     # "pairrm", "rubric", "factuality"
    scores: Dict[str, float]            # e.g. {"pairrm": 0.82} or {"faithfulness": 91, ...}
    gates: Dict[str, bool] = field(default_factory=dict)       # e.g. {"factuality_ok": True}
    rationale: Dict[str, str] = field(default_factory=dict)    # per-dimension rationales
    meta: Dict[str, Any] = field(default_factory=dict)         # raw model ids, latency, etc.


@dataclass
class CandidateJudgeRecord:
    """
    Aggregate record for a single blog candidate across all stages.
    """
    candidate_id: str
    text: str
    stages: List[JudgeStageResult] = field(default_factory=list)
    final_score: Optional[float] = None
    disqualified: bool = False

    def get_stage(self, name: str) -> Optional[JudgeStageResult]:
        for s in self.stages:
            if s.stage_name == name:
                return s
        return None


class JudgeStack:
    """
    Orchestrates:
      1) PairRM stage (fast pairwise pruning)
      2) Rubric stage (multi-dimension scoring)
      3) Factuality stage (source-grounded gate)
    and then aggregates to a single final_score per candidate.

    This is deliberately generic: it only knows about
    *interfaces* for tools, not concrete models.
    """

    def __init__(self, cfg: Dict[str, Any], tools: Dict[str, Any], logger):
        """
        cfg is usually cfg['judge_stack'] from YAML.

        Expected shape (all optional, sensible defaults):

          judge_stack:
            pairrm:
              enabled: true
              top_k: 3
            rubric:
              enabled: true
            factuality:
              enabled: true
              min_score: 70.0         # factuality score gate
              gate_dimension: factuality
              gate_mode: disqualify   # or "penalize"
              gate_penalty: 40.0
            weights:
              pairrm: 0.2
              rubric: 0.6
              factuality: 0.2
        """
        self.cfg = cfg or {}
        self.tools = tools or {}
        self.log = logger

    async def evaluate(
        self,
        paper_pack: str,
        candidates: List[Dict[str, Any]],
    ) -> List[CandidateJudgeRecord]:
        """
        candidates: list of dicts with at least:
          { "id": "...", "text": "..." } OR { "name": "...", "text": "..." }

        Returns CandidateJudgeRecord list with final_score + stage breakdowns.
        """
        records: List[CandidateJudgeRecord] = []
        for i, c in enumerate(candidates):
            cid = (
                c.get("candidate_id")
                or c.get("id")
                or c.get("name")
                or f"cand_{i}"
            )
            txt = c.get("text") or ""
            records.append(CandidateJudgeRecord(candidate_id=str(cid), text=txt))

        # 1) optional PairRM pruning
        if self.cfg.get("pairrm", {}).get("enabled", False) and len(records) > 1:
            await self._run_pairrm_stage(paper_pack, records)

        # 2) rubric judge on remaining
        if self.cfg.get("rubric", {}).get("enabled", True):
            await self._run_rubric_stage(paper_pack, records)

        # 3) factuality gate
        if self.cfg.get("factuality", {}).get("enabled", False):
            await self._run_factuality_stage(paper_pack, records)

        # 4) aggregate into final_score / winner
        self._aggregate_and_pick_winner(records)

        return records

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_pairrm_stage(
        self,
        paper_pack: str,
        records: List[CandidateJudgeRecord],
    ) -> None:
        """
        PairRM interface:

          pairrm_tool: PairRMCallable
              async def pairrm_tool(paper_pack: str,
                                    candidates: List[str]) -> Dict[str, float]
          where the returned dict maps candidate_id or index -> score.
        """
        pairrm_tool: Optional[PairRMCallable] = self.tools.get("pairrm")
        if not pairrm_tool:
            self.log.warning("JudgeStack: pairrm.enabled=True but no 'pairrm' tool provided")
            return

        texts = [r.text for r in records]
        raw_scores = await pairrm_tool(paper_pack=paper_pack, candidates=texts)

        for idx, rec in enumerate(records):
            # allow key by id, string(id), idx, string(idx)
            key_candidates = [
                rec.candidate_id,
                str(rec.candidate_id),
                idx,
                str(idx),
            ]
            s: Optional[float] = None
            for k in key_candidates:
                if k in raw_scores:
                    s = raw_scores[k]
                    break
            if s is None:
                continue

            stage = JudgeStageResult(
                stage_name="pairrm",
                scores={"pairrm": float(s)},
                gates={},
                rationale={},
                meta={},
            )
            rec.stages.append(stage)

    async def _run_rubric_stage(
        self,
        paper_pack: str,
        records: List[CandidateJudgeRecord],
    ) -> None:
        """
        Rubric interface:

          rubric_tool: RubricCallable
              async def rubric_tool(paper_pack: str, text: str) -> Dict[str, Any]

        Expected return shape (loosely aligned with PaperBlogJudgeAgent):

          {
            "dimensions": {
              "faithfulness": {"score": 91.0, "rationale": "..."},
              "coverage":     {"score": 80.0, "rationale": "..."},
              ...
            },
            "raw": ...
          }
        """
        rubric_tool: Optional[RubricCallable] = self.tools.get("rubric")
        if not rubric_tool:
            self.log.warning("JudgeStack: rubric.enabled=True but no 'rubric' tool provided")
            return

        for rec in records:
            if not rec.text.strip():
                continue

            out = await rubric_tool(paper_pack=paper_pack, text=rec.text)
            dims = (out or {}).get("dimensions") or {}

            scores: Dict[str, float] = {}
            rationale: Dict[str, str] = {}
            for name, d in dims.items():
                if not isinstance(d, dict):
                    continue
                scores[str(name)] = float(d.get("score", 0.0))
                rationale[str(name)] = str(d.get("rationale", "")).strip()

            stage = JudgeStageResult(
                stage_name="rubric",
                scores=scores,
                gates={},
                rationale=rationale,
                meta={"raw": out},
            )
            rec.stages.append(stage)

    async def _run_factuality_stage(
        self,
        paper_pack: str,
        records: List[CandidateJudgeRecord],
    ) -> None:
        """
        Factuality interface:

          factuality_tool: FactualityCallable
              async def factuality_tool(paper_pack: str, text: str) -> Dict[str, Any]

        Expected return shape (flexible, but we look for):

          {
            "score": 0-100,               # higher = more factual / consistent
            "ok": True/False,             # gate result
            "details": "...",             # optional text
            ...
          }
        """
        factuality_tool: Optional[FactualityCallable] = self.tools.get("factuality")
        if not factuality_tool:
            self.log.warning("JudgeStack: factuality.enabled=True but no 'factuality' tool provided")
            return

        for rec in records:
            if not rec.text.strip():
                continue

            out = await factuality_tool(paper_pack=paper_pack, text=rec.text)  # type: ignore[arg-type]
            score = float((out or {}).get("score", 0.0))
            ok = bool((out or {}).get("ok", True))
            details = str((out or {}).get("details", "")).strip()

            stage = JudgeStageResult(
                stage_name="factuality",
                scores={"factuality": score},
                gates={"factuality_ok": ok},
                rationale={"factuality": details},
                meta={"raw": out},
            )
            rec.stages.append(stage)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_and_pick_winner(
        self,
        records: List[CandidateJudgeRecord],
    ) -> None:
        """
        Compute final_score for each candidate using stack weights + gates.

        Strategy (simple, but works):
          - normalize PairRM scores to [0, 100] if present
          - average rubric dimensions to a single rubric_score
          - use factuality score directly
          - combine with weights from cfg['weights']
          - apply factuality gate if configured
        """
        if not records:
            return

        weights_cfg = self.cfg.get("weights", {}) or {}
        w_pairrm = float(weights_cfg.get("pairrm", 0.2))
        w_rubric = float(weights_cfg.get("rubric", 0.6))
        w_fact = float(weights_cfg.get("factuality", 0.2))
        w_sum = max(w_pairrm + w_rubric + w_fact, 1e-6)

        fact_cfg = self.cfg.get("factuality", {}) or {}
        fact_gate_dim = str(fact_cfg.get("gate_dimension", "factuality"))
        fact_min = float(fact_cfg.get("min_score", 0.0))
        gate_mode = str(fact_cfg.get("gate_mode", "disqualify"))
        gate_penalty = float(fact_cfg.get("gate_penalty", 40.0))

        # gather PairRM scores to normalize
        pairrm_vals: List[float] = []
        for rec in records:
            st = rec.get_stage("pairrm")
            if st and "pairrm" in st.scores:
                pairrm_vals.append(float(st.scores["pairrm"]))
        pr_min = min(pairrm_vals) if pairrm_vals else 0.0
        pr_max = max(pairrm_vals) if pairrm_vals else 1.0
        pr_range = max(pr_max - pr_min, 1e-6)

        best_score = -1e9
        best: Optional[CandidateJudgeRecord] = None

        for rec in records:
            # --- PairRM contribution
            st_pair = rec.get_stage("pairrm")
            raw_pair = float(st_pair.scores.get("pairrm", 0.0)) if st_pair else 0.0
            pair_norm = 0.0
            if pairrm_vals:
                pair_norm = (raw_pair - pr_min) / pr_range * 100.0

            # --- Rubric contribution (average across dimensions)
            st_rub = rec.get_stage("rubric")
            if st_rub and st_rub.scores:
                rubric_score = sum(st_rub.scores.values()) / max(len(st_rub.scores), 1)
            else:
                rubric_score = 0.0

            # --- Factuality contribution
            st_fact = rec.get_stage("factuality")
            fact_score = float(st_fact.scores.get("factuality", 100.0)) if st_fact else 100.0
            fact_ok = bool((st_fact.gates.get("factuality_ok", True)) if st_fact else True)

            # --- base weighted score
            weighted = (
                w_pairrm * pair_norm +
                w_rubric * rubric_score +
                w_fact * fact_score
            ) / w_sum

            # --- gate application
            failed_gate = (fact_score < fact_min) or (not fact_ok)
            if failed_gate and gate_mode == "disqualify":
                final = -1.0
                rec.disqualified = True
            elif failed_gate and gate_mode == "penalize":
                final = weighted - gate_penalty
                rec.disqualified = False
            else:
                final = weighted
                rec.disqualified = False

            rec.final_score = final

            if final > best_score:
                best_score = final
                best = rec

        # We don't store a winner field here; caller can pick max(final_score).
        if best is None:
            return

        self.log.info(
            "JudgeStack winner: %s (score=%.2f)",
            best.candidate_id,
            best.final_score or 0.0,
        )
