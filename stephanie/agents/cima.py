from __future__ import annotations
import dataclasses, logging, math
from typing import Any, Dict, List, Optional

# ── deps you already have ─────────────────────────────────────────────────────
from stephanie.agents.care_executor_agent import CAREExecutorAgent, CARETrace, LLMAdapter
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_knowledge import KnowledgeScorer
from stephanie.data.score_bundle import ScoreBundle

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StepScore:
    idx: int
    score: float
    human_prob: float
    ai_prob: float
    alpha: float
    head_gap: float
    human_fraction: float
    ai_fraction: float
    violations: List[str]
    retrievals: List[str]
    text: str


@dataclasses.dataclass
class KnowledgeCAREOutput:
    goal: str
    memo: str
    steps: List[StepScore]
    trace: CARETrace
    summary: Dict[str, Any]           # rollups for dashboards


class KnowledgeCAREAgent:
    """
    Unified agent:
      1) Runs CARE (evidence-tagged, abstention-aware reasoning) over provided context.
      2) Scores each step with KnowledgeScorer (two-head, calibrated blend).
      3) Ranks/filters steps, emits a clean memo with inline footnotes.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        knowledge_scorer: KnowledgeScorer,
        *, no I'm fine It's going to work
        min_keep: int = 3,
        max_keep: int = 8,
        score_floor: float = 0.55,         # drop weak steps
        prefer_human_fraction: float = 0.55,  # route/flag steps dominated by AI
        strict_verbatim_check: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.log = logger or logging.getLogger(__name__)
        self.care = CAREExecutorAgent(
            llm=llm,
            strict_verbatim_check=strict_verbatim_check,
            min_steps=min_keep,
            max_steps=max_keep,
        )
        self.ks = knowledge_scorer
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.score_floor = float(score_floor)
        self.prefer_human_fraction = float(prefer_human_fraction)

    # ── public API ────────────────────────────────────────────────────────────

    async def run(self, goal: str, context: str) -> KnowledgeCAREOutput:
        trace = await self.care.run(goal, context)
        step_scores = self._score_steps(goal, trace)
        kept = self._select_steps(step_scores)

        memo = self._render_memo(goal, kept)

        rollup = self._rollup(step_scores, kept, trace)
        return KnowledgeCAREOutput(goal=goal, memo=memo, steps=kept, trace=trace, summary=rollup)

    def run_sync(self, goal: str, context: str) -> KnowledgeCAREOutput:
        trace = self.care.run_sync(goal, context)
        step_scores = self._score_steps(goal, trace)
        kept = self._select_steps(step_scores)
        memo = self._render_memo(goal, kept)
        rollup = self._rollup(step_scores, kept, trace)
        return KnowledgeCAREOutput(goal=goal, memo=memo, steps=kept, trace=trace, summary=rollup)

    # ── internals ────────────────────────────────────────────────────────────

    def _score_steps(self, goal: str, trace: CARETrace) -> List[StepScore]:
        out: List[StepScore] = []
        for s in trace.steps:
            # turn CARE step into a Scorable
            sc = Scorable(
                id=str(s.idx),
                text=s.answer.strip(),
                target_type="conversation_turn",  # matches your scorer
                meta={
                    "order_index": s.idx,
                    "conv_length": max(len(trace.steps), 1),
                    "has_retrieval": bool(s.retrievals),
                    "retrieval_fidelity": 1.0 if s.retrievals else 0.0,
                    "text_len_norm": min(1.0, len(s.answer) / 2000.0),
                },
            )
            ctx = {"goal": {"goal_text": goal}}
            bundle: ScoreBundle = self.ks.score(ctx, scorable=sc)  # returns knowledge ScoreResult
            res = bundle.results["knowledge"]
            attrs = res.attributes or {}

            out.append(
                StepScore(
                    idx=s.idx,
                    score=float(res.score),
                    human_prob=float(attrs.get("human_prob", math.nan)),
                    ai_prob=float(attrs.get("ai_prob", math.nan)),
                    alpha=float(attrs.get("alpha_human_weight", math.nan)),
                    head_gap=float(attrs.get("head_gap", math.nan)),
                    human_fraction=float(attrs.get("human_fraction", math.nan)),
                    ai_fraction=float(attrs.get("ai_fraction", math.nan)),
                    violations=list(s.violations),
                    retrievals=list(s.retrievals),
                    text=s.answer.strip(),
                )
            )
        # stable sort: score desc, then lower violations count, then original idx
        return sorted(out, key=lambda t: (-t.score, len(t.violations), t.idx))

    def _select_steps(self, steps: List[StepScore]) -> List[StepScore]:
        # keep high-score steps; always keep at least min_keep; cap at max_keep
        kept = [s for s in steps if s.score >= self.score_floor]
        if len(kept) < self.min_keep:
            kept = steps[: self.min_keep]
        return kept[: self.max_keep]

    def _render_memo(self, goal: str, steps: List[StepScore]) -> str:
        """
        Simple memo with inline numeric footnotes per step.
        """
        lines = [f"# Memo: {goal}\n"]
        foot_i = 1
        for s in steps:
            # Construct footnotes from retrievals (truncate long)
            notes = []
            for r in s.retrievals:
                snippet = (r if len(r) <= 220 else (r[:200] + "… " + r[-20:])).replace("\n", " ")
                notes.append(f"[{foot_i}] {snippet}")
                foot_i += 1
            claim = s.text
            lines.append(f"{s.idx}. {claim}")
            if notes:
                for n in notes:
                    lines.append(f"    {n}")
            if s.violations:
                lines.append(f"    ⚠︎ CARE violations: {', '.join(s.violations)}")
            # attribution hint
            lines.append(
                f"    (knowledge={s.score:.2f} • human={s.human_fraction:.2f} • ai={s.ai_fraction:.2f})"
            )
            lines.append("")
        return "\n".join(lines).strip()

    def _rollup(self, all_steps: List[StepScore], kept: List[StepScore], trace: CARETrace) -> Dict[str, Any]:
        def avg(xs: List[float]) -> float:
            return float(sum(xs) / max(len(xs), 1))

        return {
            "steps_total": len(all_steps),
            "steps_kept": len(kept),
            "avg_score_all": avg([s.score for s in all_steps]),
            "avg_score_kept": avg([s.score for s in kept]),
            "avg_human_fraction_kept": avg([s.human_fraction for s in kept if not math.isnan(s.human_fraction)]),
            "avg_ai_fraction_kept": avg([s.ai_fraction for s in kept if not math.isnan(s.ai_fraction)]),
            "avg_head_gap_kept": avg([s.head_gap for s in kept if not math.isnan(s.head_gap)]),
            "any_contract_violations": bool(trace.contract_violations),
            "contract_violations": trace.contract_violations,
        }
