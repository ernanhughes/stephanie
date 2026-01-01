from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass
class TickResult:
    tick: int
    draft: str
    score: float
    rationale: str
    improved: bool

class TraceBOOSTCaseEvolver:
    """
    SRDP controller for Paper→Blog sections.
    Memory = CaseBook (cases + attrs).
    Trace = PlanTrace (ticks as steps).
    """

    def __init__(self, *, casebook_store, plan_trace_store, judge_agent, llm):
        self.casebooks = casebook_store
        self.traces = plan_trace_store
        self.judge = judge_agent
        self.llm = llm

    async def evolve_section(
        self,
        *,
        pipeline_run_id: int,
        agent_name: str,
        paper_id: str,
        paper_title: str,
        section_title: str,
        goal_text: str,
        initial_prompt: str,
        max_ticks: int,
        quality_threshold: float,
        tick_min_delta: float,
        casebook_tag: str = "paper_blog:srdp",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        # 1) CaseBook scope + Case
        cb = self.casebooks.ensure_casebook_scope(
            pipeline_run_id=pipeline_run_id,
            agent_name=agent_name,
            tag=casebook_tag,
        )  # :contentReference[oaicite:4]{index=4}

        case = self.casebooks.ensure_case(
            casebook_id=cb.id,
            goal_text=goal_text,
            agent_name=agent_name,
        )  # :contentReference[oaicite:5]{index=5}

        # Static attrs for retrieval/analysis
        self.casebooks.set_case_attr(case.id, "paper_id", value_text=paper_id)
        self.casebooks.set_case_attr(case.id, "paper_title", value_text=paper_title)
        self.casebooks.set_case_attr(case.id, "section_title", value_text=section_title)

        # 2) Tick loop
        best_draft = ""
        best_score = float("-inf")
        best_rationale = ""
        last_best = float("-inf")

        for tick in range(1, max_ticks + 1):
            if tick == 1:
                draft = await self.llm.generate(initial_prompt)
            else:
                refine_prompt = self._build_refine_prompt(
                    best_draft=best_draft,
                    feedback=best_rationale,
                    section_title=section_title,
                )
                draft = await self.llm.generate(refine_prompt)

            score, rationale = await self.judge.score_section(
                paper_id=paper_id,
                paper_title=paper_title,
                section_title=section_title,
                draft=draft,
            )

            improved = score > best_score
            if improved:
                best_draft, best_score, best_rationale = draft, score, rationale

            # 3) Write tick result into CaseBook attrs
            self.casebooks.set_case_attr(case.id, f"tick.{tick}.score", value_num=float(score))
            self.casebooks.set_case_attr(case.id, f"tick.{tick}.draft", value_text=draft)
            self.casebooks.set_case_attr(case.id, f"tick.{tick}.rationale", value_text=rationale)
            self.casebooks.set_case_attr(case.id, f"tick.{tick}.improved", value_bool=bool(improved))

            # 4) Halting
            if best_score >= quality_threshold:
                break
            if tick > 1 and (best_score - last_best) < tick_min_delta:
                break
            last_best = best_score

        # Optional: update goal state champion if better (CaseGoalStateORM)
        self.casebooks.upsert_goal_state(
            casebook_id=cb.id,
            goal_id=case.goal_id,
            case_id=case.id,
            quality=float(best_score),
            only_if_better=True,
        )

        return best_draft, best_score

    def _build_refine_prompt(self, *, best_draft: str, feedback: str, section_title: str) -> str:
        return f"""
You are refining a technical blog section: {section_title}

## Current best draft
{best_draft}

## Judge feedback (what to fix)
{feedback}

Rewrite the section to address the feedback while preserving correct technical content.
Return markdown only.
""".strip()
