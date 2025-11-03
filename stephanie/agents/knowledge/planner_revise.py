# stephanie/agents/quality/plan_revise_agent.py
from __future__ import annotations

import re
from typing import Any, List

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PLAN_TRACE_ID
from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.scorable import Scorable


class PlannerReviseAgent(BaseAgent):
    """
    Post-plan 'Revise' pass:
      - Critique candidate plan
      - Optionally propose a revised plan
      - Produce revise_score (0–1), plus optional sub-dimensions
      - Persist all signals; adopt revised plan if above threshold
    """

    def __init__(self, cfg, memory, container, logger, full_cfg=None):
        super().__init__(cfg, memory, container, logger)
        self.min_revise_score = cfg.get("min_revise_score", 0.65)
        self.enable_edit = cfg.get("enable_edit", True)
        self.revise_dimensions = cfg.get(
            "revise_dimensions",
            ["coherence", "feasibility", "completeness"]  # optional sub-scores
        )
        # Optional: custom prompt template via PromptLoader; fallback text if not set
        self.prompt_key = cfg.get("revise_prompt_template", "revise_plan_prompt.txt")
        self.judge_on_low = cfg.get("judge_on_low", True)          # enable pairwise judge when score is low
        self.judge_scorer = cfg.get("judge_scorer", "reward")      # 'reward' or 'llm_judge' (registered in ScoringService)
        self.judge_dimensions = cfg.get("judge_dimensions", None)  # optional dimension list for judge
        self.judge_margin = cfg.get("judge_margin", 0.05)          # tie-break margin
        self.on_fail = cfg.get("on_fail", "replan")                # 'replan' | 'retry' | 'none' | 'ask_human'
        self.max_revise_attempts = cfg.get("max_revise_attempts", 1)


    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "")

        attempt = int(context.get("revise_attempts", 0))
        context["revise_attempts"] = attempt + 1

        # Input: pull plan from either configured input_key or the conventional 'plan'
        candidate_plan: List[str] = context.get(self.input_key)

        examples = context.get("examples", [])  # from planner_reuse
        if not candidate_plan:
            self.logger.log("PlanReviseNoPlan", {"detail": "No plan found in context"})
            return context

        self.report({
            "event": "revise_start",
            "step": "PlanRevise",
            "goal": goal_text[:300],
            "num_examples": len(examples),
        })

        # Build prompt context and call LLM
        merged = {
            **context,
            "goal_text": goal_text,
            "candidate_plan": candidate_plan,
            "examples": examples,
            "enable_edit": self.enable_edit,
        }

        try:
            prompt = self.prompt_loader.load_prompt(self.cfg, context=merged)
            response = self.call_llm(prompt, context=merged)
            parsed = self._extract_revise_from_response(response)
        except Exception as e:
            self.logger.log("PlanReviseLLMError", {"error": str(e)})
            # non-fatal; keep original plan
            parsed = {"rationale": "", "score": None, "revised_plan": []}

        # Decide adoption
        revise_score = parsed.get("score")
        revised_plan = parsed.get("revised_plan", []) if self.enable_edit else []
        adopted = False
        final_plan = candidate_plan
        judge_res = None

        # Optional judge when score is low but we HAVE a revision to compare
        if (self.judge_on_low 
            and self.enable_edit 
            and revised_plan 
            and (not isinstance(revise_score, (int, float)) or revise_score < self.min_revise_score)):

            try:
                a_text = "\n".join(candidate_plan)
                b_text = "\n".join(revised_plan)
                judge_res = self.scoring.compare_pair(
                    scorer_name=self.judge_scorer,
                    context=context,  # carries goal + prefs, etc.
                    a=Scorable(id="candidate", text=a_text, target_type="plan"),
                    b=Scorable(id="revised",  text=b_text, target_type="plan"),
                    dimensions=self.judge_dimensions,
                    margin=self.judge_margin,
                )
                self.logger.log("PlanReviseJudgeResult", judge_res)
            except Exception as e:
                self.logger.log("PlanReviseJudgeError", {"error": str(e)})
                judge_res = None

        adopt_reason = None
        if self.enable_edit and revised_plan:
            if isinstance(revise_score, (int, float)) and revise_score >= self.min_revise_score:
                final_plan = revised_plan
                adopted = True
                adopt_reason = "revise_score"
            elif judge_res and judge_res.get("winner") == "b":
                # Revised beats original by the pairwise judge
                final_plan = revised_plan
                adopted = True
                adopt_reason = f"judge:{self.judge_scorer}"

        # If we didn’t adopt, set recall/next-action flags for the pipeline
        if not adopted:
            context["revise_outcome"] = {
                "adopted": False,
                "reason": ("low_score" if not isinstance(revise_score, (int, float)) or revise_score < self.min_revise_score else "no_revision"),
                "action": self.on_fail,
                "attempt": context["revise_attempts"],
            }
            if self.on_fail == "replan":
                context["recall_planner"] = True
            elif self.on_fail == "ask_human":
                context["needs_review"] = True

        # Persist evaluation + scores
        try:
            self._persist_revise_scores(context, candidate_plan, parsed, adopted)
        except Exception as e:
            self.logger.log("PlanRevisePersistError", {"error": str(e)})

        # Update context
        context[self.output_key] = final_plan
        context[f"{self.output_key}_meta"] = {
            "source": "revise_agent",
            "adopted": adopted,
            "adopt_reason": adopt_reason,
            "revise_score": revise_score,
            "issues_count": len(parsed.get("issues", [])),
        }

        # Reporting
        self.report({
            "event": "revise_done",
            "step": "PlanRevise",
            "adopted": adopted,
            "revise_score": revise_score,
            "orig_len": len(candidate_plan),
            "revised_len": len(revised_plan) if revised_plan else 0,
        })

        return context

    def _format_examples_for_prompt(self, examples: List[dict], max_n: int = 3) -> str:
        if not examples:
            return "None"
        out = []
        for e in examples[:max_n]:
            out.append(
                f"- trace_id={e.get('trace_id')}, hrm={e.get('hrm')}, rank={e.get('rank_score')}, knn={e.get('knn_score')}\n"
                f"  goal: {str(e.get('goal') or '')[:160]}"
            )
        return "\n".join(out)

    def _extract_revise_from_response(self, response: str) -> dict:
        """
        Parse a revise response.
        Supports:
          1) JSON with keys: rationale, score, revised_plan (list[str]), issues (list[str]), subscores{...}
          2) Markdown with headers (## rationale, ## revised plan, ## score: x)
        """
        res = {"rationale": "", "score": None, "revised_plan": [], "issues": [], "subscores": {}}
        txt = (response or "").strip()

        # 1) Try JSON first (fenced or plain)
        try:
            t = txt
            if t.startswith("```"):
                import re as _re
                t = _re.sub(r"^```[^\n]*\n", "", t)
                t = _re.sub(r"\n?```$", "", t).strip()
            if t.startswith("{"):
                import json
                data = json.loads(t)
                res["rationale"] = str(data.get("rationale", ""))[:4000]
                if isinstance(data.get("revised_plan"), list):
                    res["revised_plan"] = [str(s).strip() for s in data["revised_plan"] if str(s).strip()]
                if isinstance(data.get("issues"), list):
                    res["issues"] = [str(s).strip() for s in data["issues"] if str(s).strip()]
                if isinstance(data.get("subscores"), dict):
                    res["subscores"] = {k: float(v) for k, v in data["subscores"].items() if self._is_num(v)}
                sc = data.get("score")
                if self._is_num(sc):
                    res["score"] = float(sc)
                return res
        except Exception:
            pass

        # 2) Markdown fallback
        m_rat = re.search(r"##\s*rationale:\s*(.*)", txt, re.IGNORECASE)
        if m_rat:
            res["rationale"] = m_rat.group(1).strip()

        # issues: bullet lines after ## issues:
        m_issues = re.split(r"##\s*issues:\s*", txt, flags=re.IGNORECASE)
        if len(m_issues) > 1:
            block = m_issues[1]
            bullets = re.findall(r"^\s*[-*]\s*(.+)$", block, flags=re.MULTILINE)
            res["issues"] = [b.strip() for b in bullets if b.strip()]

        # revised plan: numbered steps
        m_plan = re.split(r"##\s*revised\s*plan:\s*", txt, flags=re.IGNORECASE)
        if len(m_plan) > 1:
            block = m_plan[1]
            steps = re.findall(r"^\s*\d+\.\s*(.+)$", block, flags=re.MULTILINE)
            res["revised_plan"] = [s.strip() for s in steps if s.strip()]

        # score:
        m_score = re.search(r"##\s*score:\s*([0-9]+(\.[0-9]+)?)", txt, re.IGNORECASE)
        if m_score:
            try:
                res["score"] = float(m_score.group(1))
            except Exception:
                res["score"] = None

        return res

    def _is_num(self, x: Any) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    def _to_float(self, x, default=None):
        try: 
            if x is None:
                return default
            # allow numeric strings like "0.82" and ints
            return float(x)
        except (TypeError, ValueError):
            return default


    def _persist_revise_scores(self, context: dict, orig_plan: list[str], parsed: dict, adopted: bool) -> None:
        """
        Persist revise_score (+ optional subscores) directly on the *current* plan_trace.
        Assumes PLAN_TRACE_ID is present in context (as you confirmed).
        """
        trace_id = str(context.get(PLAN_TRACE_ID))
        if not trace_id:
            # Just in case, but per your setup this should never hit
            self.logger.log("PlanReviseMissingTraceId", {"note": "PLAN_TRACE_ID absent; skipping persist"})
            return

        # Compact scorable text: original plan (first ~50 steps)
        scorable_text = "\n".join(orig_plan[:50])

        scorable = Scorable(
            id=trace_id,
            text=scorable_text,
            target_type="plan_trace",
        )

        # Main revise score (0–1)
        results = {
            "revise_score": ScoreResult(
                dimension="revise_score",
                score=float(parsed.get("score") or 0.0),
                weight=1.0,
                source="plan_revise",
                rationale=(parsed.get("rationale") or "")[:1000],
                attributes={
                    "adopted": bool(adopted),
                    "issues_count": len(parsed.get("issues") or []),
                    "orig_len": len(orig_plan),
                    "revised_len": len(parsed.get("revised_plan") or []),
                    "plan_trace_id": trace_id,
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "goal_id": (context.get("goal") or {}).get("id"),
                },
            )
        }

        # Optional sub-dimensions (if you included them in the prompt/response JSON)
        subs = parsed.get("subscores") or {}
        if not isinstance(subs, dict):
            subs = {}

        # Prefer configured keys, but also accept any other numeric subscores returned
        wanted_dims = list(self.revise_dimensions or [])
        extra_dims = [d for d in subs.keys() if d not in wanted_dims]
        all_dims = wanted_dims + extra_dims

        for dim in all_dims:
            raw = subs.get(dim, None)
            v = self._to_float(raw, default=None)
            if v is None:
                # Log once per missing/invalid subscore, but don’t crash
                self.logger.log("PlanReviseSubscoreSkipped", {
                    "dimension": dim,
                    "raw_value": raw,
                    "reason": "missing_or_non_numeric"
                })
                continue

            results[dim] = ScoreResult(
                dimension=dim,
                score=v,
                weight=1.0,
                source="plan_revise",
                rationale=f"Subscore: {dim}",
            )

        bundle = ScoreBundle(results=results)

        # This will:
        # - create an EvaluationORM tied to (scorable_id=trace_id, scorable_type="plan_trace")
        # - attach ScoreORM rows for each dimension
        # - auto-link to RuleApplication(s) via pipeline_run_id / goal_id (your save_bundle handles this)
        self.memory.evaluations.save_bundle(
            bundle=bundle,
            scorable=scorable,
            context=context,          # carries pipeline_run_id + goal
            cfg=self.cfg,
            agent_name=self.name,
            source="plan_revise",
            embedding_type=self.memory.embedding.name,
        )

        self.logger.log("PlanRevisePersisted", {
            "plan_trace_id": trace_id,
            "revise_score": results["revise_score"].score,
            "subscores": {k: v.score for k, v in results.items() if k != "revise_score"}
        })

