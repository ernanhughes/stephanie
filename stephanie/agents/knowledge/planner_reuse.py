# stephanie/agents/planning/planner_reuse.py
import re
from tqdm import tqdm


from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.scorable_ranker import ScorableRanker
from stephanie.scoring.scorable import Scorable
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PLAN_TRACE_ID
from stephanie.scoring.hrm_scorer import HRMScorer
from stephanie.scoring.scorable_factory import TargetType


class PlannerReuseAgent(BaseAgent):
    def __init__(self, cfg, memory, logger, full_cfg):
        super().__init__(cfg, memory, logger)
        self.ranker = ScorableRanker(cfg, memory, logger)
        self.top_k = cfg.get("top_k", 100)
        self.min_hrm = cfg.get("min_hrm", 0.6)
        self.use_db_knn = cfg.get("use_db_knn", True)
        self.rerank_with_scorable_ranker = cfg.get("rerank_with_scorable_ranker", False)
        self.dimensions = cfg.get("dimensions", ["alignment"])
        self.scorer = HRMScorer(full_cfg["scorer"]["hrm"], memory=self.memory, logger=self.logger)

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "")

        # --- 1. Retrieve candidate past traces ---
        candidates = []

        filtered_traces = []

        # get ids of matching traces
        related_scorables = self.memory.embedding.search_related_scorables(goal_text, TargetType.PLAN_TRACE, self.top_k)
        for scorable in related_scorables:
            pt = self.memory.plan_traces.get_by_trace_id(scorable.get("id"))
            if not pt:
                continue
            to_score = ScorableFactory.from_plan_trace(pt, goal_text=goal_text)
            bundle = self.scorer.score(context, to_score, self.dimensions)
            score = bundle.aggregate()
            self.logger.log("PlannerReuseHRMScore", {"score": score, "trace_id": pt.trace_id})
            if bundle.aggregate() >= self.min_hrm:
                filtered_traces.append(pt)
            else: 
                self.logger.log("PlannerReuseFilteredTrace", {"trace_id": pt.trace_id})

        # filter list based upon hrm score

        pbar = tqdm(filtered_traces, desc="Embedding Candidates", disable=not self.cfg.get("progress", True))
        for idx, pt in enumerate(pbar, start=1):
            related_scorables = ScorableFactory.from_plan_trace(pt, goal_text=goal_text)
            embed_id = self.memory.scorable_embeddings.get_or_create(related_scorables)
            self.logger.log("PlannerReuseCandidate", {
                "scorable_id": related_scorables.id,
                "embedding_id": embed_id
            })

            # Build hybrid text: goal + final output (+ step outputs if wanted)
            trace_text_parts = [pt.goal.goal_text if pt.goal else "", pt.final_output_text]
            if self.cfg.get("include_steps", False):
                trace_text_parts.extend([s.output_text for s in pt.execution_steps])

            candidate_text = "\n".join([t for t in trace_text_parts if t])
            candidates.append(
                Scorable(id=pt.trace_id, text=candidate_text, target_type="plan_trace")
            )
            pbar.set_postfix({"candidates": f"{idx}/{len(filtered_traces)}"})

        if not candidates:
            self.logger.log("PlannerReuseNoCandidates", {"goal_text": goal_text})
            self.report({
                "event": "planner_reuse",
                "step": "PlannerReuse",
                "details": "No past traces available for reuse",
                "goal_text": goal_text,
            })
            return context

        # --- 2. Rank candidates ---
        query_scorable = Scorable(id="current_goal", text=goal_text, target_type="goal")
        ranked = self.ranker.rank(query=query_scorable, candidates=candidates, context=context)

        # --- 3. Report (convert ORM → dict for SYS)
        ranked_dicts = [ev.to_dict() for ev in ranked]
        self.report(self.ranker.to_report_dict(query_scorable, ranked_dicts))

        top = ranked[:self.top_k]  # take top k

        # --- 2. Gather top examples ---
        examples = []
        pbar = tqdm(zip(top, candidates), total=len(top), desc="Collecting Top Examples", disable=not self.cfg.get("progress", True))
        for bundle, cand in pbar:
            # Match bundles back to the original candidate
            # (since rank() processed them in the same order)
            pt = self.memory.plan_traces.get_by_trace_id(cand.id)
            goal_text = self.memory.plan_traces.get_goal_text(cand.id)
            if pt:
                examples.append({
                    "trace_id": pt.trace_id,
                    "goal": goal_text,
                    "plan": pt.plan_signature,
                    "rank_score": bundle.results["rank_score"].score,
                })
                pbar.set_postfix({"examples": f"{len(examples)}/{len(top)}"})

        self.report({
            "event": "planner_reuse",
            "step": "PlannerReuse",
            "details": f"Retrieved {len(examples)} past traces for reuse",
            "examples": examples,
            "goal_text": goal_text,
        })

        # --- 3. Adaptation step (LLM) ---
        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        merged = {"examples": examples, **context}
        response = self.call_llm(prompt, context=merged)
        parsed = self._extract_plan_from_response(response)

        # --- 4. Update context ---
        context[self.output_key] = parsed["plan"]
        context[f"{self.output_key}_meta"] = {
            "rationale": parsed["rationale"],
            "confidence_score": parsed["score"],
        }

        self.logger.log("PlannerReuseGenerated", {
            "goal_text": goal_text,
            "plan": parsed["plan"],
            "confidence_score": parsed["score"],
        })

        self.report({
            "event": "planner_reuse",
            "step": "PlannerReuse",
            "details": "New plan adapted from past traces",
            "goal_text": goal_text,
            "plan": parsed["plan"],
            "confidence_score": parsed["score"],
        })

        # --- 5. Record reuse links in DB ---
        try:
            # We need the trace_id of the *new* plan being generated.
            # Assume Supervisor/Monitor has already created a PlanTrace in memory.
            new_trace_id = context.get(PLAN_TRACE_ID)

            if new_trace_id:
                for ex in examples:
                    parent_trace_id = ex.get("trace_id") or None
                    if parent_trace_id:
                        self.memory.plan_traces.add_reuse_link(
                            parent_trace_id=parent_trace_id,
                            child_trace_id=new_trace_id
                        )
                self.logger.log("PlannerReuseLinksCreated", {
                    "child": new_trace_id,
                    "parents": [ex.get("trace_id") for ex in examples if ex.get("trace_id")]
                })
        except Exception as e:
            self.logger.log("PlannerReuseLinkError", {"error": str(e)})

        self.logger.log("PlannerReuseGenerated", {
            "goal_text": goal_text,
            "plan": parsed["plan"],
            "confidence_score": parsed["score"],
        })

        self.report({
            "event": "planner_reuse",
            "step": "PlannerReuse",
            "details": "New plan adapted from past traces",
            "goal_text": goal_text,
            "plan": parsed["plan"],
            "confidence_score": parsed["score"],
        })

        return context

    def _extract_plan_from_response(self, response: str) -> dict:
        """
        Parse rationale, score, and plan steps from LLM response.
        Returns: {"rationale": str, "score": float, "plan": list[str]}
        """
        result = {"rationale": "", "score": None, "plan": []}

        # Rationale
        rationale_match = re.search(r"##\s*rationale:\s*(.*)", response, re.IGNORECASE)
        if rationale_match:
            result["rationale"] = rationale_match.group(1).strip()

        # Score
        score_match = re.search(r"##\s*score:\s*(\d+)", response, re.IGNORECASE)
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
            except ValueError:
                result["score"] = None

        # Plan steps (lines after "## plan:")
        plan_match = re.split(r"##\s*plan:", response, flags=re.IGNORECASE)
        if len(plan_match) > 1:
            plan_block = plan_match[1]
            steps = re.findall(r"^\s*\d+\.\s*(.+)$", plan_block, flags=re.MULTILINE)
            result["plan"] = [s.strip() for s in steps if s.strip()]

        return result