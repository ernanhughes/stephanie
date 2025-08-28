# stephanie/agents/planning/planner_reuse.py
from stephanie.scoring.scorable_ranker import ScorableRanker
from stephanie.scoring.scorable import Scorable
from stephanie.models.report import ReportORM  # ✅ assuming this is your reporting ORM
import json
import time


class PlannerReuseAgent:
    """
    Retrieves similar past PlanTraces/Scorables and adapts them into a new plan.
    Writes a structured report into SYS for visibility.
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.ranker = ScorableRanker(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        if not goal_text:
            return context

        # --- 1. Retrieve candidate past traces ---
        candidates = [
            Scorable(id=pt.trace_id, text=pt.goal_text, target_type="plan_trace")
            for pt in self.memory.plan_traces.get_all(limit=200)
        ]

        if not candidates:
            self.logger.log("PlannerReuseNoCandidates", {"goal_text": goal_text})
            return context

        ranked = self.ranker.score(goal_text, candidates)
        top = ranked[:3]  # take top 3

        # --- 2. Adaptation step (LLM) ---
        examples = []
        for c in top:
            pt = self.memory.plan_traces.get(c["scorable_id"])
            if not pt:
                continue
            examples.append({
                "goal": pt.goal_text,
                "plan": pt.plan_signature,
            })

        prompt = {
            "system": "You are a planner. Adapt past successful plans to the new goal.",
            "user": {
                "new_goal": goal_text,
                "examples": examples,
            }
        }
        new_plan = self.memory.llm.generate_json(prompt).get("plan", [])

        # --- 3. Update context ---
        context["reused_plan"] = new_plan

        # --- 4. Report to SYS ---
        report_data = {
            "goal": goal_text,
            "reused_plan": new_plan,
            "based_on": [c["scorable_id"] for c in top],
            "examples": examples,
        }

        report = ReportORM(
            agent_name="PlannerReuseAgent",
            run_id=context.get("pipeline_run_id"),
            content=json.dumps(report_data, indent=2),
            created_at=int(time.time())
        )
        self.memory.reports.add(report)

        # --- 5. Logging ---
        self.logger.log("PlannerReuseGenerated", {
            "goal_text": goal_text,
            "plan": new_plan,
            "based_on": [c["scorable_id"] for c in top],
            "report_id": report.id if hasattr(report, "id") else None,
        })

        return context
