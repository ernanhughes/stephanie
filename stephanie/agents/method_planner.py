# stephanie/agents/method_planner.py
import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, HYPOTHESES, PIPELINE
from stephanie.models.method_plan import MethodPlanORM


class MethodPlannerAgent(BaseAgent):
    """
    The MethodPlannerAgent converts abstract research ideas into executable methodological frameworks.

    Based on NOVELSEEK's Method Development Agent:
    > _"The transformation function is represented as: T: I × T × B × L → M"_

    Where:
    - I = Research idea
    - T = Task description
    - B = Baseline implementation
    - L = Relevant literature or knowledge base
    - M = Resulting method plan

    This agent supports both initial planning and iterative refinement of methodologies.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_refinements = cfg.get("max_refinements", 3)
        self.use_refinement = cfg.get("use_refinement", True)

    async def run(self, context: dict) -> dict:
        """
        Main execution loop for the MethodPlannerAgent.

        Args:
            context (dict): Contains goal, hypotheses, baseline code, and literature summary

        Returns:
            dict: Updated context with generated method plan
        """
        # Extract input from context
        goal = context.get(GOAL, {})
        hypothesis = context.get(HYPOTHESES, [])
        baseline_approach = self._get_baseline(goal.get("focus_area"))
        literature_summary = context.get("knowledge_base_summaries", [])
        pipeline_stage = context.get(PIPELINE, "initial_method_plan")

        # Build prompt context
        prompt_context = {
            "idea": hypothesis or goal.get("goal_text"),
            "task_description": self._extract_task_description(goal),
            "baseline_approach": baseline_approach,
            "literature_summary": self._summarize_literature(literature_summary),
            "preferences": self.cfg.get("preferences", ["novelty", "feasibility"]),
        }

        merged = {**context, **prompt_context}

        # Load and render prompt
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        # Call LLM to generate method plan
        raw_plan = self.call_llm(prompt, merged)

        # Parse output into structured format
        try:
            plan_data = self.parse_method_plan_output(raw_plan)
        except Exception as e:
            self.logger.log("MethodPlanParseFailed", {"error": str(e), "raw": raw_plan})
            return context

        # Save to database
        method_plan = self._save_to_db(plan_data, merged)

        # Update context with result
        context[self.output_key] = plan_data
        context["method_plan_id"] = method_plan.id
        context["code_plan"] = plan_data.get("code_plan", "")

        self.logger.log(
            "MethodPlanGenerated", {"plan": plan_data, "pipeline_stage": pipeline_stage}
        )

        return context

    def _extract_task_description(self, goal: dict) -> str:
        """
        Extract domain-specific constraints and goals
        Example: Reaction Yield Prediction on Suzuki-Miyaura dataset using SMILES input
        """
        if goal.get("focus_area") == "chemistry":
            return f"{goal.get('goal_text')} ({goal.get('focus_area')})"

        elif goal.get("focus_area") == "nlp":
            return f"{goal.get('goal_text')} ({goal.get('focus_area')})"

        else:
            return goal.get("goal_text", "")

    def _get_baseline(self, focus_area: str) -> str:
        """
        Retrieve baseline implementation from config or file system
        """
        if focus_area == "chemistry":
            return self.cfg.get("baselines").get("reaction_yield_model", "")
        elif focus_area == "nlp":
            return self.cfg.get("baselines").get("sentiment_transformer", "")
        elif focus_area == "cv":
            return self.cfg.get("baselines").get("pointnet_classifier", "")
        else:
            return ""

    def _summarize_literature(self, literature: list) -> str:
        """
        Format literature summaries for use in prompt
        """
        if not literature:
            return "No relevant prior work found."

        return "\n".join(
            [f"- {r['title']}: {r['refined_summary']}" for r in literature[:5]]
        )

    def parse_method_plan_output(self, output: str) -> dict:
        sections = {
            "research_objective": r"\*\*Research Objective:\*\*(.*?)\n\n",
            "key_components": r"\*\*Key Components:\*\*(.*?)\n\n",
            "experimental_plan": r"\*\*Experimental Plan:\*\*(.*?)\n\n",
            "hypothesis_mapping": r"\*\*Hypothesis Mapping:\*\*(.*?)\n\n",
            "search_strategy": r"\*\*Search Strategy:\*\*(.*?)\n\n",
            "knowledge_gaps": r"\*\*Knowledge Gaps:\*\*(.*?)\n\n",
            "next_steps": r"\*\*Next Steps:\*\*(.*?)$",
        }

        result = {}
        for key, pattern in sections.items():
            match = re.search(pattern, output, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if key in ["key_components"]:
                    result[key] = [
                        line.strip() for line in content.splitlines() if line.strip()
                    ]
                else:
                    result[key] = content
            else:
                result[key] = ""

        return result

    def _save_to_db(self, plan_data: dict, context: dict) -> MethodPlanORM:
        """
        Store method plan in ORM with metadata
        """
        plan = MethodPlanORM(
            idea_text=context.get("idea"),
            task_description=plan_data.get("task_description"),
            baseline_method=plan_data.get("baseline_used"),
            literature_summary=plan_data.get("relevant_papers"),
            code_plan=plan_data.get("code_plan"),
            score_novelty=plan_data.get("score_novelty"),
            score_feasibility=plan_data.get("score_feasibility"),
            score_impact=plan_data.get("score_impact"),
            score_alignment=plan_data.get("score_alignment"),
            goal_id=context.get("goal", {}).get("id"),
            focus_area=context.get("goal", {}).get("focus_area"),
            strategy=context.get("goal", {}).get("strategy"),
            evolution_level=0,  # Initial plan
        )

        self.memory.method_plans.add_method_plan(plan.to_dict())  # Or plan.to_dict() if needed
        return plan

    def _refine_plan(self, plan: dict, feedback: dict) -> dict:
        """
        Apply refinement logic based on critique or scoring data
        """
        refinement_prompt = self.prompt_loader.load_prompt(
            "prompts/method_refine.j2", {"current_plan": plan, "feedback": feedback}
        )

        raw_refined = self.call_llm(refinement_prompt)
        return self._parse_plan_output(raw_refined)

    def _score_plan(self, plan: dict, context: dict) -> dict:
        """
        Use ScorerAgent to evaluate methodology quality
        """
        scorer = self.memory.scorer
        scores = scorer.score(plan, context)
        return scores