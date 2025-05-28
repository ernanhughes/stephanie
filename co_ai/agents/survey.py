# co_ai/agents/survey.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL


class SurveyAgent(BaseAgent):
    """
    The Survey Agent generates adaptive search queries for literature exploration.
    
    From the paper:
    > 'The Survey Agent deconstructs the research task into multiple keyword combinations'
    > 'It supports two distinct modes: literature review mode and deep research mode'
    > 'Each idea is mapped to testable components before being executed'
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_queries = cfg.get("max_queries", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, {})
        if not goal:
            self.logger.log("NoGoalProvided", {"reason": "survey_agent_skipped"})
            return context

        # Generate new queries based on goal + baseline + preferences
        prompt_context = {
            "goal_text": goal.get("goal_text"),
            "focus_area": goal.get("focus_area"),
            "baseline_method": context.get("baseline_method", ""),
            "preferences": context.get("preferences", ["novelty", "feasibility"]),
            "previous_ideas": context.get("ideas", [])
        }
        merged = {**self.cfg, **prompt_context}

        prompt = self.prompt_loader.load_prompt(self.cfg, merged)


        raw_output = self.call_llm(prompt, context)
        queries = self._parse_query_response(goal, raw_output)

        # Store in context for SearchOrchestratorAgent
        context["search_queries"] = queries
        context["search_strategy"] = self.strategy

        self.logger.log("SurveyQueriesGenerated", {
            "queries": queries,
            "strategy_used": self.strategy,
            "pipeline_stage": context.get("pipeline_stage")
        })

        return context

    def _parse_query_response(self, goal, response: str) -> list:
        """Parse LLM output into clean list of search queries"""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if not lines:
            # Fallback strategy
            return [
                f"{goal.get('focus_area')} machine learning",
                f"{goal.get('goal_text')}"
            ]
        return lines[:self.max_queries]

    def expand_queries_to_goals(self, queries: list, base_goal: dict) -> list:
        """
        Convert queries into sub-goals for future pipeline stages
        
        Args:
            queries (list): Generated search strings
            base_goal (dict): Original goal
            
        Returns:
            list: List of structured sub-goals
        """
        return [
            {
                "goal_text": q,
                "parent_goal": base_goal.get("goal_text"),
                "focus_area": base_goal.get("focus_area"),
                "strategy": base_goal.get("strategy"),
                "source": "survey_agent"
            }
            for q in queries
        ]