# stephanie/agents/idea_innovation.py
from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.memory import IdeaStore
from stephanie.models.idea import IdeaORM


class IdeaInnovationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        survey_results = context.get("survey_results", [])
        search_results = context.get("search_results", [])

        # Build prompt context
        prompt_context = {
            "goal_text": goal.get("goal_text"),
            "focus_area": goal.get("focus_area"),
            "goal_type": goal.get("goal_type"),
            "strategy": goal.get("strategy"),
            "survey_summary": self._summarize_results(survey_results),
            "search_result_summaries": self._summarize_results(search_results),
            "preferences": self.cfg.get("preferences", []),
        }

        merged = {**context, **prompt_context}

        # Load and render prompt
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)

        # Call LLM to generate ideas
        raw_ideas = self.call_llm(prompt, merged)

        # Parse and structure ideas
        ideas = self._parse_raw_ideas(raw_ideas, goal)

        # Store generated ideas
        stored_ideas = self.memory.ideas.bulk_add_ideas(ideas)

        # Update context with results
        context["ideas"] = [idea.to_dict() for idea in stored_ideas]
        context["idea_ids"] = [idea.id for idea in stored_ideas]

        return context

    def _summarize_results(self, results: list) -> str:
        """Converts list of result dicts into a summary string"""
        if not results:
            return "No prior research found."
        summaries = []
        for r in results[:5]:  # limit to top 5 for brevity
            title = r.get("title", "")
            summary = (
                r.get("summary", "")[:200] + "..."
                if len(r.get("summary", "")) > 200
                else ""
            )
            url = r.get("url", "")
            summaries.append(f"- {title}: {summary} ({url})")
        return "\n".join(summaries)

    def _parse_raw_ideas(self, raw_text: str, goal: dict) -> list:
        """Parses raw LLM response into structured idea objects"""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        ideas = []

        for line in lines:
            ideas.append(
                {
                    "idea_text": line,
                    "parent_goal": goal.get("goal_text"),
                    "focus_area": goal.get("focus_area"),
                    "strategy": goal.get("strategy"),
                    "source": "generated_by_IdeaInnovationAgent",
                    "origin": "llm",
                    "extra_data": {},
                }
            )

        return ideas
