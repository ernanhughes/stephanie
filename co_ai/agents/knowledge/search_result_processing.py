# co_ai/agents/search_result_processing.py
from co_ai.agents.base_agent import BaseAgent


class SearchResultProcessingAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "default")
        self.output_key = cfg.get("output_key", "knowledge_base")

    async def run(self, context: dict) -> dict:
        """
        Takes raw search results from SurveyAgent or SearchOrchestratorAgent,
        processes them into structured knowledge.
        """
        goal = context.get("goal")
        raw_results = context.get("search_results", [])

        if not raw_results:
            self.logger.log("NoResultsToProcess", {})
            return context

        processed_results = []

        for result in raw_results:
            # Build prompt context
            prompt_context = {
                "goal_text": goal.get("goal_text"),
                "focus_area": goal.get("focus_area"),
                "goal_type": goal.get("goal_type"),
                "strategy": goal.get("strategy"),
                "preferences": goal.get("preferences", []),
                "title": result.get("title", ""),
                "summary": result.get("summary", ""),
                "source": result.get("source", "")
            }

            # Load prompt template
            prompt = self.prompt_loader.load_prompt(
                self.cfg.get("prompt_file", "prompts/refine_result.j2"),
                prompt_context
            )

            # Call LLM to extract insights
            response = await self.call_llm(prompt)

            try:
                structured = self._parse_refined_result(response)
                processed_results.append(structured)
            except Exception as e:
                self.logger.log("RefinementFailed", {"error": str(e), "raw_response": response})

        # Update context with refined knowledge
        context[self.output_key] = processed_results
        return context

    def _parse_refined_result(self, raw_output: str) -> dict:
        """
        Parse LLM output into structured format (assumes JSON-like structure).
        """
        import json
        return json.loads(raw_output.strip())