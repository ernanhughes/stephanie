# stephanie/agents/knowledge/dimension_generator.py

import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class DimensionGeneratorAgent(BaseAgent):
    """
    Generates goal-informed evaluation dimensions dynamically.

    Instead of relying on fixed domains or pre-defined scoring axes,
    this agent reads the current goal and invents relevant dimensions
    like 'stability', 'generalization', 'adaptivity', etc.

    These dimensions are then used by the IdeaParser and SVMScorer
    to evaluate ideas in context.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Settings
        self.default_dimensions = cfg.get(
            "default_dimensions",
            ["usefulness", "novelty", "alignment", "epistemic_gain"]
        )
        self.max_dimensions = cfg.get("max_dimensions", 6)
        self.min_similarity_score = cfg.get("min_similarity_score", 0.65)
        self.use_memory = cfg.get("use_memory", True)

        # Injected tools
        self.prompt_loader = None
        self.call_llm = None

    async def run(self, context: dict) -> dict:
        """
        Main pipeline:
        1. Get goal from context
        2. Generate dimensions based on goal + memory
        3. Store them in context for downstream scorers
        """
        try:
            goal = context.get(GOAL, {})
            goal_text = goal.get("goal_text", "").strip()
            goal_id = goal.get("id")

            if not goal_text:
                self.logger.log("NoGoalText", {"stage": "dimension_generation"})
                context["dimensions"] = self.default_dimensions
                return context

            # Step 1: Use memory to find similar past goals
            similar_goals = []
            if self.use_memory:
                similar_goals = self._find_similar_goals(goal_text, top_k=3)

            # Step 2: Prompt LLM to generate dimensions
            prompt_context = {
                "goal_text": goal_text,
                "similar_goals": [g["goal_text"] for g in similar_goals],
                "existing_dimensions": self.default_dimensions,
                "max_dimensions": self.max_dimensions,
            }

            prompt = self.prompt_loader.from_file(
                self.cfg.get("dimension_prompt_file", "generate_dimensions.txt"),
                self.cfg, prompt_context
            )

            raw_output = self.call_llm(prompt, prompt_context)
            dimensions = self._parse_dimension_response(raw_output)

            # Step 3: Optionally filter duplicates or irrelevant ones
            filtered = self._filter_irrelevant_dimensions(dimensions)

            # Step 4: Store in memory for future reuse
            if goal_id and filtered:
                self._save_dimensions_to_goal(goal_id, filtered)

            # Step 5: Return in context
            context["dimensions"] = filtered
            context["dynamic_dimensions"] = filtered
            context["dimension_source"] = "DimensionGeneratorAgent"

            self.logger.log(
                "DimensionsGenerated",
                {
                    "count": len(filtered),
                    "dimensions": filtered,
                    "source": "LLM"
                }
            )

            return context

        except Exception as e:
            self.logger.log("DimensionGenerationFailed", {"error": str(e)})
            context["dimensions"] = self.default_dimensions
            return context

    def _find_similar_goals(self, goal_text: str, top_k: int = 3) -> list:
        """
        Find previous goals with similar intent using embeddings
        """
        self.memory.embedding.get_or_create(goal_text)
        embedding_id = self.memory.embedding.get_id_for_text(goal_text)

        results = self.memory.goal_embeddings.find_similar(embedding_id, top_k=top_k)

        similar_goals = []
        for result in results:
            goal_id = result[0]
            score = result[1]
            if score > self.min_similarity_score:
                goal_data = self.memory.goals.get(goal_id)
                similar_goals.append({
                    "goal_id": goal_id,
                    "goal_text": goal_data.text,
                    "score": score
                })

        return similar_goals

    def _parse_dimension_response(self, response: str) -> list:
        """
        Parses LLM output into clean dimension list.
        Handles both numbered lists and free-form text.
        """
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        dimensions = []

        for line in lines:
            # Match bullet points or numbers
            match = re.match(r"(?:\d+\.|\-)\s*([a-zA-Z][a-zA-Z_\-\s]+)", line)
            if match:
                dim = match.group(1).strip().lower()
                dim = re.sub(r"[^\w\s]", "", dim)
                dim = re.sub(r"\s+", "_", dim)
                if dim:
                    dimensions.append(dim)

        # Remove duplicates while preserving order
        seen = set()
        unique_dims = []
        for d in dimensions:
            if d not in seen:
                seen.add(d)
                unique_dims.append(d)

        return unique_dims[:self.max_dimensions]

    def _filter_irrelevant_dimensions(self, dimensions: list) -> list:
        """
        Optional filtering step to remove low-value dimensions.
        Could be replaced with an SVM or heuristic-based filter.
        """
        common_stopwords = [
            "relevance", "importance", "impact", "quality", "value", "meaning"
        ]
        return [d for d in dimensions if d not in common_stopwords]

    def _save_dimensions_to_goal(self, goal_id: int, dimensions: list):
        """
        Stores generated dimensions with the goal for future reference
        """
        for i, dim in enumerate(dimensions):
            self.memory.goal_dimensions.insert({
                "goal_id": goal_id,
                "dimension": dim,
                "rank": i + 1
            })

        self.logger.log("GoalDimensionsSaved", {"goal_id": goal_id, "dimensions": dimensions})