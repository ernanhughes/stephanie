# stephanie/scoring/llm_scorer.py

import re
from string import Template

from stephanie.scoring.base_scorer import BaseScorer


class LLMScorer(BaseScorer):
    """
    Scores a hypothesis using an LLM per dimension.
    Uses structured templates and flexible response parsers.
    """

    def __init__(self, cfg, memory, logger, prompt_loader=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader

    def score(self, goal, hypothesis, dimensions):
        """
        Returns dict of dimension -> {score, rationale, weight} + final_score.
        Accepts either:
        - A list of dimension names (strings)
        - A list of dimension dicts: {name, prompt_template, weight, parser, etc.}
        """
        results = {}

        for dim in dimensions:
            if isinstance(dim, str):
                dim = {"name": dim, "prompt_template": self._default_prompt(dim)}

            prompt = self._render_prompt(dim, goal, hypothesis)
            response = self.call_llm(prompt)

            try:
                parser = dim.get("parser_fn") or self._get_parser(dim)
                score = parser(response)
            except Exception as e:
                score = 0.0
                if self.logger:
                    self.logger.log("LLMScoreParseError", {
                        "dimension": dim["name"],
                        "response": response,
                        "error": str(e)
                    })

            if self.logger:
                self.logger.log("LLMJudgeScorerDimension", {
                    "dimension": dim["name"],
                    "score": score,
                    "rationale": response,
                })

            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim.get("weight", 1.0),
            }

        results["final_score"] = self._aggregate(results)
        return results

    def _render_prompt(self, dim: dict, goal: dict, hypothesis: dict) -> str:
        context = {
            "goal": goal.get("goal_text", ""),
            "hypothesis": hypothesis.get("text", "")
        }
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.from_file(file_name=dim["file"], config=self.cfg, context=context)
        else:
            return Template(dim["prompt_template"]).substitute(context)

    def _default_prompt(self, dimension):
        return (
            "Evaluate the following hypothesis based on $dimension:\n\n"
            "Goal: $goal\nHypothesis: $hypothesis\n\n"
            "Respond with a score and rationale."
        ).replace("$dimension", dimension)

    def _aggregate(self, results: dict) -> float:
        total = 0.0
        weight_sum = 0.0
        for dim, val in results.items():
            if not isinstance(val, dict):
                continue
            total += val["score"] * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)
        return round(total / weight_sum, 2) if weight_sum else 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"score[:\-]?\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        match = re.search(r"<answer>\s*\[\[(\d+(?:\.\d+)?)\]\]\s*</answer>", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not extract numeric score from CoR-style answer: {response}")
        return float(match.group(1))

    def _get_parser(self, dim: dict):
        parser_type = dim.get("parser", "numeric")
        if parser_type == "numeric":
            return self.extract_score_from_last_line
        if parser_type == "numeric_cor":
            return self.parse_numeric_cor
        return lambda r: 0.0
