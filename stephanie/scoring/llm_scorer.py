# stephanie/scoring/llm_scorer.py

import re
from string import Template

from stephanie.scoring.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_result import ScoreResult
from stephanie.scoring.score_bundle import ScoreBundle


class LLMScorer(BaseScorer):
    """
    Scores a hypothesis using an LLM per dimension.
    Uses structured templates and flexible response parsers.
    """

    def __init__(self, cfg, memory, logger, prompt_loader=None, llm_fn=None):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.prompt_loader = prompt_loader
        self.llm_fn = llm_fn 

    @property
    def name(self) -> str:
        return "llm"


    def score(self, goal, scorable: Scorable, dimensions: list[dict]) -> ScoreBundle:
        """
        Scores a Scorable across multiple dimensions using an LLM.
        Returns a ScoreBundle object.
        Accepts either:
        - A list of dimension names (strings)
        - A list of dimension dicts: {name, prompt_template, weight, parser, etc.}
        """
        results = []

        for dim in dimensions:
            prompt = self._render_prompt(dim, goal, scorable)
            response = self.llm_fn(prompt, {"goal": goal})

            try:
                parser = dim.get("parser") or self._get_parser(dim)
                score = parser(response)
            except Exception as e:
                score = 0.0
                if self.logger:
                    self.logger.log(
                        "LLMScoreParseError",
                        {
                            "dimension": dim["name"],
                            "response": response,
                            "error": str(e),
                        },
                    )

            if self.logger:
                self.logger.log(
                    "LLMJudgeScorerDimension",
                    {
                        "dimension": dim["name"],
                        "score": score,
                        "rationale": response,
                    },
                )

            results.append(
                ScoreResult(
                    dimension=dim["name"],
                    score=score,
                    rationale=response,
                    weight=dim.get("weight", 1.0),
                    source="llm",
                    target_type=scorable.target_type,
                )
            )

        return ScoreBundle(results={r.dimension: r for r in results})

    def _render_prompt(self, dim: dict, goal: dict, scorable: Scorable) -> str:
        context = {
            "goal": goal,
            "scorable": scorable,
        }
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.from_file(
                file_name=dim["file"], config=self.cfg, context=context
            )
        else:
            return Template(dim["prompt_template"]).substitute(context)

    def _default_prompt(self, dimension):
        return (
            "Evaluate the following document based on $dimension:\n\n"
            "Goal: $goal\nHypothesis: $scorable\n\n"
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
            match = re.search(
                r"score[:\-]?\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE
            )
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        match = re.search(
            r"<answer>\s*\[\[(\d+(?:\.\d+)?)\]\]\s*</answer>", response, re.IGNORECASE
        )
        if not match:
            raise ValueError(
                f"Could not extract numeric score from CoR-style answer: {response}"
            )
        return float(match.group(1))

    def _get_parser(self, dim: dict):
        parser_type = dim.get("parser", "numeric")
        if parser_type == "numeric":
            return self.extract_score_from_last_line
        if parser_type == "numeric_cor":
            return self.parse_numeric_cor
        return lambda r: 0.0
