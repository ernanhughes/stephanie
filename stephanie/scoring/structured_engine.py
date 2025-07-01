# stephanie/scoring/structured_engine.py
import re
from string import Template


class StructuredScoringEngine:
    def __init__(self, dimensions, prompt_loader=None, cfg=None, logger=None):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg or {}
        self.logger = logger

    def evaluate(self, hypothesis: dict, context: dict = {}, llm_fn=None) -> dict:
        if llm_fn is None:
            raise ValueError("You must provide an llm_fn (e.g., agent.call_llm)")

        results = {}
        for dim in self.dimensions:
            prompt = self._render_prompt(dim, hypothesis, context)
            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
            except Exception as e:
                score = 0.0
                if self.logger:
                    self.logger.log("StrctScoreParseError", {
                        "dimension": dim["name"],
                        "response": response,
                        "error": str(e)
                    })
            if self.logger:
                self.logger.log("StructuredDimensionEvaluated", {
                    "dimension": dim["name"],
                    "score": score,
                    "response": response
                })
            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim.get("weight", 1.0),
            }

        results["final_score"] = self._aggregate(results)
        return results

    def _render_prompt(self, dim: dict, hypothesis: dict, context: dict) -> str:
        ctx = {"hypothesis": hypothesis, **context}
        if self.prompt_loader and dim.get("file"):
            return self.prompt_loader.from_file(file_name=dim["file"], config=self.cfg, context=ctx)
        else:
            return Template(dim["prompt_template"]).substitute(ctx)

    def _aggregate(self, results: dict) -> float:
        total = 0.0
        weight_sum = 0.0
        for dim, val in results.items():
            if not isinstance(val, dict):  # skip final_score key
                continue
            total += val["score"] * val.get("weight", 1.0)
            weight_sum += val.get("weight", 1.0)
        return round(total / weight_sum, 2) if weight_sum else 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"score:\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        match = re.search(r"<answer>\s*\[\[(\d+(?:\.\d+)?)\]\]\s*</answer>", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not extract numeric score from CoR-style answer: {response}")
        return float(match.group(1))

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return StructuredScoringEngine.extract_score_from_last_line
        if parser_type == "numeric_cor":
            return StructuredScoringEngine.parse_numeric_cor
        return lambda r: 0.0
