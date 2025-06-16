import re
from pathlib import Path

import yaml
from jinja2 import Template
from sqlalchemy.orm import Session

from co_ai.models.document_score import DocumentEvaluationORM, DocumentScoreORM


class DocumentScoreEvaluator:
    def __init__(self, dimensions, prompt_loader, cfg, logger, memory):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg
        self.logger = logger
        self.memory = memory

    @classmethod
    def from_file(cls, filepath: str, prompt_loader, cfg, logger, memory):
        with open(Path(filepath), "r") as f:
            data = yaml.safe_load(f)

        dimensions = [
            {
                "name": d["name"],
                "file": d.get("file"),
                "prompt_template": d.get("prompt_template", d.get("file")),
                "weight": d.get("weight", 1.0),
                "parser": cls.get_parser(d.get("extra_data", {})),
            }
            for d in data["dimensions"]
        ]

        return cls(dimensions, prompt_loader, cfg, logger, memory)

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return lambda r: DocumentScoreEvaluator.extract_score_from_response(r)
        return lambda r: 0.0

    @staticmethod
    def extract_score_from_response(response: str) -> float:
        """
        Extracts justification + a numeric score (out of 100) from a line like:
        "Justification... Score: 88"
        """
        match = re.search(r"score:\s*(\d{1,3})", response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0

    def evaluate(self, document: dict, context: dict = {}, llm_fn=None):
        if llm_fn is None:
            raise ValueError("Must pass a `llm_fn` to evaluate documents.")

        results = {}
        for dim in self.dimensions:
            if self.prompt_loader and dim.get("file"):
                prompt = self.prompt_loader.from_file(
                    file_name=dim["file"],
                    config=self.cfg,
                    context={"document": document, **context},
                )
            elif dim.get("prompt_template"):
                prompt = Template(dim["prompt_template"]).render(
                    document=document, **context
                )
            else:
                raise ValueError(f"No prompt found for dimension {dim['name']}")

            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
            except Exception as e:
                self.logger.log(
                    "DocumentScoreParseError",
                    {"dimension": dim["name"], "response": response, "error": str(e)},
                )
                score = 0.0

            self.logger.log(
                "DocumentDimensionEvaluated",
                {"dimension": dim["name"], "score": score, "response": response},
            )

            results[dim["name"]] = {
                "score": score,
                "rationale": response.strip(),
                "weight": dim["weight"],
            }

        self.save_score_to_memory(results, document, context)
        return results

    def save_score_to_memory(self, results, document, context):
        weighted_score = sum(
            s["score"] * s.get("weight", 1.0) for s in results.values()
        ) / max(sum(s.get("weight", 1.0) for s in results.values()), 1.0)

        scores_json = {
            "stage": self.cfg.get("stage", "review"),
            "dimensions": results,
            "final_score": round(weighted_score, 2),
        }

        evaluation = DocumentEvaluationORM(
            document_id=document.get("id"),
            agent_name=self.cfg.get("name"),
            model_name=self.cfg.get("model", {}).get("name"),
            evaluator_name=self.cfg.get("evaluator", "DocumentScoreEvaluator"),
            strategy=self.cfg.get("strategy"),
            scores=scores_json,
            extra_data={"source": "DocumentScoreEvaluator"},
        )

        self.memory.session.add(evaluation)
        self.memory.session.flush()  # to get evaluation.id

        for dim_name, result in results.items():
            self.memory.session.add(DocumentScoreORM(
                evaluation_id=evaluation.id,
                dimension=dim_name,
                score=result["score"],
                weight=result["weight"],
                rationale=result["rationale"]
            ))

        self.memory.session.commit()

        self.logger.log("DocumentScoreSaved", {
            "document_id": document.get("id"),
            "scores": scores_json,
        })
