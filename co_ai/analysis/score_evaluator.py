import re
from pathlib import Path

import yaml
from jinja2 import Template
from sqlalchemy.orm import Session
from tabulate import tabulate

from co_ai.models.evaluation import EvaluationORM
from co_ai.models.score_dimension import ScoreDimensionORM
from co_ai.models.score import ScoreORM


class ScoreEvaluator:
    def __init__(self, dimensions, prompt_loader, cfg, logger, memory):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg
        self.logger = logger
        self.memory = memory
        self.output_format = cfg.get("output_format", "simple")  # default fallback

    @classmethod
    def from_db(
        cls, session: Session, stage: str, prompt_loader=None, agent_config=None
    ):
        rows = session.query(ScoreDimensionORM).filter_by(stage=stage).all()
        dimensions = [
            {
                "name": row.name,
                "prompt_template": row.prompt_template,
                "weight": row.weight,
                "parser": cls.get_parser(row.extra_data or {}),
                "file": row.extra_data.get("file") if row.extra_data else None,
            }
            for row in rows
        ]
        return cls(dimensions, prompt_loader=prompt_loader, agent_config=agent_config)

    @classmethod
    def from_file(cls, filepath: str, prompt_loader, cfg, logger, memory):
        with open(Path(filepath), "r") as f:
            data = yaml.safe_load(f)

        # Default to 'simple' if not provided
        output_format = data.get("output_format", "simple")

        dimensions = [
            {
                "name": d["name"],
                "file": d.get("file"),
                "prompt_template": d.get(
                    "prompt_template", d.get("file")
                ),  # fallback to file
                "weight": d.get("weight", 1.0),
                "parser": cls.get_parser(d.get("extra_data", {})),
            }
            for d in data["dimensions"]
        ]

        # Ensure the output_format is accessible in instance
        cfg = cfg.copy()
        cfg["output_format"] = output_format

        return cls(
            dimensions=dimensions,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
            memory=memory,
        )

    @staticmethod
    def get_parser(extra_data):
        parser_type = extra_data.get("parser", "numeric")
        if parser_type == "numeric":
            return lambda r: ScoreEvaluator.extract_score_from_last_line(r)
        if parser_type == "numeric_cor":
            return lambda r: ScoreEvaluator.parse_numeric_cor(r)

        return lambda r: 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        """
        Looks for a line ending with 'score: <number>' (case-insensitive).
        """
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"score:\s*(\d+(\.\d+)?)", line.strip(), re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        """
        Extracts the numeric score from a <answer>All right [[X]]</answer> block.
        Example: <answer>[[3]]</answer> → 3.0
        """
        match = re.search(
            r"(?:<answer>\s*)?\[\[(\d+(?:\.\d+)?)\]\](?:\s*</answer>)?",
            response,
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"Could not extract numeric score from CoR-style answer: {response}"
            )
        return float(match.group(1))

    def evaluate(self, hypothesis: dict, context: dict = {}, llm_fn=None):
        if self.output_format == "cor":
            return self._evaluate_cor(
                hypothesis=hypothesis, context=context, llm_fn=llm_fn
            )
        else:
            return self._evaluate_simple(
                hypothesis=hypothesis, context=context, llm_fn=llm_fn
            )

    def _evaluate_cor(self, hypothesis: dict, context: dict = {}, llm_fn=None):
        """
        Evaluate using Chain-of-Rubrics (CoR) format with rubric, eval, and <answer>[[score]]</answer>.
        """
        if llm_fn is None:
            raise ValueError(
                "You must pass a call_llm function (e.g., agent.call_llm) to ScoreEvaluator.evaluate"
            )

        results = {}
        for dim in self.dimensions:
            # Load prompt using prompt_loader and dimension-specific CoR template
            if self.prompt_loader and dim.get("file"):
                prompt = self.prompt_loader.from_file(
                    file_name=dim["file"],
                    config=self.cfg,
                    context={"hypothesis": hypothesis, **context},
                )
            elif dim.get("prompt_template"):
                prompt = Template(dim["prompt_template"]).render(
                    hypothesis=hypothesis, **context
                )
            else:
                raise ValueError(f"No prompt found for dimension {dim['name']}")

            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
            except Exception as e:
                self.logger.log(
                    "ScoreParseError",
                    {"dimension": dim["name"], "response": response, "error": str(e)},
                )
                score = 0.0

            self.logger.log(
                "CorDimensionEvaluated",
                {"dimension": dim["name"], "score": score, "response": response},
            )

            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim["weight"],
            }

        self.save_score_to_memory(results, hypothesis, context)
        return results

    def _evaluate_simple(self, hypothesis: dict, context: dict = {}, llm_fn=None):
        if llm_fn is None:
            raise ValueError(
                "You must pass a call_llm function (e.g., agent.call_llm) to ScoreEvaluator.evaluate"
            )

        results = {}
        for dim in self.dimensions:
            if self.prompt_loader and dim.get("file"):
                prompt = self.prompt_loader.from_file(
                    file_name=dim["file"],
                    config=self.cfg,
                    context={
                        **context,
                        "goal": context.get("goal").get("goal_text"),
                        "hypothesis": hypothesis.get("text"),
                    },
                )
            else:
                prompt = Template(dim["prompt_template"]).render(
                    hypothesis=hypothesis, **context
                )

            response = llm_fn(prompt, context=context)
            score = dim["parser"](response)
            self.logger.log(
                "DimensionEvaluated",
                {"dimension": dim["name"], "score": score, "response": response},
            )
            results[dim["name"]] = {
                "score": score,
                "rationale": response,
                "weight": dim["weight"],
            }
        self.save_score_to_memory(results, hypothesis, context)
        return results

    def save_score_to_memory(self, results, hypothesis, context):
        """Save all dimension scores and associated ScoreORM entries."""
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        hypothesis_id = hypothesis.get("id")

        weighted_score = sum(
            s["score"] * s.get("weight", 1.0) for s in results.values()
        ) / max(sum(s.get("weight", 1.0) for s in results.values()), 1.0)

        scores_json = {
            "stage": self.cfg.get("stage", "review"),
            "dimensions": results,
            "final_score": round(weighted_score, 2),
        }

        # Step 1: Insert EvaluationORM
        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            hypothesis_id=hypothesis_id,
            agent_name=self.cfg.get("name"),
            model_name=self.cfg.get("model", {}).get("name"),
            evaluator_name=self.cfg.get("evaluator", "ScoreEvaluator"),
            strategy=self.cfg.get("strategy"),
            reasoning_strategy=self.cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": "ScoreEvaluator"},
        )
        self.memory.session.add(eval_orm)
        self.memory.session.flush()  # Get eval_orm.id before committing

        # Step 2: Insert ScoreORM entries
        for dimension_name, result in results.items():
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=dimension_name,
                score=result["score"],
                weight=result["weight"],
                rationale=result["rationale"],
            )
            self.memory.session.add(score)

        self.memory.session.commit()

        self.logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "hypothesis_id": hypothesis_id,
                "scores": scores_json,
            },
        )

        self.display_results(results, weighted_score)

    def display_results(self, results, weighted_score):
        table_data = [
            [
                dim_name,
                f"{dim_data['score']:.2f}",
                dim_data["weight"],
                dim_data["rationale"][:60],
            ]
            for dim_name, dim_data in results.items()
        ]
        table_data.append(["FINAL", f"{weighted_score:.2f}", "-", "Weighted average"])

        print("\n📊 Dimension Scores Summary")
        print(tabulate(
            table_data,
            headers=["Dimension", "Score", "Weight", "Rationale (preview)"],
            tablefmt="fancy_grid"
        ))
