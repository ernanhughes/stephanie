import re
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from co_ai.models.evaluation import EvaluationORM
from co_ai.models.score import ScoreORM
from co_ai.models.score_dimension import ScoreDimensionORM
from co_ai.prompts.prompt_renderer import PromptRenderer
from co_ai.scoring.calculations.score_delta import ScoreDeltaCalculator
from co_ai.scoring.calculations.weighted_average import \
    WeightedAverageCalculator
from co_ai.scoring.score_bundle import ScoreBundle
from co_ai.scoring.score_display import ScoreDisplay
from co_ai.scoring.score_result import ScoreResult


class ScoringManager:
    def __init__(self, dimensions, prompt_loader, cfg, logger, memory, calculator=None, dimension_filter_fn=None):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg
        self.logger = logger
        self.memory = memory
        self.output_format = cfg.get("output_format", "simple")  # default
        self.prompt_renderer = PromptRenderer(prompt_loader, cfg)
        self.calculator = calculator or WeightedAverageCalculator()
        self.dimension_filter_fn = dimension_filter_fn  # Optional hook

    def filter_dimensions(self, hypothesis, context):
        """
        Returns the list of dimensions to use for this evaluation.
        Override or provide a hook function to filter dynamically.
        """
        if self.dimension_filter_fn:
            return self.dimension_filter_fn(self.dimensions, hypothesis, context)
        return self.dimensions

    @staticmethod
    def get_postprocessor(extra_data):
        """Returns a postprocessor function based on the 'postprocess' key."""
        ptype = extra_data.get("postprocess")
        if ptype == "clip_0_5":
            return lambda s: max(0, min(s, 5))
        if ptype == "normalize_10":
            return lambda s: min(s / 10.0, 1.0)
        if ptype == "exp_boost":
            return lambda s: round((1.2 ** s) - 1, 2)
        return lambda s: s  # Default is identity

    @classmethod
    def from_db(
        cls, session: Session, stage: str, prompt_loader=None, cfg=None, logger=None, memory=None
    ):
        rows = session.query(ScoreDimensionORM).filter_by(stage=stage).all()
        dimensions = [
            {
                "name": row.name,
                "prompt_template": row.prompt_template,
                "weight": row.weight,
                "parser": cls.get_parser(row.extra_data or {}),
                "file": row.extra_data.get("file") if row.extra_data else None,
                "postprocess": cls.get_postprocessor(row.extra_data or {}), 
            }
            for row in rows
        ]
        return cls(dimensions, prompt_loader=prompt_loader, cfg=cfg, logger=logger, memory=memory)

    def get_dimensions(self):
        return [d["name"] for d in self.dimensions]


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
                "postprocess": cls.get_postprocessor(d.get("extra_data", {})), 
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
            return lambda r: ScoringManager.extract_score_from_last_line(r)
        if parser_type == "numeric_cor":
            return lambda r: ScoringManager.parse_numeric_cor(r)

        return lambda r: 0.0

    @staticmethod
    def extract_score_from_last_line(response: str) -> float:
        """
        Extracts a numeric score from any line containing 'score: <number>' (case-insensitive),
        scanning in reverse for the most recent score mention.
        """
        lines = response.strip().splitlines()
        for line in reversed(lines):
            match = re.search(r"\bscore:\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    @staticmethod
    def parse_numeric_cor(response: str) -> float:
        """
        Extracts a numeric score from a <answer> block or a line containing 'score: <number>'.
        """
        # Try CoR-style first
        match = re.search(
            r"<answer>\s*\[*\s*(\d+(?:\.\d+)?)\s*\]*\s*</answer>",
            response,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))

        # Fallback to score: X
        match = re.search(r"\bscore:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        raise ValueError(f"Could not extract numeric score from response: {response}")

    def evaluate(self, hypothesis: dict, context: dict = {}, llm_fn=None, is_document=False):
        if self.output_format == "cor":
            return self._evaluate(hypothesis, context, llm_fn, format="cor", is_document=is_document)
        else:
            return self._evaluate(hypothesis, context, llm_fn, format="simple", is_document=is_document)

    def _evaluate(self, hypothesis: dict, context: dict, llm_fn, format="simple", is_document=False):
        if llm_fn is None:
            raise ValueError("You must pass a call_llm function to evaluate")

        results = []
    
         # Use filter_dimensions if available
        dimensions_to_use = self.filter_dimensions(hypothesis, context)

        for dim in dimensions_to_use:
            print(f"Evaluating dimension: {dim['name']}")
            prompt = self.prompt_renderer.render(
                dim, {"hypothesis": hypothesis, **context}
            )
            prompt_hash = str(hash(prompt + str(hypothesis.get("id", ""))))
            # 2. Check cache or memory for existing score
            cached_result = self.memory.scores.get_score_by_prompt_hash(prompt_hash)
            if cached_result:
                self.logger.log("ScoreCacheHit", {"dimension": dim["name"]})
                result = cached_result
                results.append(result)
                continue

            response = llm_fn(prompt, context=context)
            try:
                score = dim["parser"](response)
                score = dim.get("postprocess", lambda s: s)(score) 
            except Exception as e:
                self.logger.log(
                    "MgrScoreParseError",
                    {"dimension": dim["name"], "response": response, "error": str(e)},
                )
                self.handle_score_error(dim, response, e)
                score = 0.0

            result = ScoreResult(
                dimension=dim["name"],
                score=score,
                weight=dim["weight"],
                rationale=response,
                prompt_hash=prompt_hash,
                source="llm",
            )
            results.append(result)

            log_key = (
                "CorDimensionEvaluated" if format == "cor" else "DimensionEvaluated"
            )
            self.logger.log(
                log_key,
                {"dimension": dim["name"], "score": score, "response": response},
            )

        bundle = ScoreBundle(results={r.dimension: r for r in results})
        if is_document:
            self.save_document_score_to_memory(bundle, hypothesis, context, self.cfg, self.memory, self.logger)
        else:
            self.save_score_to_memory(bundle, hypothesis, context, self.cfg, self.memory, self.logger)
        return bundle

    def handle_score_error(self, dim, response, error):
        if self.cfg.get("fail_silently", True):
            return 0.0
        raise ValueError(f"Failed to parse score {response} for {dim['name']}: {error}")

    @staticmethod
    def save_score_to_memory(bundle, hypothesis, context, cfg, memory, logger, source="ScoreEvaluator"):
        
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        hypothesis_id = hypothesis.get("id")
        weighted_score = bundle.calculator.calculate(bundle)

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            hypothesis_id=hypothesis_id,
            agent_name=cfg.get("name"),
            model_name=cfg.get("model", {}).get("name"),
            evaluator_name=cfg.get("evaluator", "ScoreEvaluator"),
            strategy=cfg.get("strategy"),
            reasoning_strategy=cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": source},
        )
        memory.session.add(eval_orm)
        memory.session.flush()

        for result in bundle.results:
            score_result = bundle.results[result]
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=score_result.dimension,
                score=score_result.score,
                weight=score_result.weight,
                rationale=score_result.rationale,
                prompt_hash=score_result.prompt_hash,
            )
            memory.session.add(score)

        memory.session.commit()

        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "hypothesis_id": hypothesis_id,
                "scores": scores_json,
            },
        )
        ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
            hypothesis_id, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(bundle.to_dict(), weighted_score)


    @staticmethod
    def save_document_score_to_memory(bundle, document, context, cfg, memory, logger, source="DocumentEvaluator"):
        
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        document_id = document.get("id")
        weighted_score = bundle.calculator.calculate(bundle)

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            document_id=document_id,
            agent_name=cfg.get("name"),
            model_name=cfg.get("model", {}).get("name"),
            evaluator_name=cfg.get("evaluator", "ScoreEvaluator"),
            strategy=cfg.get("strategy"),
            reasoning_strategy=cfg.get("reasoning_strategy"),
            scores=scores_json,
            extra_data={"source": source},
        )
        memory.session.add(eval_orm)
        memory.session.flush()

        for result in bundle.results:
            score_result = bundle.results[result]
            score = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=score_result.dimension,
                score=score_result.score,
                weight=score_result.weight,
                rationale=score_result.rationale,
                prompt_hash=score_result.prompt_hash,
            )
            memory.session.add(score)

        memory.session.commit()

        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "hypothesis_id": document_id,
                "scores": scores_json,
            },
        )
        ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
            document_id, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(bundle.to_dict(), weighted_score)
