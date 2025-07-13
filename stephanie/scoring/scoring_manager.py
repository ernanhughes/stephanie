# stephanie/scoring/scoring_manager.py
import re
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_dimension import ScoreDimensionORM
from stephanie.prompts.prompt_renderer import PromptRenderer
from stephanie.scoring.calculations.score_delta import ScoreDeltaCalculator
from stephanie.scoring.calculations.weighted_average import \
    WeightedAverageCalculator
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.score_display import ScoreDisplay
from stephanie.scoring.score_result import ScoreResult


class ScoringManager:
    def __init__(
        self,
        dimensions,
        prompt_loader,
        cfg,
        logger,
        memory,
        calculator=None,
        dimension_filter_fn=None,
        scoring_profile=None,
        scorer=None,
    ):
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.cfg = cfg
        self.logger = logger
        self.memory = memory
        self.output_format = cfg.get("output_format", "simple")  # default
        self.prompt_renderer = PromptRenderer(prompt_loader, cfg)
        self.calculator = calculator or WeightedAverageCalculator()
        self.dimension_filter_fn = dimension_filter_fn
        self.scoring_profile = scoring_profile
        self.scorer = scorer

    def dimension_names(self):
        """Returns the names of all dimensions."""
        return [dim["name"] for dim in self.dimensions]

    def filter_dimensions(self, scorable, context):
        """
        Returns the list of dimensions to use for this evaluation.
        Override or provide a hook function to filter dynamically.
        """
        if self.dimension_filter_fn:
            return self.dimension_filter_fn(self.dimensions, scorable, context)
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
            return lambda s: round((1.2**s) - 1, 2)
        return lambda s: s  # Default is identity

    @classmethod
    def from_db(
        cls,
        session: Session,
        stage: str,
        prompt_loader=None,
        cfg=None,
        logger=None,
        memory=None,
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
        return cls(
            dimensions,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
            memory=memory,
        )

    def get_dimensions(self):
        return [d["name"] for d in self.dimensions]

    @classmethod
    def from_file(
        cls,
        filepath: str,
        prompt_loader,
        cfg,
        logger,
        memory,
        scoring_profile=None,
        llm_fn=None,
    ):
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

        from stephanie.scoring.llm_scorer import LLMScorer
        from stephanie.scoring.mrq.mrq_scorer import MRQScorer
        from stephanie.scoring.svm_scorer import SVMScorer

        if data["scorer"] == "mrq":
            # Use MRQ scoring profile if specified
            scorer = MRQScorer(cfg, memory, logger)
            scorer.load_models()
        elif data["scorer"] == "svm":
            # Use SVM scoring profile if specified
            scorer = SVMScorer(cfg, memory, logger)
            scorer.load_models()
        else:
            # Default to LLM scoring profile
            scorer = LLMScorer(
                cfg, memory, logger, prompt_loader=prompt_loader, llm_fn=llm_fn
            )

        return cls(
            dimensions=dimensions,
            prompt_loader=prompt_loader,
            cfg=cfg,
            logger=logger,
            memory=memory,
            scoring_profile=scoring_profile,
            scorer=scorer,
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

    def evaluate(self, scorable: Scorable, context: dict = {}, llm_fn=None):
        try:
            score = self.scorer.score(
                context.get("goal"), scorable, self.dimension_names(), llm_fn=llm_fn
            )
        except Exception as e:
            self.logger.log(
                "MgrScoreParseError",
                {"scorable": scorable, "error": str(e)},
            )
            score = self.evaluate_llm(scorable, context, llm_fn)
        log_key = "CorDimensionEvaluated" if format == "cor" else "DimensionEvaluated"
        self.logger.log(
            log_key,
            {"ScoreCompleted": score.to_dict()},
        )

        return score

    def evaluate_llm(self, scorable: Scorable, context: dict = {}, llm_fn=None):
        if llm_fn is None:
            raise ValueError("You must pass a call_llm function to evaluate")

        results = []
        force_rescore = context.get("force_rescore", False)

        # Use filter_dimensions if available
        dimensions_to_use = self.filter_dimensions(scorable, context)

        for dim in dimensions_to_use:
            print(f"Evaluating dimension: {dim['name']}")
            prompt = self.prompt_renderer.render(
                dim, {"hypothesis": scorable, **context}
            )

            prompt_hash = ScoreORM.compute_prompt_hash(prompt, scorable)
            if not force_rescore:
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
                target_type=scorable.target_type,
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
        self.save_score_to_memory(
            bundle, scorable, context, self.cfg, self.memory, self.logger
        )
        return bundle

    def handle_score_error(self, dim, response, error):
        if self.cfg.get("fail_silently", True):
            return 0.0
        raise ValueError(f"Failed to parse score {response} for {dim['name']}: {error}")

    @staticmethod
    def save_score_to_memory(
        bundle: ScoreBundle,
        scorable: Scorable,
        context: dict,
        cfg: dict,
        memory,
        logger,
        source="ScoreEvaluator",
        model_name=None,
    ):
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        weighted_score = bundle.calculator.calculate(bundle)

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        if not model_name:
            model_name = cfg.get("model", {}).get("name", "UnknownModel")

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            target_type=scorable.target_type,
            target_id=scorable.id,
            agent_name=cfg.get("name"),
            model_name=model_name,
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
                source=score_result.source,
                weight=score_result.weight,
                rationale=score_result.rationale,
                prompt_hash=score_result.prompt_hash
                or ScoreORM.compute_prompt_hash(goal.get("goal_text", ""), scorable),
            )
            memory.session.add(score)

        memory.session.commit()

        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id"),
                "target_id": scorable.id,
                "target_type": scorable.target_type,
                "scores": scores_json,
            },
        )
        ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
            scorable, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)

    @staticmethod
    def save_document_score_to_memory(
        bundle, document, context, cfg, memory, logger, source="DocumentEvaluator"
    ):
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")
        document_id = document.get("id")
        weighted_score = bundle.calculator.calculate(bundle)

        soring_text = ScoringManager.get_scoring_text(document)
        scorable = Scorable(
            text=soring_text, target_type=TargetType.DOCUMENT, id=document_id
        )

        scores_json = {
            "stage": cfg.get("stage", "review"),
            "dimensions": bundle.to_dict(),
            "final_score": round(weighted_score, 2),
        }

        eval_orm = EvaluationORM(
            goal_id=goal.get("id"),
            pipeline_run_id=pipeline_run_id,
            target_type=TargetType.DOCUMENT,
            target_id=document_id,
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
            scorable, weighted_score, goal.get("id")
        )
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)

    @staticmethod
    def get_scoring_text(document: dict) -> str:
        if document.get("summary"):
            return f"{document.get('title', '')}\n\n{document['summary']}".strip()
        elif document.get("content"):
            return document["content"][:1500]  # Safely truncate
        else:
            return document.get("title", "")
