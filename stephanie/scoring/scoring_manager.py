# stephanie/scoring/scoring_manager.py
import re
from pathlib import Path
from typing import Optional
import json

import yaml
from sqlalchemy.orm import Session

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_dimension import ScoreDimensionORM
from stephanie.prompts.prompt_renderer import PromptRenderer
from stephanie.scoring.calculations.score_delta import ScoreDeltaCalculator
from stephanie.scoring.calculations.weighted_average import \
    WeightedAverageCalculator
from stephanie.scoring.fallback_scorer import FallbackScorer
from stephanie.scoring.scorable import Scorable
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.score_display import ScoreDisplay
from stephanie.data.score_result import ScoreResult
from stephanie.models.score_attribute import ScoreAttributeORM

class ScoringManager(BaseAgent):
    def __init__(
        self,
        dimensions,
        prompt_loader,
        cfg,
        logger,
        memory,
        calculator=None,
        dimension_filter_fn=None,
        scorer: Optional[FallbackScorer] = None,
        scoring_profile: str = "default",
    ):
        super().__init__(cfg, memory, logger)
        self.dimensions = dimensions
        self.prompt_loader = prompt_loader
        self.output_format = cfg.get("output_format", "simple")  # default
        self.prompt_renderer = PromptRenderer(prompt_loader, cfg)
        self.calculator = calculator or WeightedAverageCalculator()
        self.dimension_filter_fn = dimension_filter_fn
        self.scoring_profile = scoring_profile
         # Initialize fallback scorer if not provided
        if scorer is None:
            from stephanie.scoring.llm_scorer import LLMScorer
            from stephanie.scoring.mrq_scorer import MRQScorer
            from stephanie.scoring.svm_scorer import SVMScorer

            svm_scorer = SVMScorer(cfg, memory, logger)
            mrq_scorer = MRQScorer(cfg, memory, logger)
            llm_scorer = LLMScorer(cfg, memory, logger, prompt_loader=prompt_loader, llm_fn=self.call_llm)

            self.scorer = FallbackScorer(
                cfg=self.cfg,
                memory=self.memory,
                logger=self.logger,
                scorers=[svm_scorer, mrq_scorer, llm_scorer],
                fallback_order=["svm", "mrq", "llm"],
                default_fallback="llm",
            )
        else:
            self.scorer = scorer

    async def run(self, context: dict) -> dict:
        """
        Main entry point for running the scoring manager.
        This method can be overridden by subclasses to implement custom logic.
        """
        # Default implementation just returns the context
        return context

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
        from stephanie.scoring.mrq_scorer import MRQScorer
        from stephanie.scoring.svm_scorer import SVMScorer

        if data["scorer"] == "mrq":
            # Use MRQ scoring profile if specified
            scorer = MRQScorer(cfg, memory, logger)
        elif data["scorer"] == "svm":
            # Use SVM scoring profile if specified
            scorer = SVMScorer(cfg, memory, logger)
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

    def evaluate(self, context: dict, scorable: Scorable, llm_fn=None):
        try:
            if self.scorer.name == "llm":
                score = self.scorer.score(
                    context, scorable, self.dimensions, llm_fn=llm_fn
                )
            else:
                score = self.scorer.score(
                    context, scorable, self.dimensions
                )
        except Exception as e:
            self.logger.log(
                "MgrScoreParseError",
                {"scorable": scorable, "error": str(e)},
            )
            score = self.evaluate_llm(context, scorable, llm_fn or self.call_llm)
        log_key = "CorDimensionEvaluated" if format == "cor" else "DimensionEvaluated"
        self.logger.log(
            log_key,
            {"ScoreCompleted": score.to_dict()},
        )

        return score

    def evaluate_llm(self, context: dict, scorable: Scorable, llm_fn=None):
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
                    "ScoreParseError",
                    {"dimension": dim["name"], "response": response, "error": str(e)},
                )
                self.handle_score_error(dim, response, e)
                score = 0.0

            result = ScoreResult(
                dimension=dim["name"],
                score=score,
                weight=dim["weight"],
                rationale=response,
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
        self.save_score_to_memory(
            bundle=bundle, scorable=scorable,
            context=context, cfg=self.cfg, memory=self.memory, logger=self.logger,
            source=self.scorer.name
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
        source,
        model_name=None,
    ):
        goal = context.get("goal")
        pipeline_run_id = context.get("pipeline_run_id")

        if not model_name:
            model_name = cfg.get("model", {}).get("name", "UnknownModel")

        eval_orm = EvaluationORM(
            goal_id=goal.get("id") if goal else None,
            pipeline_run_id=pipeline_run_id,
            target_type=scorable.target_type,
            target_id=scorable.id,
            source=source,
            agent_name=cfg.get("name"),
            model_name=model_name,
            embedding_type=memory.embedding.type,
            evaluator_name=cfg.get("evaluator", cfg.get("model_type", "ScoreEvaluator")),
            strategy=cfg.get("strategy"),
            reasoning_strategy=cfg.get("reasoning_strategy"),
            extra_data={"source": source},
        )
        memory.session.add(eval_orm)
        memory.session.flush()

        # Store all scores and attributes
        score_orms = []
        attribute_orms = []
        
        for result in bundle.results.values():
            # Create ScoreORM with core fields
            score_orm = ScoreORM(
                evaluation_id=eval_orm.id,
                dimension=result.dimension,
                score=result.score,
                source=result.source,
                weight=result.weight,
                rationale=result.rationale,
            )
            score_orms.append(score_orm)
            
            # Create attributes for ScoreAttributeORM
            if result.attributes:
                for key, value in result.attributes.items():
                    # Determine data type for proper storage
                    if isinstance(value, (int, float)):
                        data_type = "float"
                    elif isinstance(value, (list, tuple, dict)):
                        data_type = "json"
                    else:
                        data_type = "string"
                    
                    # Convert to string representation
                    if data_type == "json":
                        value_str = json.dumps(value)
                    else:
                        value_str = str(value)
                    
                    attribute_orms.append(ScoreAttributeORM(
                        score_id=None,  # Will be set after score_orm is committed
                        key=key,
                        value=value_str,
                        data_type=data_type
                    ))

        # Add all scores to database
        memory.session.add_all(score_orms)
        memory.session.flush()  # Get score IDs
        
        # Update attribute_orms with score IDs and add to session
        score_id_map = {score_orm.dimension: score_orm.id for score_orm in score_orms}
        for attr_orm in attribute_orms:
            # Find corresponding score ID (simplified for this example)
            # In practice, you'd need a better mapping strategy
            attr_orm.score_id = next(
                (id for dim, id in score_id_map.items() if dim == result.dimension), 
                None
            )
        
        memory.session.add_all(attribute_orms)
        memory.session.commit()

        # Log successful save
        scores_json = json.dumps(bundle.to_dict(include_attributes=False))
        logger.log(
            "ScoreSavedToMemory",
            {
                "goal_id": goal.get("id") if goal else None,
                "target_id": scorable.id,
                "target_type": scorable.target_type,
                "scores": scores_json,
            },
        )
        
        # Calculate weighted score for display
        weighted_score = bundle.aggregate()
        
        # Log score delta (if applicable)
        if goal and "id" in goal:
            ScoreDeltaCalculator(cfg, memory, logger).log_score_delta(
                scorable, weighted_score, goal["id"]
            )
        
        # Display score
        ScoreDisplay.show(scorable, bundle.to_dict(), weighted_score)
