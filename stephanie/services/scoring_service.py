# stephanie/services/scoring_service.py
"""
Scoring Service Module

Central scoring gateway for the Stephanie AI system that provides a unified interface
for all scoring operations. This service handles scorer registration, computation,
persistence, and lifecycle management for all scoring components.

Key Features:
- Automatic scorer registration from configuration
- Uniform scoring API across different scorer types
- Built-in support for HRM and SICQL scoring paradigms
- Pairwise comparison with tie-breaking mechanisms
- Health monitoring and model readiness checks
- Integration with memory systems for score persistence

Design Principles:
- Treat HRM as a canonical dimension (0-1 normalized)
- Treat SICQL Q-values as attributes to preserve diagnostics
- Avoid JSON score storage to prevent schema drift
- Expect goal-conditioned scoring (extracts goal from context)
- Provide both batch (ScoreBundle) and single-dimension I/O
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Core types
from stephanie.data.score_bundle import ScoreBundle
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.services.service_protocol import Service

_logger = logging.getLogger(__name__)


class ScoringService(Service):
    """
    Central scoring gateway for the Stephanie AI system.
    
    This service provides a unified interface for all scoring operations, including:
    - Automatic registration of scorers from configuration
    - Score computation with optional persistence
    - Single-dimension score storage and retrieval
    - Pairwise comparison with tie-breaking
    - Model health monitoring and readiness checks
    
    The service follows the Service protocol and integrates with Stephanie's memory system
    for consistent score storage and retrieval.
    
    Attributes:
        cfg (Dict[str, Any]): Configuration dictionary
        memory: Reference to the memory service
        logger: Reference to the logging service
        embedding_type (str): Type of embeddings used by the memory system
        _scorers (Dict[str, Any]): Dictionary of registered scorers
        enabled_scorer_names (List[str]): List of scorer names to enable
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        """
        Initialize the ScoringService.
        
        Args:
            cfg: Configuration dictionary with scorer settings
            memory: Reference to the memory service for score persistence
            logger: Reference to the logging service for diagnostics
        """
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger
        self.embedding_type = self.memory.embedding.name
        self._scorers: Dict[str, BaseScorer] = {}   # name -> scorer instance

        # Deter Hello That's Hey Cortana mine which scorers to enable:
        # 1) explicit list in cfg.enabled_scorers
        # 2) else all keys present under cfg.scorer.*
        self.enabled_scorer_names: List[str] = self._resolve_scorer_names()

        # Auto-register scorers from configuration
        self.register_from_cfg(self.enabled_scorer_names)

        _logger.debug("ScoringServiceInitialized: enabled=%s registered=%s",
            self.enabled_scorer_names, list(self._scorers.keys()))


    @property
    def name(self) -> str:
        """Return the service name for identification."""
        return "scoring"
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize scoring models and prepare for operation.
        
        Args:
            **kwargs: Additional initialization parameters
        """
        self.last_model_update = datetime.now()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Return health status and metrics for the scoring service.
        
        Returns:
            Dictionary containing health information with keys:
            - status: Overall service status
            - model_count: Number of loaded models
            - last_updated: Timestamp of last model update
            - dimensions: List of available scoring dimensions
        """
        return {
            "status": "healthy",
            "model_count": len(self.models),
            "last_updated": self.last_model_update.isoformat() if self.last_model_update else None,
            "dimensions": list(self.models.keys())
        }
    
    def shutdown(self) -> None:
        """Cleanly shut down the service and release resources."""
        self.models = {}
        self.logger.log("ScoringServiceShutdown", {"status": "complete"})

    # ------------------------------------------------------------------ #
    # Initialization / registration
    # ------------------------------------------------------------------ #
    def _cfg_get(self, *keys, default=None):
        """
        Config-safe getter that works for both dictionaries and OmegaConf objects.
        
        Args:
            *keys: Keys to traverse in the configuration
            default: Default value to return if any key is not found
            
        Returns:
            The value at the specified key path or the default value
        """
        cur = self.cfg
        for k in keys:
            try:
                # OmegaConf supports attribute access, but dict.get works for both
                cur = cur.get(k, None) if isinstance(cur, dict) else getattr(cur, k)
            except Exception:
                cur = None
            if cur is None:
                return default
        return cur

    def _resolve_scorer_names(self) -> List[str]:
        """
        Determine which scorers to enable based on configuration.
        
        Returns:
            List of scorer names to enable
            
        Note:
            First checks for explicit list in cfg.scoring.service.enabled_scorers,
            then falls back to all keys under cfg.scorer.*, and finally defaults
            to ["hrm", "sicql"] if nothing is configured.
        """
        names = []
        try:
            # use /scoring/service if available
            scoring_cfg = self.cfg.get("scoring", {}).get("service", {})
            explicit = scoring_cfg.get("enabled_scorers")
            if explicit:
                return list(explicit)
            scorer_block = self.cfg.get("scorer", {})
            # use the section names under cfg.scorer.*
            names = list(scorer_block)
        except Exception as e:
            if self.logger:
                self.logger.log("ScoringServiceResolveNamesError", {"error": str(e)})

        # sane default if nothing is configured
        if not names:
            names = ["hrm", "sicql"]
        return names

    def register_from_cfg(self, names: List[str]) -> None:
        """
        Register scorers from configuration for the specified names.
        
        Args:
            names: List of scorer names to register
            
        Note:
            Skips already registered scorers and logs any errors during registration.
        """
        for name in names:
            if name in self._scorers:
                continue
            scorer_cfg = self.cfg.scorer[name]
            scorer = self._build_scorer(name, scorer_cfg)
            if scorer:
                self.register_scorer(name, scorer)

    def _build_scorer(self, name: str, scorer_cfg: Dict[str, Any]):
        """
        Build a scorer instance for the specified name and configuration.

        Args:
            name: Name of the scorer to build
            scorer_cfg: Configuration for the scorer
            
        Returns:
            Scorer instance or None if building fails
            
        Note:
            Maps scorer names to their respective implementation classes.
            Supports: hrm, sicql, svm, mrq, ebt, contrastive_ranker/contrastive/reward
        """
        try:
            if name == "hrm":
                from stephanie.scoring.scorer.hrm_scorer import HRMScorer
                return HRMScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "epistemic_hrm": 
                from stephanie.scoring.scorer.ep_hrm_scorer import \
                    EpistemicPlanHRMScorer
                return EpistemicPlanHRMScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "sicql":
                from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
                return SICQLScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "svm":
                from stephanie.scoring.scorer.svm_scorer import SVMScorer
                return SVMScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "mrq":
                from stephanie.scoring.scorer.mrq_scorer import MRQScorer
                return MRQScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "ebt":
                from stephanie.scoring.scorer.ebt_scorer import EBTScorer
                return EBTScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "tiny":
                from stephanie.scoring.scorer.tiny_scorer import TinyScorer
                return TinyScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name == "knowledge":
                from stephanie.scoring.scorer.knowledge_scorer import \
                    KnowledgeScorer
                return KnowledgeScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
            if name in ("contrastive_ranker", "contrastive", "reward"):
                # We allow "reward" to be an alias for contrastive pairwise scorer.
                from stephanie.scoring.scorer.contrastive_ranker_scorer import \
                    ContrastiveRankerScorer
                return ContrastiveRankerScorer(scorer_cfg, memory=self.memory, container=self.container, logger=self.logger)
        except Exception as e:
            if self.logger:
                self.logger.log("ScoringServiceBuildScorerError", {"name": name, "error": str(e)})
        return None

    def register_scorer(self, name: str, scorer: Any) -> None:
        """
        Register a scorer instance with the service.
        
        Args:
            name: Name to register the scorer under
            scorer: Scorer instance to register
        """
        self._scorers[name] = scorer
        if self.logger:
            self.logger.log("ScoringServiceScorerRegistered", {"name": name})

    def _log_init(self):
        """Log initialization details for the scoring service."""
        if self.logger:
            self.logger.log("ScoringServiceInitialized", {
                "enabled": self.enabled_scorer_names,
                "registered": list(self._scorers.keys())
            })

    # ------------------------------------------------------------------ #
    # Compute paths (ScoreBundle) + persistence
    # ------------------------------------------------------------------ #
    def score(
        self,
        scorer_name: str,
        scorable: Scorable,
        context: Dict[str, Any],
        dimensions: Optional[List[str]] = None,
    ) -> ScoreBundle:
        """
        Call the underlying scorer and return its ScoreBundle.
        
        Args:
            scorer_name: Name of the registered scorer to use
            scorable: The scorable object to score
            context: Context dictionary containing goal and other information
            dimensions: Optional list of dimensions to score against
            
        Returns:
            ScoreBundle containing scoring results
            
        Raises:
            ValueError: If the specified scorer is not registered
            
        Note:
            Does not persist automatically. Extracts goal from context and passes
            it to the scorer for goal-conditioned scoring.
        """
        scorer = self._scorers.get(scorer_name)
        if not scorer:
            raise ValueError(f"Scorer '{scorer_name}' not registered")

        return scorer.score(context, scorable, dimensions)

    def score_and_persist(
        self,
        scorer_name: str,
        scorable: Scorable,
        context: Dict[str, Any],
        *,
        dimensions: Optional[List[str]] = None,
        source: Optional[str] = None,
        evaluator: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ScoreBundle:
        """
        Compute and persist scores using EvaluationStore.save_bundle.
        
        Args:
            scorer_name: Name of the registered scorer to use
            scorable: The scorable object to score
            context: Context dictionary containing goal and other information
            dimensions: Optional list of dimensions to score against
            source: Optional source identifier for the evaluation
            evaluator: Optional evaluator identifier
            model_name: Optional model name identifier
            
        Returns:
            ScoreBundle containing scoring results
            
        Note:
            Persists scores using EvaluationStore.save_bundle to ensure consistent
            storage of ScoreORM and EvalAttributes.
        """
        bundle = self.score(scorer_name, scorable, context, dimensions)
        try:
            self.memory.evaluations.save_bundle(
                bundle=bundle,
                scorable=scorable,
                context=context,
                cfg=self.cfg,
                source=source or scorer_name,
                embedding_type=self.embedding_type,
                evaluator=evaluator or scorer_name,
                model_name=model_name or (self._cfg_get("model", "name", default="unknown")),
                agent_name=self.name,
            )
        except Exception as e: 
            if self.logger:
                self.logger.log("ScoringServicePersistError", {
                    "scorer": scorer_name,
                    "scorable_id": scorable.id,
                    "scorable_type": scorable.target_type,
                    "error": str(e)
                })
        return bundle

    def save_bundle(
        self,
        scorer_name: str,
        bundle: ScoreBundle,
        scorable: Scorable,
        context: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        source: Optional[str] = None,
        evaluator: Optional[str] = None,
        model_name: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> ScoreBundle:
        """
        Compute and persist scores using EvaluationStore.save_bundle.
        
        Args:
            scorer_name: Name of the registered scorer to use
            scorable: The scorable object to score
            context: Context dictionary containing goal and other information
            dimensions: Optional list of dimensions to score against
            source: Optional source identifier for the evaluation
            evaluator: Optional evaluator identifier
            model_name: Optional model name identifier
            
        Returns:
            ScoreBundle containing scoring results
            
        Note:
            Persists scores using EvaluationStore.save_bundle to ensure consistent
            storage of ScoreORM and EvalAttributes.
        """
        try:
            self.memory.evaluations.save_bundle(
                bundle=bundle,
                scorable=scorable,
                context=context,
                cfg=cfg,
                source=source or scorer_name,
                embedding_type=self.embedding_type,
                evaluator=evaluator or scorer_name,
                model_name=model_name or (cfg.get("model", "name", default="unknown")),
                agent_name=agent_name or self.name,
            )
        except Exception as e: 
            if self.logger:
                self.logger.log("ScoringServicePersistError", {
                    "scorer": scorer_name,
                    "scorable_id": scorable.id,
                    "scorable_type": scorable.target_type,
                    "error": str(e)
                })
        return bundle

    # ------------------------------------------------------------------ #
    # Generic single-dimension I/O (passthrough to ScoreStore)
    # ------------------------------------------------------------------ #
    def save_score(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        score_type: str,
        score_value: float,
        weight: float = 1.0,
        rationale: Optional[str] = None,
        source: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        **evaluation_kwargs,   # goal_id, plan_trace_id, pipeline_run_id, agent_name, etc.
    ):
        """
        Persist a single dimension score by delegating to ScoreStore.save_score.
        
        Args:
            scorable_id: Identifier of the scorable object
            scorable_type: Type of the scorable object
            score_type: Type of score being saved
            score_value: Numeric value of the score
            weight: Weight of the score (default: 1.0)
            rationale: Optional rationale for the score
            source: Optional source identifier
            attributes: Optional additional attributes
            **evaluation_kwargs: Additional evaluation context parameters
            
        Returns:
            Result of the ScoreStore.save_score operation
        """
        return self.memory.scores.save_score(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            score_type=score_type,
            score_value=score_value,
            weight=weight,
            rationale=rationale,
            source=source,
            attributes=attributes,
            **evaluation_kwargs,
        )

    def get_score(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        score_type: str,
        normalize: bool = False,
        attribute_sources: tuple[str, ...] = ("hrm", "mars", "sicql"),
        attribute_field: Optional[str] = None,
        dimension_filter: Optional[str] = None,
        fallback_to_scores_json: bool = False,  # kept for legacy compat (no-op if you removed JSON)
    ) -> Optional[float]:
        """
        Retrieve a score by delegating to ScoreStore.get_score.
        
        Args:
            scorable_id: Identifier of the scorable object
            scorable_type: Type of the scorable object
            score_type: Type of score to retrieve
            normalize: Whether to normalize the score
            attribute_sources: Tuple of attribute sources to check
            attribute_field: Specific attribute field to retrieve
            dimension_filter: Optional dimension filter
            fallback_to_scores_json: Whether to fall back to JSON scores (legacy)
            
        Returns:
            The retrieved score value or None if not found
        """
        return self.memory.scores.get_score(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            score_type=score_type,
            normalize=normalize,
            attribute_sources=attribute_sources,
            attribute_field=attribute_field,
            dimension_filter=dimension_filter,
            fallback_to_scores_json=fallback_to_scores_json,
        )

    # ------------------------------------------------------------------ #
    # Canonical HRM helpers (dimension row in scores)
    # ------------------------------------------------------------------ #
    def save_hrm_score(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        value: float,
        **evaluation_kwargs,
    ):
        """
        Store HRM score canonically as a dimension row.
        
        Args:
            scorable_id: Identifier of the scorable object
            scorable_type: Type of the scorable object
            value: HRM score value (0-1 normalized)
            **evaluation_kwargs: Additional evaluation context parameters
            
        Returns:
            Result of the HRM score save operation
        """
        return self.memory.scores.save_hrm_score(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            value=value,
            **evaluation_kwargs,
        )

    def get_hrm_score(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        normalize: bool = True,
        compute_if_missing: bool = False,
        compute_context: Optional[Dict[str, Any]] = None,
        scorable_builder: Optional[Callable[[Dict[str, Any]], Scorable]] = None,
        scorer_name: str = "hrm",
        dimensions: Optional[List[str]] = None,
    ) -> Optional[float]:
        """
        Retrieve HRM score with optional computation if missing.
        
        Args:
            scorable_id: Identifier of the scorable object
            scorable_type: Type of the scorable object
            normalize: Whether to normalize the score (default: True)
            compute_if_missing: Whether to compute the score if not found
            compute_context: Context to use for computation if needed
            scorable_builder: Function to build scorable from context
            scorer_name: Name of the scorer to use for computation
            dimensions: Dimensions to use for computation
            
        Returns:
            HRM score value (0-1 normalized) or None if not found/computable
        """
        val = self.memory.scores.get_hrm_score(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            normalize=normalize,
        )
        if val is not None or not compute_if_missing:
            return val

        # Compute-if-missing
        scorer = self._scorers.get(scorer_name)
        if not scorer:
            return None
        ctx = compute_context or {}
        scorable = scorable_builder(ctx) if scorable_builder else ScorableFactory.from_context(ctx)
        if not scorable:
            return None

        bundle = scorer.score(ctx, scorable, dimensions or ["alignment"])
        hrm = float(bundle.aggregate())

        # persist canonical HRM
        self.save_hrm_score(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            value=hrm,
            **{k: ctx.get(k) for k in ("goal_id", "plan_trace_id", "pipeline_run_id") if k in ctx},
        )
        return hrm

    # ------------------------------------------------------------------ #
    # Model readiness / lifecycle
    # ------------------------------------------------------------------ #
    def get_model_status(self, name: str) -> dict:
        """
        Get the status of a scorer model.
        
        Args:
            name: Name of the scorer to check
            
        Returns:
            Dictionary with model status information including:
            - name: Scorer name
            - registered: Whether the scorer is registered
            - ready: Whether the model is ready for use
            - info: Additional model information if available
        """
        s = self._scorers.get(name)
        if not s:
            return {"name": name, "registered": False, "ready": False, "info": None}
        has_model = getattr(s, "has_model", None)
        get_info = getattr(s, "get_model_info", None)
        ready = bool(has_model and has_model())
        info = get_info() if callable(get_info) else None
        return {"name": name, "registered": True, "ready": ready, "info": info}

    def ensure_ready(self, required: list[str], auto_train: bool = False, fail_on_missing: bool = False) -> dict:
        """
        Ensure required scorers are ready, with optional auto-training.
        
        Args:
            required: List of scorer names that must be ready
            auto_train: Whether to automatically train missing models
            fail_on_missing: Whether to raise an exception if models can't be made ready
            
        Returns:
            Dictionary with readiness report for each required scorer
            
        Raises:
            RuntimeError: If fail_on_missing is True and a required scorer is not ready
        """
        report = {}
        for name in required or []:
            st = self.get_model_status(name)
            if not st["registered"]:
                self.logger and self.logger.log("ScoringModelMissing", {"scorer": name, "reason": "not_registered"})
                st["error"] = "not_registered"
            if not st["ready"]:
                if auto_train:
                    trainer = getattr(self._scorers.get(name), "train", None)
                    if callable(trainer):
                        try:
                            trainer()  # allow scorer to load/fit its model
                            st = self.get_model_status(name)
                        except Exception as e:
                            st["error"] = f"train_failed: {e}"
                            self.logger and self.logger.log("ScoringModelTrainFailed", {"scorer": name, "error": str(e)})
                    else:
                        st["error"] = "no_train_method"
                        self.logger and self.logger.log("ScoringModelNoTrain", {"scorer": name})
                if fail_on_missing and not st["ready"]:
                    raise RuntimeError(f"Required scorer '{name}' not ready: {st}")
            report[name] = st
        self.logger and self.logger.log("ScoringEnsureReadyReport", report)
        return report

    # ------------------------------------------------------------------ #
    # Pairwise reward convenience
    # ------------------------------------------------------------------ #
    def _coerce_scorable(self, x, default_type: str = "document"):
        """
        Convert input to a Scorable object if it isn't already.
        
        Args:
            x: Input to convert (string or Scorable)
            default_type: Default type to use if input is a string
            
        Returns:
            Scorable object representation of the input
        """
        if isinstance(x, Scorable):
            return x
        return Scorable(id=None, text=str(x), target_type=default_type)

    def compare_pair(
        self,
        *,
        scorer_name: str,
        context: dict,
        a,
        b,
        dimensions: list[str] | None = None,
        margin: float | None = None,
    ) -> dict:
        """
        Compare two items using the specified scorer.
        
        Args:
            scorer_name: Name of the scorer to use for comparison
            context: Context dictionary containing goal information
            a: First item to compare (string or Scorable)
            b: Second item to compare (string or Scorable)
            dimensions: Optional list of dimensions to compare on
            margin: Optional margin for tie-breaking
            
        Returns:
            Dictionary with comparison results including:
            - winner: Which item won ("a" or "b")
            - score_a: Score for item a
            - score_b: Score for item b
            - mode: Comparison mode used
            - per_dimension: Per-dimension scores if available
            
        Note:
            Uses scorer's native compare method if available, otherwise falls back
            to aggregate scoring. Applies goal similarity tie-breaking if margin is provided.
        """
        scorer = self._scorers.get(scorer_name)
        if not scorer:
            raise ValueError(f"No scorer registered under '{scorer_name}'")

        a_s = self._coerce_scorable(a)
        b_s = self._coerce_scorable(b)

        # If scorer exposes native pairwise compare, use it.
        if hasattr(scorer, "compare"):
            res = scorer.compare(context=context, a=a_s, b=b_s, dimensions=dimensions)
        else:
            # Fallback: score each against the goal and prefer higher aggregate
            dims = dimensions or getattr(scorer, "dimensions", []) or ["alignment"]
            bundle_a = scorer.score(context=context, scorable=a_s, dimensions=dims)
            bundle_b = scorer.score(context=context, scorable=b_s, dimensions=dims)

            sa = sum(sr.score for sr in bundle_a.results.values()) / max(1, len(bundle_a.results))
            sb = sum(sr.score for sr in bundle_b.results.values()) / max(1, len(bundle_b.results))
            res = {
                I: "a" if sa >= sb else "b",
                "score_a": float(sa),
                "score_b": float(sb),
                "mode": "aggregate_fallback",
                "scorer": scorer_name,
                "per_dimension": [
                    {"dimension": d, "score_a": float(bundle_a.results[d].score), "score_b": float(bundle_b.results[d].score)}
                    for d in bundle_a.results.keys()
                ],
            }

        # Optional margin-based tie-break (goal similarity)
        if margin is not None and abs(res["score_a"] - res["score_b"]) < float(margin):
            gtxt = (context.get("goal") or {}).get("goal_text", "") or ""
            try:
                gvec = np.asarray(self.memory.embedding.get_or_create(gtxt), dtype=float).reshape(1, -1)

                def _sim(x):
                    s = self._coerce_scorable(x)
                    v = np.asarray(self.memory.embedding.get_or_create(s.text), dtype=float).reshape(1, -1)
                    return float(cosine_similarity(gvec, v)[0, 0])

                res["winner"] = "a" if _sim(a_s) >= _sim(b_s) else "b"
                res["tie_break"] = "goal_similarity"
            except Exception:
                pass  # leave result as-is if embeddings/tie-break fails

        self.logger and self.logger.log("ScoringServicePairwise", {
            "scorer": scorer_name,
            "winner": res.get("winner"),
            "score_a": res.get("score_a"),
            "score_b": res.get("score_b"),
            "mode": res.get("mode"),
        })
        return res

    def reward_compare(
        self,
        *,
        context: dict,
        a,
        b,
        dimensions: list[str] | None = None,
        margin: float | None = None,
    ) -> dict:
        """
        Compare two items using the 'reward' scorer.
        
        Args:
            context: Context dictionary containing goal information
            a: First item to compare (string or Scorable)
            b: Second item to compare (string or Scorable)
            dimensions: Optional list of dimensions to compare on
            margin: Optional margin for tie-breaking
            
        Returns:
            Dictionary with comparison results (see compare_pair for format)
            
        Note:
            Uses configuration defaults for dimensions and margin if not provided.
        """
        # allow defaulting from cfg if not provided
        if dimensions is None:
            dims_cfg = self._cfg_get("scorer", "reward", "dimensions", default=None)
            dimensions = list(dims_cfg) if isinstance(dims_cfg, (list, tuple)) else None
        if margin is None:
            margin = float(self._cfg_get("scorer", "reward", "margin", default=0.05) or 0.05)

        return self.compare_pair(
            scorer_name="reward",
            context=context,
            a=a,
            b=b,
            dimensions=dimensions,
            margin=margin,
        )

    def reward_decide(self, context: dict, a, b) -> str:
        """
        Minimal API that returns just the winner of a comparison.
        
        Args:
            context: Context dictionary containing goal information
            a: First item to compare (string or Scorable)
            b: Second item to compare (string or Scorable)
            
        Returns:
            "a" or "b" indicating which item won
            
        Note:
            Uses a deterministic fallback (length comparison) if comparison fails.
        """
        try:
            res = self.reward_compare(context=context, a=a, b=b)
            return "a" if res.get("winner") == "a" else "b"
        except Exception as e:
            # Deterministic last-resort fallback
            self.logger and self.logger.log("ScoringServiceRewardFallback", {"error": str(e)})
            la = len(getattr(a, "text", str(a)))
            lb = len(getattr(b, "text", str(b)))
            return "a" if la >= lb else "b"

    def _cos1d(self, a, b) -> float:
        """
        Compute cosine similarity between two 1D vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity value between the vectors
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if np.isclose(na, 0.0) or np.isclose(nb, 0.0):
            return 0.0
        return float(np.dot(a, b) / (na * nb))