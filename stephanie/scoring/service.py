# stephanie/scoring/service.py
from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List

# Core types
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory

# Stores
from stephanie.memory.score_store import ScoreStore

# Scorers (imported lazily in _build_scorer)
# - HRMScorer
# - SICQLScorer
# - SVMScorer
# - MRQScorer
# - EBTScorer
# - ContrastiveRankerScorer

class ScoringService:
    """
    Central scoring gateway.

    - Auto-registers scorers from cfg.scorer.*
    - Uniform read/write APIs by (scorable_id, scorable_type, score_type[, dimension])
    - Optional compute_if_missing using registered scorers
    - Persist via EvaluationStore.save_bundle when computing, or ScoreStore when writing single dims
    """

    def __init__(self, cfg: Dict[str, Any], memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self._scorers: Dict[str, Any] = {}   # name -> scorer instance

        # Which scorers to enable:
        # 1) explicit list in scoring.enabled_scorers
        # 2) else all keys present under cfg.scorer.*
        self.enabled_scorer_names: List[str] = self._resolve_scorer_names(cfg)

        # Auto-register
        self.register_from_cfg(self.enabled_scorer_names)

        self._log_init()

    # ------------------------------------------------------------------ #
    # Initialization / registration
    # ------------------------------------------------------------------ #
    def _resolve_scorer_names(self, cfg: Dict[str, Any]) -> List[str]:
        names = []
        try:
            scoring_cfg = cfg.get("scoring", {})
            explicit = scoring_cfg.get("enabled_scorers")
            if explicit:
                return list(explicit)

            scorer_block = cfg.get("scorer", {})
            if isinstance(scorer_block, dict):
                # use the section names under cfg.scorer.*
                names = [k for k, v in scorer_block.items() if isinstance(v, (dict,))]
        except Exception as e:
            if self.logger:
                self.logger.log("ScoringServiceResolveNamesError", {"error": str(e)})

        # sane default if nothing is configured
        if not names:
            names = ["hrm", "sicql"]
        return names

    def register_from_cfg(self, names: List[str]) -> None:
        for name in names:
            if name in self._scorers:
                continue
            scorer_cfg = self._get_scorer_cfg(name)
            scorer = self._build_scorer(name, scorer_cfg)
            if scorer:
                self.register_scorer(name, scorer)

    def _get_scorer_cfg(self, name: str) -> Dict[str, Any]:
        try:
            block = self.cfg.get("scorer", {}).get(name, {})
            return block if isinstance(block, dict) else {}
        except Exception:
            return {}

    def _build_scorer(self, name: str, scorer_cfg: Dict[str, Any]):
        """
        Map name -> scorer class. Keep this small and explicit.
        """
        try:
            if name == "hrm":
                from stephanie.scoring.hrm_scorer import HRMScorer
                return HRMScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "sicql":
                from stephanie.scoring.sicql_scorer import SICQLScorer
                return SICQLScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "svm":
                from stephanie.scoring.svm_scorer import SVMScorer
                return SVMScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "mrq":
                from stephanie.scoring.mrq_scorer import MRQScorer
                return MRQScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "ebt":
                from stephanie.scoring.ebt_scorer import EBTScorer
                return EBTScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name in ("contrastive_ranker", "contrastive"):
                from stephanie.scoring.contrastive_ranker_scorer import ContrastiveRankerScorer
                return ContrastiveRankerScorer(scorer_cfg, memory=self.memory, logger=self.logger)
        except Exception as e:
            if self.logger:
                self.logger.log("ScoringServiceBuildScorerError", {"name": name, "error": str(e)})
        return None

    def register_scorer(self, name: str, scorer: Any) -> None:
        self._scorers[name] = scorer
        if self.logger:
            self.logger.log("ScoringServiceScorerRegistered", {"name": name})

    def _log_init(self):
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
    ):
        """
        Call the underlying scorer and return its ScoreBundle.
        Does not persist automatically.
        """
        scorer = self._scorers.get(scorer_name)
        if not scorer:
            raise ValueError(f"Scorer '{scorer_name}' not registered")
        return scorer.score(context=context, scorable=scorable, dimensions=dimensions)

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
        embedding_type: Optional[str] = None,
    ):
        """
        Compute and persist using EvaluationStore.save_bundle (so ScoreORM + EvalAttributes
        are written consistently by your existing pipeline).
        """
        bundle = self.score(scorer_name, scorable, context, dimensions)
        try:
            self.memory.evaluations.save_bundle(
                bundle=bundle,
                scorable=scorable,
                context=context,
                cfg=self.cfg,
                source=source or scorer_name,
                embedding_type=embedding_type or getattr(self.memory.embedding, "name", None),
                evaluator=evaluator or scorer_name,
                model_name=model_name or self.cfg.get("model", {}).get("name", "unknown"),
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
        """Store HRM canonically as a `scores` row with dimension='hrm'."""
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
        Read latest HRM (0..1). Optionally compute with the registered HRM scorer if missing.
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

        bundle = scorer.score(context=ctx, scorable=scorable, dimensions=dimensions or ["alignment"])
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
    # SICQL Q helpers (EvaluationAttribute row with source='sicql')
    # ------------------------------------------------------------------ #
    def save_sicql_q(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        q_value: float,
        dimension: str = "alignment",
        **evaluation_kwargs,
    ):
        """
        Store SICQL Q-value in EvaluationAttribute (source='sicql').
        """
        return self.memory.scores.save_sicql_q(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            q_value=q_value,
            dimension=dimension,
            **evaluation_kwargs,
        )

    def get_sicql_q(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        dimension: Optional[str] = None,
    ) -> Optional[float]:
        return self.memory.scores.get_sicql_q(
            scorable_id=scorable_id,
            scorable_type=scorable_type,
            dimension=dimension,
        )


    # -------- Model readiness / lifecycle --------
    def get_model_status(self, name: str) -> dict:
        s = self._scorers.get(name)
        if not s:
            return {"name": name, "registered": False, "ready": False, "info": None}
        has_model = getattr(s, "has_model", None)
        get_info = getattr(s, "get_model_info", None)
        ready = bool(has_model and has_model())
        info = get_info() if callable(get_info) else None
        return {"name": name, "registered": True, "ready": ready, "info": info}

    def ensure_ready(self, required: list[str], auto_train: bool = False, fail_on_missing: bool = False) -> dict:
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

    # -------- Pairwise reward convenience --------
    def compare_pair(
        self,
        scorer_name: str,
        goal_text: str,
        doc_a: str,
        doc_b: str,
        target_type: str = "text",
        dimensions: Optional[list[str]] = None,
    ):
        """Score doc_a and doc_b with the registered 'reward' (or any) scorer."""
        ctx = {"goal": {"goal_text": goal_text}}
        sa = self.score(scorer_name, Scorable(id="A", text=doc_a, target_type=target_type), ctx, dimensions).aggregate()
        sb = self.score(scorer_name, Scorable(id="B", text=doc_b, target_type=target_type), ctx, dimensions).aggregate()
        return ("a" if sa >= sb else "b"), float(sa), float(sb)
