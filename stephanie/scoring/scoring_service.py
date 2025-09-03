# stephanie/scoring/service.py
"""
Scoring Service (central gateway)
=================================

What this is
------------
A single, opinionated entrypoint for all scoring I/O across the system.

- **Registration**: Auto-register scorers from `cfg.scorer.*` (or explicit list in `cfg.scoring.enabled_scorers`).
- **Compute**: Call model scorers uniformly (goal-conditioned or not), get back a ScoreBundle.
- **Persist**: When you want, persist bundles through EvaluationStore so Scores + Attributes are consistently written.
- **Single-dimension I/O**: Canonical helpers for HRM and SICQL Q; generic `save_score`/`get_score` passthroughs too.
- **Pairwise**: Built-in pairwise compare (native if scorer supports it; otherwise aggregate fallback with optional tie-breaks).
- **Readiness**: Introspect scorer model readiness, with optional auto-train hook.

Design notes
------------
- We treat **HRM** as a *dimension* (`scores.dimension = "hrm"`). It’s ubiquitous and normalized to [0,1].
- We treat **SICQL Q** as an **attribute** row (source="sicql", field `q_value`) so we don’t lose its rich diagnostics.
- We DO NOT use `EvaluationORM.scores` JSON anymore (it caused drift & confusion).
- Scorer `score(...)` is assumed to be **goal-conditioned** and expects a `goal=dict` (not a whole `context`).
  The service extracts `goal` from `context` and passes it down correctly.

You get:
- Uniform API: register → score → (optional) persist → query point-values when you need them.
- One place to adjust normalization / fallbacks / tie-break behavior.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Core types
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory

# NOTE: We don’t import ScoreStore here directly; we use `self.memory.scores` (already a ScoreStore).

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
        self.embedding_type = self.memory.embedding.name
        self._scorers: Dict[str, Any] = {}   # name -> scorer instance

        # Which scorers to enable:
        # 1) explicit list in cfg.enabled_scorers
        # 2) else all keys present under cfg.scorer.*
        self.enabled_scorer_names: List[str] = self._resolve_scorer_names()

        # Auto-register
        self.register_from_cfg(self.enabled_scorer_names)

        self._log_init()

    # ------------------------------------------------------------------ #
    # Initialization / registration
    # ------------------------------------------------------------------ #
    def _cfg_get(self, *keys, default=None):
        """Config-safe getter that works for dicts and OmegaConf."""
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
        for name in names:
            if name in self._scorers:
                continue
            scorer_cfg = self.cfg.scorer[name]
            scorer = self._build_scorer(name, scorer_cfg)
            if scorer:
                self.register_scorer(name, scorer)

    def _build_scorer(self, name: str, scorer_cfg: Dict[str, Any]):
        """
        Map name -> scorer class. Keep this small and explicit.
        """
        try:
            if name == "hrm":
                from stephanie.scoring.scorer.hrm_scorer import HRMScorer
                return HRMScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "sicql":
                from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
                return SICQLScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "svm":
                from stephanie.scoring.scorer.svm_scorer import SVMScorer
                return SVMScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "mrq":
                from stephanie.scoring.scorer.mrq_scorer import MRQScorer
                return MRQScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name == "ebt":
                from stephanie.scoring.scorer.ebt_scorer import EBTScorer
                return EBTScorer(scorer_cfg, memory=self.memory, logger=self.logger)
            if name in ("contrastive_ranker", "contrastive", "reward"):
                # We allow "reward" to be an alias for contrastive pairwise scorer.
                from stephanie.scoring.scorer.contrastive_ranker_scorer import \
                    ContrastiveRankerScorer
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

        HOT: Our scorers typically expect **goal-only** (not full context).
             We extract goal and call `scorer.score(goal=..., scorable, dimensions)`.
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
                embedding_type=self.embedding_type,
                evaluator=evaluator or scorer_name,
                model_name=model_name or (self._cfg_get("model", "name", default="unknown")),
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
        """Convenience: persist a single dimension row (delegates to ScoreStore.save_score)."""
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
        """Unified read backed by ScoreStore (ScoreORM → Attribute)."""
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

    # ------------------------------------------------------------------ #
    # Pairwise reward convenience
    # ------------------------------------------------------------------ #
    def _coerce_scorable(self, x, default_type: str = "document"):
        """Accept str|Scorable, return Scorable (text only is fine for pairwise)."""
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
        Generic pairwise compare via registered scorer.
        Returns a rich dict: {winner, score_a, score_b, per_dimension, ...}

        HOT: If the scorer implements `compare(...)`, we use it. Otherwise we
             fall back to “score vs score” aggregation across dimensions.
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
                "winner": "a" if sa >= sb else "b",
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
        Convenience: compare using the 'reward' scorer registration.
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
        Minimal API that returns just 'a' or 'b'.
        (Useful to plug into components that expect a simple callable.)
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
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if np.isclose(na, 0.0) or np.isclose(nb, 0.0):
            return 0.0
        return float(np.dot(a, b) / (na * nb))
