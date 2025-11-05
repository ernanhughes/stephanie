# stephanie/agents/prompt_evolver_agent.py — Single-file prompt evolver agent (cleaned)
# ---------------------------------------------------------------------------------
# What this file provides
# - PromptTuningSignature (DSPy)
# - EvaluatorAdapter (normalizes evaluator APIs)
# - StrategyMutationPass (optional symbolic mutations)
# - PromptEvolver (DSPy tuning + optional mutations)
# - PromptEvolverAgent (BaseAgent wrapper with telemetry + optional ZeroModel timeline)
#
# Inputs (context)
#   context["examples"]: List[{
#       "goal": str,
#       "prompt_text": str,
#       "hypothesis_text": str,
#       "review": Optional[str],
#       "elo_rating": Optional[int|float],
#   }]
#   context["goal"]["goal_text"]: Optional[str]
#   Optional: "evaluator_cfg", "sample_size", "use_strategy_mutation", "include_original"
#
# Outputs (context)
#   "refined_prompts": List[str]
#   "prompt_candidates": List[{"prompt", "score", "source", "rank"}]
#   "best_prompt": str
#   "best_prompt_score": Optional[float]
#   "best_prompt_source": Optional[str]
#   "timeline_path": Optional[str]
#   "prompt_evolver_stats": Dict[str, Any]
# ---------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dspy import BootstrapFewShot, Example, Predict
from dspy import InputField, OutputField, Signature

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.evaluator.evaluator_loader import get_evaluator


_logger = logging.getLogger(__name__)


# =============================
# DSPy signature
# =============================
class PromptTuningSignature(Signature):
    goal = InputField(desc="Scientific research goal or question")
    input_prompt = InputField(desc="Original prompt used to generate hypotheses")
    hypotheses = InputField(desc="Best hypothesis generated")
    review = InputField(desc="Expert review of the hypothesis")
    score = InputField(desc="Numeric score evaluating the hypothesis quality")
    refined_prompt = OutputField(desc="Improved version of the original prompt")


# =============================
# Evaluator adapter (resilient over heterogeneous APIs)
# =============================
class EvaluatorAdapter:
    """Adapter that supports either .score_single(...) or .evaluate(...).

    Expected evaluator methods (best effort):
      - score_single(prompt, reference_output=?, goal=?, context=?) -> float
      - evaluate(obj, context=?) -> object with .score or {"score": x}
    """

    def __init__(self, evaluator, logger=None):
        self.evaluator = evaluator
        self.logger = logger

    def _extract_score(self, obj: Any) -> Optional[float]:
        try:
            if obj is None:
                return None
            if isinstance(obj, (int, float)):
                return float(obj)
            if hasattr(obj, "score"):
                return float(getattr(obj, "score"))
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
        return None

    def score_prompt_only(self, prompt: str, goal: str = "", context: Dict[str, Any] | None = None) -> Optional[float]:
        if self.evaluator is None:
            return None
        try:
            if hasattr(self.evaluator, "score_single"):
                return float(self.evaluator.score_single(prompt, goal=goal, context=context or {}))
            res = self.evaluator.evaluate({"prompt": prompt, "goal": goal}, context=context or {})
            return self._extract_score(res)
        except Exception as e:
            if self.logger:
                self.logger.log("EvaluatorScorePromptOnlyFailed", {"error": str(e)})
            return None

    def score_against_reference(
        self, prompt: str, reference_output: str = "", context: Dict[str, Any] | None = None
    ) -> Optional[float]:
        if self.evaluator is None:
            return None
        try:
            if hasattr(self.evaluator, "score_single"):
                return float(
                    self.evaluator.score_single(prompt, reference_output=reference_output, context=context or {})
                )
            res = self.evaluator.evaluate(
                {"prompt": prompt, "reference_output": reference_output}, context=context or {}
            )
            return self._extract_score(res)
        except Exception as e:
            if self.logger:
                self.logger.log("EvaluatorScoreAgainstRefFailed", {"error": str(e)})
            return None


# =============================
# Strategy mutation (optional symbolic edits)
# =============================
class StrategyMutationPass:
    def __init__(self, score_fn, logger=None):
        """score_fn: Callable[[prompt: str, metadata: dict], float|None]"""
        self.score_fn = score_fn
        self.logger = logger

    def apply(self, base_prompt: str, metadata: dict) -> list[dict]:
        """Generate and score simple symbolic mutations.
        Returns: [{"prompt": str, "score": float|None}]
        """
        mutations: List[Dict[str, Any]] = []
        candidates = [
            base_prompt.replace("Let's think step by step.", "Let's work through this carefully."),
            base_prompt + "\nProvide a rationale before giving your final answer.",
            base_prompt.replace("explain", "analyze"),
        ]
        for variant in candidates:
            try:
                score = self.score_fn(variant, metadata) if self.score_fn else None
            except Exception as e:
                score = None
                if self.logger:
                    self.logger.log(
                        "StrategyMutationEvalError", {"error": str(e), "prompt_snippet": variant[:100]}
                    )
            mutations.append({"prompt": variant, "score": score})

        # Highest score first; None sorted last
        return sorted(mutations, key=lambda x: (-1e9 if x["score"] is None else -float(x["score"])) )


# =============================
# PromptEvolver core (DSPy + optional strategy mutation)
# =============================
class PromptEvolver:
    def __init__(self, llm, logger=None, use_strategy_mutation: bool = False, evaluator_cfg=None, memory=None):
        self.llm = llm
        self.logger = logger
        self.use_strategy_mutation = bool(use_strategy_mutation)
        try:
            dspy.configure(lm=self.llm)
        except Exception as e:
            if self.logger:
                self.logger.log("DSPyConfigureFailed", {"error": str(e)})

        # evaluator (optional) + adapter
        self.evaluator = None
        if evaluator_cfg:
            self.evaluator = get_evaluator(evaluator_cfg, memory=memory, llm=llm, logger=logger)
        self.adapter = EvaluatorAdapter(self.evaluator, logger=self.logger)

        # strategy pass wiring
        self.strategy_pass: Optional[StrategyMutationPass] = None
        if self.use_strategy_mutation:
            def _score_cb(prompt: str, metadata: dict) -> Optional[float]:
                goal = metadata.get("goal", "") if isinstance(metadata, dict) else ""
                ref = metadata.get("hypotheses", "") if isinstance(metadata, dict) else ""
                score = None
                if goal:
                    score = self.adapter.score_prompt_only(prompt, goal=goal, context=metadata)
                if score is None and ref:
                    score = self.adapter.score_against_reference(prompt, reference_output=ref, context=metadata)
                return score
            self.strategy_pass = StrategyMutationPass(score_fn=_score_cb, logger=self.logger)

        # provenance for last evolve() call
        self.sources_map: Dict[str, str] = {}

    def evolve(self, examples: list[dict], context: dict | None = None, sample_size: int = 10) -> list[str]:
        """DSPy tuning + optional strategy mutation → refined prompts.
        Returns ordered unique list of strings.
        """
        context = context or {}
        self.sources_map = {}
        if not examples:
            return []

        # Build DSPy trainset from the first sample_size examples
        train_items = examples[: max(1, int(sample_size))]
        training_set = [
            Example(
                goal=ex.get("goal", ""),
                input_prompt=ex.get("prompt_text", ""),
                hypotheses=ex.get("hypothesis_text", ""),
                review=ex.get("review", ""),
                score=ex.get("elo_rating", 1000),
            ).with_inputs("goal", "input_prompt", "hypotheses", "review", "score")
            for ex in train_items
        ]

        def eval_metric(example: Example, pred, trace=None):
            """If evaluator present, score refined prompt vs ref/goal; else return 1.0."""
            try:
                refined = (getattr(pred, "refined_prompt", "") or "").strip()
                if not refined or self.adapter is None:
                    return 1.0
                ref = getattr(example, "hypotheses", "") or ""
                goal = getattr(example, "goal", "") or ""
                score = None
                if ref:
                    score = self.adapter.score_against_reference(refined, reference_output=ref, context=context)
                if score is None and goal:
                    score = self.adapter.score_prompt_only(refined, goal=goal, context=context)
                return float(score) if isinstance(score, (int, float)) else 1.0
            except Exception as e:
                if self.logger:
                    self.logger.log("DSPyMetricFailed", {"error": str(e)})
                return 1.0

        tuner = BootstrapFewShot(metric=eval_metric)
        student = Predict(PromptTuningSignature)
        tuned_program = tuner.compile(student=student, trainset=training_set)

        refined_prompts: List[str] = []

        # Generate refined prompts for ALL examples
        for ex in examples:
            try:
                result = tuned_program(
                    goal=ex.get("goal", ""),
                    input_prompt=ex.get("prompt_text", ""),
                    hypotheses=ex.get("hypothesis_text", ""),
                    review=ex.get("review", ""),
                    score=ex.get("elo_rating", 1000),
                )
                refined = (getattr(result, "refined_prompt", "") or "").strip()
                if refined:
                    refined_prompts.append(refined)
                    self.sources_map[refined] = "dspy"
            except Exception as e:
                if self.logger:
                    self.logger.log("DSPyPromptEvolutionFailed", {"error": str(e)})

        # Optional symbolic strategy mutations
        if self.use_strategy_mutation and self.strategy_pass is not None:
            for ex in examples:
                base_prompt = ex.get("prompt_text", "")
                if not base_prompt:
                    continue
                metadata = {
                    "goal": ex.get("goal", ""),
                    "hypotheses": ex.get("hypothesis_text", ""),
                    "review": ex.get("review", ""),
                    "score": ex.get("elo_rating", 1000),
                }
                try:
                    mutations = self.strategy_pass.apply(base_prompt, metadata)
                    for mut in mutations:
                        prompt_text = (mut.get("prompt", "") or "").strip()
                        if prompt_text:
                            refined_prompts.append(prompt_text)
                            self.sources_map[prompt_text] = "strategy"
                except Exception as e:
                    if self.logger:
                        self.logger.log("StrategyMutationFailed", {"error": str(e)})

        # De-duplicate preserving order
        seen = set()
        refined_prompts = [p for p in refined_prompts if p and not (p in seen or seen.add(p))]
        return refined_prompts

    # Convenience wrappers
    def score_prompt(self, prompt: str, reference_output: str = "", context: dict | None = None) -> float:
        s = self.adapter.score_against_reference(prompt, reference_output, context or {})
        return float(s) if s is not None else -1.0

    def score_prompt_only(self, prompt: str, goal: str = "", context: dict | None = None) -> float:
        s = self.adapter.score_prompt_only(prompt, goal, context or {})
        return float(s) if s is not None else -1.0


# =============================
# Agent wrapper (BaseAgent)
# =============================
class PromptEvolverAgent(BaseAgent):
    """Single-file, production-ready Prompt Evolver agent.

    - DSPy refinement + optional strategy mutations
    - Optional evaluator-backed scoring
    - Optional ZeroModel timeline + bus telemetry
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.sample_size: int = max(1, int(cfg.get("sample_size", 10)))
        self.use_strategy_mutation: bool = bool(cfg.get("use_strategy_mutation", False))
        self.include_original: bool = bool(cfg.get("include_original", False))
        self.timeout: float = float(cfg.get("timeout", 300.0))

        subjects = (cfg.get("subjects") or {})
        self.event_subject: str = subjects.get("events", "prompt.evolver.events")

        timeline_cfg = (cfg.get("timeline") or {})
        self.timeline_enabled: bool = bool(timeline_cfg.get("enabled", True))

        # Services (optional)
        self.llm = self._safe_get_service("llm")
        self.zm = self._safe_get_service("zeromodel")

        # Default evaluator config (if none provided per-run)
        self.default_evaluator_cfg: Optional[Dict[str, Any]] = cfg.get("evaluator")

        self.run_id: str = ""
        self.start_ts: float = 0.0

    def _safe_get_service(self, key: str):
        try:
            return self.container.get(key)
        except Exception as e:
            self._log_warn("ServiceNotFound", {"service": key, "error": str(e)})
            return None

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.start_ts = time.time()
        self.run_id = context.get(PIPELINE_RUN_ID) or uuid.uuid4().hex
        context[PIPELINE_RUN_ID] = self.run_id

        examples: List[Dict[str, Any]] = list(context.get("examples") or [])
        if not examples:
            raise ValueError("PromptEvolverAgent: context['examples'] is required and non-empty.")

        goal_text: str = ((context.get("goal") or {}).get("goal_text") or "").strip()
        sample_size: int = max(1, int(context.get("sample_size", self.sample_size)))
        use_strategy_mutation: bool = bool(context.get("use_strategy_mutation", self.use_strategy_mutation))
        include_original: bool = bool(context.get("include_original", self.include_original))
        evaluator_cfg: Optional[Dict[str, Any]] = context.get("evaluator_cfg") or self.default_evaluator_cfg

        # Build evolver (lets it instantiate its own evaluator via cfg)
        evolver = PromptEvolver(
            llm=self.llm,
            logger=self.logger,
            use_strategy_mutation=use_strategy_mutation,
            evaluator_cfg=evaluator_cfg,
            memory=self.memory,
        )

        # Timeline open (optional)
        timeline_opened = False
        if self.timeline_enabled and self.zm:
            try:
                self.zm.timeline_open(self.run_id)
                timeline_opened = True
                await self._emit_event("timeline_opened", {"run_id": self.run_id})
            except Exception as e:
                self._log_warn("TimelineOpenFailed", {"error": str(e)})

        # Evolve in executor (non-blocking)
        loop = asyncio.get_running_loop()
        try:
            refined_prompts: List[str] = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: evolver.evolve(examples=examples, context=context, sample_size=sample_size),
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            self._log_err("EvolutionTimedOut", {"timeout": self.timeout})
            refined_prompts = []
        except Exception as e:
            self._log_err("PromptEvolutionFailed", {"error": str(e)})
            refined_prompts = []

        # Dedupe
        seen = set()
        refined_prompts = [p for p in refined_prompts if p.strip() and not (p in seen or seen.add(p))]

        # Optionally include original prompts at the front
        original_prompts = [ex.get("prompt_text", "") for ex in examples if ex.get("prompt_text")]
        if include_original:
            for p in original_prompts:
                if p not in seen:
                    refined_prompts.insert(0, p)
                    seen.add(p)

        # Score candidates (if evaluator present in evolver)
        candidates: List[Dict[str, Any]] = []
        scoring_enabled = evolver.evaluator is not None and len(refined_prompts) > 0
        ref_output = examples[0].get("hypothesis_text", "") if examples else ""

        for idx, prompt in enumerate(refined_prompts):
            try:
                source = evolver.sources_map.get(
                    prompt, "original" if prompt in original_prompts else "dspy"
                )
                score = None
                if scoring_enabled:
                    if goal_text:
                        score = float(evolver.score_prompt_only(prompt, goal=goal_text, context=context))
                    else:
                        score = float(evolver.score_prompt(prompt, reference_output=ref_output, context=context))
                candidates.append({"prompt": prompt, "score": score, "source": source, "rank": idx})
                await self._emit_event(
                    "candidate",
                    {"prompt": prompt[:160], "score": score, "source": source, "rank": idx},
                )

                if timeline_opened:
                    try:
                        self.zm.timeline_add_node(
                            self.run_id,
                            {
                                "title": f"Candidate #{idx+1}",
                                "text": (prompt[:300] + ("..." if len(prompt) > 300 else "")),
                                "metric": score,
                                "tag": source,
                            },
                        )
                    except Exception as te:
                        self._log_warn("TimelineNodeFailed", {"error": str(te)})
            except Exception as e:
                self._log_warn("CandidateBuildFailed", {"error": str(e)})

        # Choose best
        best_prompt, best_score, best_source = self._pick_best_with_source(candidates)
        if best_prompt is None and original_prompts:
            best_prompt, best_score, best_source = original_prompts[0], None, "fallback_original"

        # Finalize timeline
        timeline_path: Optional[str] = None
        if timeline_opened:
            try:
                finalize_res = await self.zm.timeline_finalize(self.run_id)
                timeline_path = (
                    finalize_res.get("gif")
                    or finalize_res.get("output_path")
                    or getattr(self.zm, "last_output_for", lambda _rid: None)(self.run_id)
                )
                await self._emit_event("timeline_finalized", {"path": timeline_path})
            except Exception as e:
                self._log_warn("TimelineFinalizeFailed", {"error": str(e)})

        # Outputs
        stats = {
            "n_examples": len(examples),
            "n_candidates": len(candidates),
            "n_refined": len(refined_prompts),
            "mutations_used": use_strategy_mutation,
            "scoring_enabled": scoring_enabled,
            "elapsed_s": round(time.time() - self.start_ts, 3),
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        context.update(
            {
                "refined_prompts": refined_prompts,
                "prompt_candidates": candidates,
                "best_prompt": best_prompt or "",
                "best_prompt_score": best_score,
                "best_prompt_source": best_source,
                "timeline_path": timeline_path,
                "prompt_evolver_stats": stats,
            }
        )

        await self._emit_event(
            "evolution_complete",
            {
                "best_score": best_score,
                "n_candidates": len(candidates),
                "timeline_path": timeline_path,
                "elapsed_s": stats["elapsed_s"],
            },
        )

        return context

    # ----------------- helpers -----------------
    def _pick_best_with_source(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        if not candidates:
            return None, None, None
        scored = [c for c in candidates if isinstance(c.get("score"), (int, float))]
        if scored:
            best = max(scored, key=lambda c: c["score"])  # type: ignore[arg-type]
            return best["prompt"], float(best["score"]), best.get("source")
        c = candidates[0]
        return c["prompt"], c.get("score"), c.get("source")

    async def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.memory or not getattr(self.memory, "bus", None):
            return
        try:
            await self.memory.bus.publish(
                subject=self.event_subject,
                payload={
                    "event": event,
                    "run_id": self.run_id,
                    "agent": "PromptEvolverAgent",
                    "payload": payload,
                    "ts": time.time(),
                },
            )
        except Exception as e:
            self._log_warn("EmitEventFailed", {"error": str(e), "event": event})

    def _log_warn(self, tag: str, payload: Dict[str, Any]) -> None:
        try:
            if self.logger:
                self.logger.log(tag, payload)
            else:
                _logger.warning("%s: %s", tag, payload)
        except Exception:
            pass

    def _log_err(self, tag: str, payload: Dict[str, Any]) -> None:
        try:
            if self.logger:
                self.logger.log(tag, payload)
            else:
                _logger.error("%s: %s", tag, payload)
        except Exception:
            pass
