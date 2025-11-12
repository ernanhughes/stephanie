# stephanie/agents/prompt_evolver.py
"""
Prompt Evolution Agent - DSPy-Powered Prompt Optimization System

Module: stephanie.agents.prompt_evolver_agent
Description: Production-ready prompt evolution system combining DSPy-based learning
             with symbolic strategy mutations and evaluator-backed scoring.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PromptEvolverAgent                               │
│  (Orchestrates end-to-end prompt evolution workflow)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │   Evaluator     │  │  Strategy        │  │    DSPy Tuning Engine       │ │
│  │    Adapter      │  │  Mutation Pass   │  │                             │ │
│  │                 │  │                  │  │ • BootstrapFewShot          │ │
│  • Unified scoring │  • Symbolic edits   │  │ • PromptTuningSignature     │ │
│  • Multi-format    │  • Rule-based       │  │ • Example-based learning    │ │
│    support         │    variations       │  └─────────────────────────────┘ │
│  • Fallback        │  • Score-guided     │                                  │
│    strategies      │    selection        │                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Key Components:
1. PromptEvolverAgent - Main agent class implementing BaseAgent interface
2. PromptEvolver - Core evolution engine with DSPy + strategy mutations
3. EvaluatorAdapter - Unified interface for heterogeneous evaluators
4. StrategyMutationPass - Symbolic prompt variation generator
5. PromptTuningSignature - DSPy signature for prompt refinement learning

Workflow:
1. Input: Examples with prompts, hypotheses, reviews, and scores
2. DSPy Phase: Few-shot learning to generate refined prompts
3. Strategy Phase: Symbolic mutations with scoring (optional)
4. Scoring: Evaluator-based ranking of candidate prompts
5. Output: Ranked prompts with provenance tracking

Features:
• Multi-method Evolution: Combines DSPy learning and rule-based mutations
• Resilient Evaluation: Handles multiple evaluator APIs with graceful fallbacks
• Provenance Tracking: Tracks prompt sources (dspy, strategy, original)
• Timeline Integration: Optional ZeroModel visualization support
• Event Telemetry: Real-time progress monitoring via event bus
• Production Ready: Timeouts, error handling, and comprehensive logging

Configuration Options:
• sample_size: Number of examples for DSPy training (default: 10)
• use_strategy_mutation: Enable symbolic mutations (default: False)
• include_original: Include original prompts in output (default: False)
• timeout: Evolution timeout in seconds (default: 300)
• evaluator: Evaluator configuration for scoring

Dependencies:
• dspy: Core prompt tuning and few-shot learning
• asyncio: Async execution and timeout management
• ZeroModel: Optional timeline visualization
• Stephanie framework: BaseAgent, memory, container services

Usage Example:
```python
agent = PromptEvolverAgent(cfg, memory, container, logger)
result = await agent.run({
    "examples": training_data,
    "goal": {"goal_text": "Discover new materials"},
    "evaluator_cfg": {"type": "quality_scorer"}
})
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dspy import (BootstrapFewShot, Example, InputField, OutputField, Predict,
                  Signature)

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.evaluator.evaluator_loader import get_evaluator

log = logging.getLogger(__name__)

# =============================
# DSPy signature
# =============================
class PromptTuningSignature(Signature):
    """
    DSPy signature for prompt refinement. Defines the input-output structure
    for the prompt tuning process where the model learns to improve prompts
    based on examples and feedback.
    """
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
    """
    Adapter that provides a unified interface for different evaluator implementations.
    Supports both .score_single() and .evaluate() methods with fallback strategies.
    """

    def __init__(self, evaluator, logger=None):
        self.evaluator = evaluator
        self.logger = logger
        log.debug(f"EvaluatorAdapter initialized with evaluator: {type(evaluator).__name__}")

    def _extract_score(self, obj: Any) -> Optional[float]:
        """
        Extract score from various evaluator response formats.
        Supports: direct numeric, objects with .score attribute, dict with 'score' key.
        """
        try:
            if obj is None:
                return None
            if isinstance(obj, (int, float)):
                return float(obj)
            if hasattr(obj, "score"):
                return float(getattr(obj, "score"))
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception as e:
            log.warning(f"Failed to extract score from evaluator response: {e}, type: {type(obj)}")
        return None

    def score_prompt_only(self, prompt: str, goal: str = "", context: Dict[str, Any] | None = None) -> Optional[float]:
        """Score a prompt against a goal without reference output."""
        log.debug(f"Scoring prompt-only (goal: {goal[:50]}..., prompt length: {len(prompt)})")
        if self.evaluator is None:
            log.warning("No evaluator available for scoring")
            return None
        try:
            if hasattr(self.evaluator, "score_single"):
                score = float(self.evaluator.score_single(prompt, goal=goal, context=context or {}))
                log.debug(f"score_single method returned: {score}")
                return score
            res = self.evaluator.evaluate({"prompt": prompt, "goal": goal}, context=context or {})
            score = self._extract_score(res)
            log.debug(f"evaluate method returned score: {score}")
            return score
        except Exception as e:
            log.error(f"Evaluator scoring failed: {e}")
            if self.logger:
                self.logger.log("EvaluatorScorePromptOnlyFailed", {"error": str(e)})
            return None

    def score_against_reference(
        self, prompt: str, reference_output: str = "", context: Dict[str, Any] | None = None
    ) -> Optional[float]:
        """Score a prompt against a reference output."""
        log.debug(f"Scoring against reference (ref length: {len(reference_output)}, prompt length: {len(prompt)})")
        if self.evaluator is None:
            log.warning("No evaluator available for reference scoring")
            return None
        try:
            if hasattr(self.evaluator, "score_single"):
                score = float(
                    self.evaluator.score_single(prompt, reference_output=reference_output, context=context or {})
                )
                log.debug(f"score_single with reference returned: {score}")
                return score
            res = self.evaluator.evaluate(
                {"prompt": prompt, "reference_output": reference_output}, context=context or {}
            )
            score = self._extract_score(res)
            log.debug(f"evaluate with reference returned score: {score}")
            return score
        except Exception as e:
            log.error(f"Reference scoring failed: {e}")
            if self.logger:
                self.logger.log("EvaluatorScoreAgainstRefFailed", {"error": str(e)})
            return None


# =============================
# Strategy mutation (optional symbolic edits)
# =============================
class StrategyMutationPass:
    """
    Generates and scores symbolic mutations of base prompts.
    Provides simple rule-based variations that can be evaluated quickly.
    """

    def __init__(self, score_fn, logger=None):
        """score_fn: Callable[[prompt: str, metadata: dict], float|None]"""
        self.score_fn = score_fn
        self.logger = logger
        log.debug("StrategyMutationPass initialized")

    def apply(self, base_prompt: str, metadata: dict) -> list[dict]:
        """
        Generate and score simple symbolic mutations of the base prompt.
        Returns sorted list of mutations with their scores (highest first).
        """
        log.debug(f"Applying strategy mutations to prompt (length: {len(base_prompt)})")
        mutations: List[Dict[str, Any]] = []
        
        # Define candidate mutations - these are simple text transformations
        candidates = [
            base_prompt.replace("Let's think step by step.", "Let's work through this carefully."),
            base_prompt + "\nProvide a rationale before giving your final answer.",
            base_prompt.replace("explain", "analyze"),
        ]
        
        log.debug(f"Generated {len(candidates)} mutation candidates")

        for i, variant in enumerate(candidates):
            try:
                log.debug(f"Scoring mutation candidate {i+1}/{len(candidates)}")
                score = self.score_fn(variant, metadata) if self.score_fn else None
                log.debug(f"Mutation candidate {i+1} scored: {score}")
            except Exception as e:
                score = None
                log.warning(f"Failed to score mutation candidate {i+1}: {e}")
                if self.logger:
                    self.logger.log(
                        "StrategyMutationEvalError", {"error": str(e), "prompt_snippet": variant[:100]}
                    )
            mutations.append({"prompt": variant, "score": score})

        # Sort by score (highest first), None scores go last
        sorted_mutations = sorted(mutations, key=lambda x: (-1e9 if x["score"] is None else -float(x["score"])))
        log.debug(f"Strategy mutations completed. Best score: {sorted_mutations[0].get('score') if sorted_mutations else 'N/A'}")
        return sorted_mutations


# =============================
# PromptEvolver core (DSPy + optional strategy mutation)
# =============================
class PromptEvolver:
    """
    Core prompt evolution engine that combines DSPy-based learning with optional
    symbolic strategy mutations. Generates refined prompts from examples and feedback.
    """

    def __init__(self, llm, logger=None, use_strategy_mutation: bool = False, evaluator_cfg=None, memory=None):
        log.info(f"Initializing PromptEvolver (strategy_mutation: {use_strategy_mutation})")
        self.llm = llm
        self.logger = logger
        self.use_strategy_mutation = bool(use_strategy_mutation)

        # Configure DSPy with the provided LLM
        try:
            dspy.configure(lm=self.llm)
            log.debug("DSPy configured successfully")
        except Exception as e:
            log.error(f"Failed to configure DSPy: {e}")
            if self.logger:
                self.logger.log("DSPyConfigureFailed", {"error": str(e)})

        # Initialize evaluator and adapter
        self.evaluator = None
        if evaluator_cfg:
            log.debug("Loading evaluator from configuration")
            self.evaluator = get_evaluator(evaluator_cfg, memory=memory, llm=llm, logger=logger)
        self.adapter = EvaluatorAdapter(self.evaluator, logger=self.logger)
        log.debug(f"Evaluator adapter initialized: {self.evaluator is not None}")

        # Initialize strategy mutation pass if enabled
        self.strategy_pass: Optional[StrategyMutationPass] = None
        if self.use_strategy_mutation:
            log.debug("Setting up strategy mutation pass")
            def _score_cb(prompt: str, metadata: dict) -> Optional[float]:
                """Callback function to score prompts for strategy mutations"""
                goal = metadata.get("goal", "") if isinstance(metadata, dict) else ""
                ref = metadata.get("hypotheses", "") if isinstance(metadata, dict) else ""
                score = None
                
                if goal:
                    log.debug("Scoring mutation against goal")
                    score = self.adapter.score_prompt_only(prompt, goal=goal, context=metadata)
                if score is None and ref:
                    log.debug("Scoring mutation against reference")
                    score = self.adapter.score_against_reference(prompt, reference_output=ref, context=metadata)
                
                log.debug(f"Mutation scoring result: {score}")
                return score
                
            self.strategy_pass = StrategyMutationPass(score_fn=_score_cb, logger=self.logger)
            log.info("Strategy mutation pass enabled")

        # Track sources of refined prompts (dspy vs strategy)
        self.sources_map: Dict[str, str] = {}
        log.info("PromptEvolver initialization complete")

    def evolve(self, examples: list[dict], context: dict | None = None, sample_size: int = 10) -> list[str]:
        """
        Main evolution method that combines DSPy tuning and optional strategy mutations.
        
        Args:
            examples: List of training examples with prompts, hypotheses, reviews, and scores
            context: Additional context for evaluation
            sample_size: Number of examples to use for DSPy training
            
        Returns:
            Ordered list of unique refined prompts
        """
        log.info(f"Starting prompt evolution with {len(examples)} examples, sample_size: {sample_size}")
        context = context or {}
        self.sources_map = {}
        
        if not examples:
            log.warning("No examples provided for evolution")
            return []

        # Build DSPy training set from examples
        train_items = examples[: max(1, int(sample_size))]
        log.debug(f"Building DSPy training set with {len(train_items)} items")
        
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
        log.debug(f"Training set built with {len(training_set)} examples")

        def eval_metric(example: Example, pred, trace=None):
            """
            DSPy metric function to evaluate refined prompts.
            Uses the evaluator if available, otherwise returns default score.
            """
            try:
                refined = (getattr(pred, "refined_prompt", "") or "").strip()
                if not refined:
                    log.debug("Empty refined prompt in evaluation")
                    return 1.0
                    
                if self.adapter is None:
                    log.debug("No evaluator adapter available, using default score")
                    return 1.0
                    
                ref = getattr(example, "hypotheses", "") or ""
                goal = getattr(example, "goal", "") or ""
                score = None
                
                if ref:
                    log.debug("Evaluating against reference output")
                    score = self.adapter.score_against_reference(refined, reference_output=ref, context=context)
                if score is None and goal:
                    log.debug("Evaluating against goal")
                    score = self.adapter.score_prompt_only(refined, goal=goal, context=context)
                    
                final_score = float(score) if isinstance(score, (int, float)) else 1.0
                log.debug(f"Evaluation metric score: {final_score}")
                return final_score
                
            except Exception as e:
                log.error(f"Evaluation metric failed: {e}")
                if self.logger:
                    self.logger.log("DSPyMetricFailed", {"error": str(e)})
                return 1.0

        # Compile DSPy program with BootstrapFewShot
        log.info("Compiling DSPy program with BootstrapFewShot")
        tuner = BootstrapFewShot(metric=eval_metric)
        student = Predict(PromptTuningSignature)
        
        try:
            tuned_program = tuner.compile(student=student, trainset=training_set)
            log.info("DSPy program compiled successfully")
        except Exception as e:
            log.error(f"DSPy compilation failed: {e}")
            return []

        # Generate refined prompts using the tuned program
        refined_prompts: List[str] = []
        log.info(f"Generating refined prompts for {len(examples)} examples")

        for i, ex in enumerate(examples):
            try:
                log.debug(f"Processing example {i+1}/{len(examples)}")
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
                    log.debug(f"Generated DSPy refined prompt (length: {len(refined)})")
                else:
                    log.debug("No refined prompt generated for this example")
            except Exception as e:
                log.error(f"Failed to generate prompt for example {i+1}: {e}")
                if self.logger:
                    self.logger.log("DSPyPromptEvolutionFailed", {"error": str(e)})

        log.info(f"DSPy evolution generated {len(refined_prompts)} refined prompts")

        # Apply strategy mutations if enabled
        if self.use_strategy_mutation and self.strategy_pass is not None:
            log.info("Applying strategy mutations")
            strategy_count = 0
            
            for i, ex in enumerate(examples):
                base_prompt = ex.get("prompt_text", "")
                if not base_prompt:
                    log.debug(f"Skipping example {i+1} - no base prompt")
                    continue
                    
                metadata = {
                    "goal": ex.get("goal", ""),
                    "hypotheses": ex.get("hypothesis_text", ""),
                    "review": ex.get("review", ""),
                    "score": ex.get("elo_rating", 1000),
                }
                try:
                    log.debug(f"Generating strategy mutations for example {i+1}")
                    mutations = self.strategy_pass.apply(base_prompt, metadata)
                    
                    for mut in mutations:
                        prompt_text = (mut.get("prompt", "") or "").strip()
                        if prompt_text:
                            refined_prompts.append(prompt_text)
                            self.sources_map[prompt_text] = "strategy"
                            strategy_count += 1
                            log.debug(f"Added strategy mutation (score: {mut.get('score')})")
                except Exception as e:
                    log.error(f"Strategy mutation failed for example {i+1}: {e}")
                    if self.logger:
                        self.logger.log("StrategyMutationFailed", {"error": str(e)})
            
            log.info(f"Strategy mutations generated {strategy_count} additional prompts")

        # Deduplicate while preserving order
        seen = set()
        deduped_prompts = [p for p in refined_prompts if p and not (p in seen or seen.add(p))]
        log.info(f"Evolution complete. Total unique prompts: {len(deduped_prompts)} "
                    f"(DSPy: {len([p for p in deduped_prompts if self.sources_map.get(p) == 'dspy'])}, "
                    f"Strategy: {len([p for p in deduped_prompts if self.sources_map.get(p) == 'strategy'])})")
        
        return deduped_prompts

    def score_prompt(self, prompt: str, reference_output: str = "", context: dict | None = None) -> float:
        """Score prompt against reference output."""
        log.debug(f"Scoring prompt against reference (lengths: prompt={len(prompt)}, ref={len(reference_output)})")
        s = self.adapter.score_against_reference(prompt, reference_output, context or {})
        score = float(s) if s is not None else -1.0
        log.debug(f"Prompt score against reference: {score}")
        return score

    def score_prompt_only(self, prompt: str, goal: str = "", context: dict | None = None) -> float:
        """Score prompt against goal only."""
        log.debug(f"Scoring prompt against goal (lengths: prompt={len(prompt)}, goal={len(goal)})")
        s = self.adapter.score_prompt_only(prompt, goal, context or {})
        score = float(s) if s is not None else -1.0
        log.debug(f"Prompt score against goal: {score}")
        return score


# =============================
# Agent wrapper (BaseAgent)
# =============================
class PromptEvolverAgent(BaseAgent):
    """
    Production-ready Prompt Evolution agent that combines DSPy-based learning
    with optional strategy mutations and evaluator-backed scoring.
    
    Features:
    - DSPy refinement using few-shot learning
    - Optional symbolic strategy mutations
    - Evaluator-backed prompt scoring
    - Integration with ZeroModel timeline for visualization
    - Event bus telemetry for monitoring
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger):
        log.info("Initializing PromptEvolverAgent")
        super().__init__(cfg, memory, container, logger)

        # Configuration parameters
        self.sample_size: int = max(1, int(cfg.get("sample_size", 10)))
        self.use_strategy_mutation: bool = bool(cfg.get("use_strategy_mutation", False))
        self.include_original: bool = bool(cfg.get("include_original", False))
        self.timeout: float = float(cfg.get("timeout", 300.0))

        # Event bus subjects
        subjects = (cfg.get("subjects") or {})
        self.event_subject: str = subjects.get("events", "prompt.evolver.events")

        # Timeline configuration
        timeline_cfg = (cfg.get("timeline") or {})
        self.timeline_enabled: bool = bool(timeline_cfg.get("enabled", True))

        # Service dependencies
        self.llm = self._safe_get_service("llm")
        self.zm = self._safe_get_service("zeromodel")

        # Default evaluator configuration
        self.default_evaluator_cfg: Optional[Dict[str, Any]] = cfg.get("evaluator")

        # Runtime state
        self.run_id: str = ""
        self.start_ts: float = 0.0

        log.info(f"PromptEvolverAgent configured: "
                    f"sample_size={self.sample_size}, "
                    f"strategy_mutation={self.use_strategy_mutation}, "
                    f"include_original={self.include_original}, "
                    f"timeout={self.timeout}s")

    def _safe_get_service(self, key: str):
        """Safely retrieve service from container with error handling."""
        try:
            service = self.container.get(key)
            log.debug(f"Successfully retrieved service: {key}")
            return service
        except Exception as e:
            log.warning(f"Service '{key}' not available: {e}")
            self._log_warn("ServiceNotFound", {"service": key, "error": str(e)})
            return None

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main agent execution method.
        
        Args:
            context: Execution context containing examples, goal, and configuration
            
        Returns:
            Updated context with refined prompts, candidates, and evolution statistics
        """
        log.info("Starting PromptEvolverAgent run")
        self.start_ts = time.time()
        self.run_id = context.get(PIPELINE_RUN_ID) or uuid.uuid4().hex
        context[PIPELINE_RUN_ID] = self.run_id

        log.debug(f"Run ID: {self.run_id}")

        # Extract input data and configuration
        examples: List[Dict[str, Any]] = list(context.get("examples") or [])
        if not examples:
            log.error("No examples provided in context")
            raise ValueError("PromptEvolverAgent: context['examples'] is required and non-empty.")

        goal_text: str = ((context.get("goal") or {}).get("goal_text") or "").strip()
        sample_size: int = max(1, int(context.get("sample_size", self.sample_size)))
        use_strategy_mutation: bool = bool(context.get("use_strategy_mutation", self.use_strategy_mutation))
        include_original: bool = bool(context.get("include_original", self.include_original))
        evaluator_cfg: Optional[Dict[str, Any]] = context.get("evaluator_cfg") or self.default_evaluator_cfg

        log.info(f"Processing {len(examples)} examples with configuration: "
                    f"sample_size={sample_size}, "
                    f"strategy_mutation={use_strategy_mutation}, "
                    f"include_original={include_original}, "
                    f"evaluator={'enabled' if evaluator_cfg else 'disabled'}")

        # Initialize the prompt evolver
        log.debug("Initializing PromptEvolver")
        evolver = PromptEvolver(
            llm=self.llm,
            logger=self.logger,
            use_strategy_mutation=use_strategy_mutation,
            evaluator_cfg=evaluator_cfg,
            memory=self.memory,
        )

        # Open timeline for visualization if enabled
        timeline_opened = False
        if self.timeline_enabled and self.zm:
            try:
                log.debug("Opening ZeroModel timeline")
                self.zm.timeline_open(self.run_id)
                timeline_opened = True
                await self._emit_event("timeline_opened", {"run_id": self.run_id})
                log.info("Timeline opened successfully")
            except Exception as e:
                log.error(f"Failed to open timeline: {e}")
                self._log_warn("TimelineOpenFailed", {"error": str(e)})

        # Execute prompt evolution (run in executor to avoid blocking)
        log.info("Starting prompt evolution process")
        loop = asyncio.get_running_loop()
        try:
            refined_prompts: List[str] = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: evolver.evolve(examples=examples, context=context, sample_size=sample_size),
                ),
                timeout=self.timeout,
            )
            log.info(f"Prompt evolution completed. Generated {len(refined_prompts)} refined prompts")
        except asyncio.TimeoutError:
            log.error(f"Prompt evolution timed out after {self.timeout} seconds")
            self._log_err("EvolutionTimedOut", {"timeout": self.timeout})
            refined_prompts = []
        except Exception as e:
            log.error(f"Prompt evolution failed: {e}")
            self._log_err("PromptEvolutionFailed", {"error": str(e)})
            refined_prompts = []

        # Deduplicate refined prompts
        seen = set()
        refined_prompts = [p for p in refined_prompts if p.strip() and not (p in seen or seen.add(p))]
        log.debug(f"After deduplication: {len(refined_prompts)} prompts")

        # Optionally include original prompts at the beginning
        original_prompts = [ex.get("prompt_text", "") for ex in examples if ex.get("prompt_text")]
        if include_original:
            log.debug("Including original prompts in results")
            for p in original_prompts:
                if p not in seen:
                    refined_prompts.insert(0, p)
                    seen.add(p)
            log.debug(f"After including originals: {len(refined_prompts)} total prompts")

        # Score candidates if evaluator is available
        candidates: List[Dict[str, Any]] = []
        scoring_enabled = evolver.evaluator is not None and len(refined_prompts) > 0
        ref_output = examples[0].get("hypothesis_text", "") if examples else ""

        log.info(f"Scoring {len(refined_prompts)} candidates (scoring_enabled: {scoring_enabled})")

        for idx, prompt in enumerate(refined_prompts):
            try:
                source = evolver.sources_map.get(
                    prompt, "original" if prompt in original_prompts else "dspy"
                )
                score = None
                
                if scoring_enabled:
                    if goal_text:
                        log.debug(f"Scoring candidate {idx+1} against goal")
                        score = float(evolver.score_prompt_only(prompt, goal=goal_text, context=context))
                    else:
                        log.debug(f"Scoring candidate {idx+1} against reference")
                        score = float(evolver.score_prompt(prompt, reference_output=ref_output, context=context))
                
                candidate_data = {"prompt": prompt, "score": score, "source": source, "rank": idx}
                candidates.append(candidate_data)
                
                log.debug(f"Candidate {idx+1}: source={source}, score={score}, "
                             f"prompt_length={len(prompt)}")
                
                await self._emit_event(
                    "candidate",
                    {"prompt": prompt[:160], "score": score, "source": source, "rank": idx},
                )

                # Add candidate to timeline if enabled
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
                        log.debug(f"Added candidate {idx+1} to timeline")
                    except Exception as te:
                        log.warning(f"Failed to add candidate to timeline: {te}")
                        self._log_warn("TimelineNodeFailed", {"error": str(te)})
                        
            except Exception as e:
                log.error(f"Failed to process candidate {idx+1}: {e}")
                self._log_warn("CandidateBuildFailed", {"error": str(e)})

        # Select best prompt based on scores
        log.info("Selecting best prompt from candidates")
        best_prompt, best_score, best_source = self._pick_best_with_source(candidates)
        
        # Fallback to original prompt if no candidates available
        if best_prompt is None and original_prompts:
            log.warning("No refined prompts available, falling back to original prompt")
            best_prompt, best_score, best_source = original_prompts[0], None, "fallback_original"

        log.info(f"Best prompt selected: source={best_source}, score={best_score}, "
                    f"length={len(best_prompt) if best_prompt else 0}")

        # Finalize timeline
        timeline_path: Optional[str] = None
        if timeline_opened:
            try:
                log.debug("Finalizing timeline")
                finalize_res = await self.zm.timeline_finalize(self.run_id)
                timeline_path = (
                    finalize_res.get("gif")
                    or finalize_res.get("output_path")
                    or getattr(self.zm, "last_output_for", lambda _rid: None)(self.run_id)
                )
                await self._emit_event("timeline_finalized", {"path": timeline_path})
                log.info(f"Timeline finalized at: {timeline_path}")
            except Exception as e:
                log.error(f"Failed to finalize timeline: {e}")
                self._log_warn("TimelineFinalizeFailed", {"error": str(e)})

        # Compile execution statistics
        elapsed_time = round(time.time() - self.start_ts, 3)
        stats = {
            "n_examples": len(examples),
            "n_candidates": len(candidates),
            "n_refined": len(refined_prompts),
            "mutations_used": use_strategy_mutation,
            "scoring_enabled": scoring_enabled,
            "elapsed_s": elapsed_time,
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        log.info(f"Evolution completed: {len(candidates)} candidates, "
                    f"best_score={best_score}, elapsed={elapsed_time}s")

        # Update context with results
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

        log.info("PromptEvolverAgent run completed successfully")
        return context

    # ----------------- helper methods -----------------
    def _pick_best_with_source(
        self, candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Select the best prompt from candidates based on score.
        Falls back to first candidate if no scores available.
        """
        log.debug(f"Selecting best prompt from {len(candidates)} candidates")
        if not candidates:
            log.warning("No candidates available for selection")
            return None, None, None
            
        # Filter candidates with valid scores
        scored = [c for c in candidates if isinstance(c.get("score"), (int, float))]
        
        if scored:
            best = max(scored, key=lambda c: c["score"])  # type: ignore[arg-type]
            log.debug(f"Best candidate selected with score: {best['score']}")
            return best["prompt"], float(best["score"]), best.get("source")
        else:
            # Fallback to first candidate if no scores available
            c = candidates[0]
            log.debug("No scored candidates available, using first candidate")
            return c["prompt"], c.get("score"), c.get("source")

    async def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        """Publish event to message bus if available."""
        if not self.memory or not getattr(self.memory, "bus", None):
            log.debug(f"No event bus available, skipping event: {event}")
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
            log.debug(f"Event published: {event}")
        except Exception as e:
            log.warning(f"Failed to publish event '{event}': {e}")
            self._log_warn("EmitEventFailed", {"error": str(e), "event": event})

    def _log_warn(self, tag: str, payload: Dict[str, Any]) -> None:
        """Helper for structured warning logging."""
        try:
            if self.logger:
                self.logger.log(tag, payload)
            else:
                log.warning("%s: %s", tag, payload)
        except Exception:
            pass

    def _log_err(self, tag: str, payload: Dict[str, Any]) -> None:
        """Helper for structured error logging."""
        try:
            if self.logger:
                self.logger.log(tag, payload)
            else:
                log.error("%s: %s", tag, payload)
        except Exception:
            pass