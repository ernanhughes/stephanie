# stephanie/agents/gild_trainer.py
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from sqlalchemy import text

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.gild import GILDConfig, GILDSignal, GILDTrainingResult
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.scoring.scorable import ScorableFactory
from stephanie.scoring.scorer.ep_hrm_scorer import \
    EpistemicPlanHRMScorer  # Adjust import
from stephanie.scoring.scorer.hrm_scorer import HRMScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder
from stephanie.utils.model_locator import ModelLocator

log = logging.getLogger(__name__)


class GILDTrainerAgent(BaseAgent):
    """
    GILDTrainerAgent v2

    - Uses GILDSignal / GILDConfig / GILDTrainingResult.
    - Extracts SICQL advantage examples from DB (or context/file).
    - Reconstructs latent `state_z` with SICQLScorer.encode().
    - Runs AWR-style π-head training per dimension.
    - Writes detailed results + proxy epistemic quality into PlanTrace.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Canonical GILD config
        self.gild_cfg = GILDConfig(
            beta=cfg.get("beta", cfg.get("gild_beta", 1.0)),
            learning_rate=cfg.get("learning_rate", cfg.get("gild_lr", 1e-4)),
            batch_size=cfg.get("batch_size", cfg.get("gild_batch_size", 32)),
            epochs=cfg.get("gild_epochs", 5),
            max_examples=cfg.get("gild_max_examples"),
            min_abs_advantage=cfg.get("gild_min_abs_advantage", 0.0),
            entropy_coef=cfg.get("gild_entropy_coef", 0.0),
            gradient_clip_norm=cfg.get("gild_gradient_clip_norm", 1.0),
            warm_start_only=cfg.get("gild_warm_start_only", False),
            warm_start_fraction=cfg.get("gild_warm_start_fraction", 0.1),
        )

        # Backwards-compatible simple attributes
        self.beta = self.gild_cfg.beta
        self.learning_rate = self.gild_cfg.learning_rate
        self.epochs = self.gild_cfg.epochs
        self.batch_size = self.gild_cfg.batch_size

        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "plan_trace")
        self.embedding_type = self.memory.embedding.name
        self.version = cfg.get("model_version", "v1")

        # --- Paths and Data Handling ---
        self.gild_data_file_path = cfg.get("gild_data_file_path")

        # --- Training Components ---
        self.optimizer = None  # Will be initialized when model is loaded
        self.dimensions = cfg.get("dimensions", [])
        self.pair_builder = PreferencePairBuilder(memory, logger)

        self.hrm_scorer = HRMScorer(cfg.get("hrm", {}), memory, logger)
        self.sicql_scorer = SICQLScorer(cfg.get("sicql", {}), memory, logger)
        self.epistemic_plan_hrm_scorer = EpistemicPlanHRMScorer(
            cfg.get("epistemic_plan_hrm", {}), memory, logger
        )

        log.info(
            "GILDTrainerAgentInitialized gild_config %s model_path %s target_type %s embedding_type %s version %s",
            asdict(self.gild_cfg),
            self.model_path,
            self.target_type,
            self.embedding_type,
            self.version,
        )

    # -----------------------------
    #  Main entry point
    # -----------------------------

    async def run(self, context: dict) -> dict:
        # --- 1. Initialize GILD Process Trace ---
        gild_trace: Optional[PlanTrace] = None
        gild_step_order_counter = 1

        goal = context.get("goal") or {}
        goal_id = goal.get("id")
        goal_text = goal.get("goal_text", "") or ""
        expert_scorer = self.epistemic_plan_hrm_scorer

        try:
            trace_id = f"gild_trace_{int(time.time() * 1000)}_{hash(str(context)) % 10000}"
            gild_trace = PlanTrace(
                trace_id=trace_id,
                goal_id=goal_id,
                goal_text=goal_text[:1000],
                plan_signature="GILD_SICQL_Pi_Head_Update_v2",
                input_data={
                    "gild_config": asdict(self.gild_cfg),
                    "expert_scorer": str(expert_scorer),
                },
                final_output_text="",
                execution_steps=[],
                target_epistemic_quality=None,
                target_epistemic_quality_source=None,
                meta={
                    "agent_name": self.__class__.__name__,
                    "started_at": datetime.now().isoformat() + "Z",
                },
            )
            log.info(
                "GILDProcessTraceStarted trace_id %s goal_id %s",
                trace_id, goal_id,
            )
        except Exception as e:
            log.error("GILDProcessTraceInitError: %s", str(e))
            gild_trace = None

        # --- 2. Log Execution Step: Data Preparation ---
        if gild_trace:
            try:
                data_prep_step = ExecutionStep(
                    step_order=gild_step_order_counter,
                    step_id=f"{gild_trace.trace_id}_step_{gild_step_order_counter}",
                    description="Load and prepare GILD training data.",
                    output_text="",
                    scores=None,
                    meta={},
                )
                gild_trace.execution_steps.append(data_prep_step)
                gild_step_order_counter += 1
            except Exception as e:
                log.error(
                    "GILDProcessTraceDataPrepStepError: %s trace_id %s",
                    str(e), gild_trace.trace_id,
                )

        # --- 3. Prepare GILD Training Data ---
        try:
            # 3a. Try to load signals directly from context/file
            raw_signals = self._load_gild_signals(context)

            # 3b. If none, fall back to SQL extraction of SICQL advantages
            if not raw_signals:
                raw_signals = self.extract_sicql_advantages(
                    dimensions=self.dimensions or None,
                    min_length=1_000,
                    limit=self.cfg.get("gild_sql_limit", 10_000),
                )

            if not raw_signals:
                raise ValueError("No GILD signals found for training.")

            # 3c. Convert raw rows → GILDSignal objects
            prepared_signals: List[GILDSignal] = self._prepare_training_data(
                raw_signals
            )

            log.info(
                "GILDDataPreparationCompleted prepared_items %d total_input_items %d",
                len(prepared_signals),
                len(raw_signals),
            )
 
            if gild_trace and gild_trace.execution_steps:
                # Update the last step (data prep) with a short summary
                gild_trace.execution_steps[-1].output_text = (
                    f"Loaded {len(raw_signals)} signals, "
                    f"prepared {len(prepared_signals)} GILDSignal examples."
                )

            if not prepared_signals:
                raise RuntimeError(
                    "No data prepared for GILD training after processing."
                )

        except Exception as e:
            log.error("GILDDataPreparationError: %s", str(e))
            context["gild_status"] = "failed_data_prep"
            context["gild_error"] = str(e)
            if gild_trace:
                gild_trace.final_output_text = f"Failed during data prep: {e}"
                gild_trace.meta["completed_at"] = (
                    datetime.now().isoformat() + "Z"
                )
            return context

        # --- 4. GILD Training Loop (per-dimension AWR over GILDSignal) ---
        dimensions_to_update = sorted(
            {s.dimension for s in prepared_signals}
        )
        training_results: Dict[str, GILDTrainingResult] = {}

        for dimension in dimensions_to_update:
            model = self.sicql_scorer.models.get(dimension)
            if not model:
                msg = f"SICQL model for dimension '{dimension}' not found."
                log.error(
                    "GILDTrainingModelError: %s dimension %s",
                    msg, dimension,
                )
                training_results[dimension] = GILDTrainingResult(
                    dimension=dimension,
                    status="model_not_found",
                    final_loss=float("inf"),
                    num_examples=0,
                    num_epochs=0,
                    meta={"error": msg},
                )
                continue

            pi_head = getattr(model, "pi_head", None)
            if pi_head is None:
                msg = f"Pi head for dimension '{dimension}' not found."
                log.error(
                    "GILDTrainingModelError: %s dimension %s",
                    msg, dimension,
                )
                training_results[dimension] = GILDTrainingResult(
                    dimension=dimension,
                    status="pi_head_not_found",
                    final_loss=float("inf"),
                    num_examples=0,
                    num_epochs=0,
                    meta={"error": msg},
                )
                continue

            # Log Training Start Step
            if gild_trace:
                try:
                    training_start_step = ExecutionStep(
                        step_order=gild_step_order_counter,
                        step_id=f"{gild_trace.trace_id}_step_{gild_step_order_counter}",
                        description=f"Start GILD training for dimension '{dimension}'.",
                        output_text="",
                        scores=None,
                        meta={
                            "trainable_params": sum(
                                p.numel() for p in pi_head.parameters()
                            )
                        },
                    )
                    gild_trace.execution_steps.append(training_start_step)
                    gild_step_order_counter += 1
                except Exception as e:
                    log.error(
                        "GILDProcessTraceTrainingStartStepError: %s trace_id %s",
                        str(e), gild_trace.trace_id,
                    )

            try:
                # 1. Filter signals for this dimension
                dim_signals = [
                    s for s in prepared_signals if s.dimension == dimension
                ]
                if not dim_signals:
                    training_results[dimension] = GILDTrainingResult(
                        dimension=dimension,
                        status="skipped",
                        final_loss=float("inf"),
                        num_examples=0,
                        num_epochs=0,
                        meta={"reason": "no_samples"},
                    )
                    continue

                # Optional cap on examples
                if (
                    self.gild_cfg.max_examples is not None
                    and len(dim_signals) > self.gild_cfg.max_examples
                ):
                    dim_signals = dim_signals[: self.gild_cfg.max_examples]

                # Move to device once
                for s in dim_signals:
                    s.to_device(self.device).detach()

                # 2. Freeze everything except the π-head
                for p in model.parameters():
                    p.requires_grad = False
                for p in pi_head.parameters():
                    p.requires_grad = True

                # 3. Optimizer for this head
                self.optimizer = torch.optim.AdamW(
                    pi_head.parameters(), lr=self.gild_cfg.learning_rate,
                    weight_decay=1e-5
                )

                # 4. Epoch loop
                epoch_losses: List[float] = []
                for epoch in range(self.gild_cfg.epochs):
                    avg_loss = self._run_training_epoch(model, dim_signals)
                    epoch_losses.append(avg_loss)
                    log.info(
                        "GILDEpochCompleted epoch %d avg_loss %.6f dimension %s",
                        epoch,
                        avg_loss,
                        dimension,
                    )

                final_avg_loss = (
                    epoch_losses[-1] if epoch_losses else float("inf")
                )

                training_results[dimension] = GILDTrainingResult(
                    dimension=dimension,
                    status="completed",
                    final_loss=final_avg_loss,
                    num_examples=len(dim_signals),
                    num_epochs=len(epoch_losses),
                    trace_id=gild_trace.trace_id if gild_trace else None,
                    meta={"loss_history": epoch_losses},
                )

                # Log Training End Step with results
                if gild_trace:
                    try:
                        training_end_step = ExecutionStep(
                            step_order=gild_step_order_counter,
                            step_id=f"{gild_trace.trace_id}_step_{gild_step_order_counter}",
                            description=f"Completed GILD training for dimension '{dimension}'.",
                            output_text=(
                                f"Final average loss: {final_avg_loss:.6f}, "
                                f"examples: {len(dim_signals)}, "
                                f"epochs: {len(epoch_losses)}"
                            ),
                            scores=None,
                            meta={
                                "final_loss": final_avg_loss,
                                "epochs": len(epoch_losses),
                                "dimension": dimension,
                            },
                        )
                        gild_trace.execution_steps.append(training_end_step)
                        gild_step_order_counter += 1
                    except Exception as e:
                        self.logger.log(
                            "GILDProcessTraceTrainingEndStepError",
                            {"error": str(e), "trace_id": gild_trace.trace_id},
                        )

                try:
                    locator = ModelLocator(
                        root_dir=self.model_path,
                        embedding_type=self.embedding_type,
                        model_type="sicql",
                        target_type=self.target_type,
                        dimension=dimension,
                        version=self.version,
                    )
                    locator.ensure_dirs()
                    pi_head_path = locator.pi_head_file()
                    torch.save(pi_head.state_dict(), pi_head_path)
                    training_results[dimension].meta["pi_head_path"] = pi_head_path
                    self.logger.log(
                        "GILDModelSaved",
                        {"dimension": dimension, "pi_head_path": pi_head_path},
                    )
                except Exception as e:
                    log.error(
                        "GILDModelSaveError dimension %s error %s",
                        dimension,
                        str(e),
                    )

            except Exception as e:
                tb = traceback.format_exc()
                log.error(
                    "GILDTrainingLoopError: %s dimension %s traceback %s",
                    str(e),
                    dimension,
                    tb,
                )
                training_results[dimension] = GILDTrainingResult(
                    dimension=dimension,
                    status="failed_training",
                    final_loss=float("inf"),
                    num_examples=0,
                    num_epochs=0,
                    meta={"error": str(e), "traceback": tb},
                )
                # Continue with other dimensions

        # --- 5. Assign Epistemic Quality and Finalize Trace ---
        completed_results = [
            r for r in training_results.values() if r.status == "completed"
        ]
        if completed_results:
            overall_final_loss = sum(r.final_loss for r in completed_results) / len(
                completed_results
            )
        else:
            overall_final_loss = float("inf")

        final_status = (
            "completed"
            if completed_results
            and all(r.status == "completed" for r in training_results.values())
            else "completed_with_errors"
        )

        # Proxy epistemic quality from normalized loss
        max_expected_loss = 0.1
        if overall_final_loss == float("inf"):
            normalized_loss_quality = 0.0
        else:
            normalized_loss_quality = max(
                0.0,
                min(1.0, 1.0 - (overall_final_loss / max_expected_loss)),
            )

        if gild_trace:
            try:
                gild_trace.target_epistemic_quality = normalized_loss_quality
                gild_trace.target_epistemic_quality_source = (
                    "proxy_final_loss_normalized"
                )
                gild_trace.final_output_text = (
                    f"GILD run {final_status}. Overall final average loss: "
                    f"{overall_final_loss:.6f}. Assigned proxy epistemic "
                    f"quality: {normalized_loss_quality:.4f}."
                )
                gild_trace.meta["completed_at"] = (
                    datetime.now().isoformat() + "Z"
                )
                gild_trace.meta["final_metrics"] = {
                    "overall_final_loss": overall_final_loss,
                    "proxy_epistemic_quality": normalized_loss_quality,
                    "epochs_run": self.gild_cfg.epochs,
                    "per_dimension_results": {
                        dim: asdict(res)
                        for dim, res in training_results.items()
                    },
                }
                log.info(
                    "GILDProcessTraceFinalized trace_id %s epistemic_quality %.4f overall_final_loss %.6f",
                    gild_trace.trace_id,
                    normalized_loss_quality,
                    overall_final_loss,
                )
            except Exception as e:
                log.error(
                    "GILDProcessTraceFinalizationError: %s",
                    str(e),
                )
                if gild_trace:
                    gild_trace.final_output_text += (
                        f" [Trace Finalization Error: {e}]"
                    )
                    gild_trace.meta["trace_finalization_error"] = str(e)

        # --- 6. Score the Trace with Epistemic HRM ---
        quality_pred = None
        if gild_trace:
            try:
                score = self.epistemic_plan_hrm_scorer.score(
                    gild_trace, self.dimensions
                )
                quality_pred = score.aggregate()
            except Exception as e:
                log.error(
                    "GILDTraceHRMScoringError: %s trace_id %s traceback %s",
                    str(e),
                    gild_trace.trace_id,
                    traceback.format_exc(),
                )

        # --- 7. Update Context and Return ---
        context["gild_status"] = final_status
        context["gild_overall_final_loss"] = overall_final_loss
        context["gild_training_results"] = {
            dim: asdict(res) for dim, res in training_results.items()
        }
        if gild_trace:
            context["gild_trace_id"] = gild_trace.trace_id
            context["gild_epistemic_quality"] = normalized_loss_quality
            if quality_pred is not None:
                context["gild_hrm_predicted_quality"] = quality_pred

        self.logger.log(
            "GILDTrainerAgentCompleted",
            {
                "status": context["gild_status"],
                "overall_final_loss": context.get("gild_overall_final_loss"),
                "trace_recorded": gild_trace is not None,
                "hrm_scored": quality_pred is not None,
            },
        )

        return context

    # -----------------------------
    #  Data loading & prep
    # -----------------------------

    def _load_gild_signals(self, context: dict) -> List[Dict[str, Any]]:
        """
        Load precomputed GILD signals from context or from a dumped JSON file.

        Expected shape is "raw" rows, which _prepare_training_data will turn
        into GILDSignal objects.
        """
        signals = context.get("policy_synthesis_results", {}).get(
            "gild_signals"
        )
        if signals:
            self.logger.log("GILDDataLoadedFromContext", {})
            return signals

        psr = context.get("policy_synthesis_results", {})
        if (
            isinstance(psr, dict)
            and psr.get("large_data_dumped")
            and "dumped_to_file" in psr
        ):
            file_path = psr["dumped_to_file"]
        else:
            file_path = self.gild_data_file_path

        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    signals = json.load(f)
                log.info(
                    "GILDDataLoadedFromFile file_path %s", file_path
                )
                return signals
            except Exception as e:
                log.error(
                    "GILDDataLoadFromFileFailed file_path %s error %s",
                    file_path, str(e)
                )

        return []

    def _prepare_training_data(
        self, sicql_advantages_data: List[Dict[str, Any]]
    ) -> List[GILDSignal]:
        """
        Convert raw SICQL advantage rows into GILDSignal objects.

        Each row is expected to have:
        - evaluation_id, goal_id, target_id, target_type, dimension
        - q_value, v_value, advantage
        Optionally:
        - source, pi_value
        If q_value / v_value are missing we recompute with SICQLScorer.
        """
        prepared: List[GILDSignal] = []

        for item in sicql_advantages_data:
            try:
                evaluation_id = item["evaluation_id"]
                target_id = item["target_id"]
                target_type = item["target_type"]
                dimension = item["dimension"]
                goal_id = item.get("goal_id")

                # Advantage → tensor
                advantage_val = float(item["advantage"])
                advantage_tensor = torch.tensor(
                    advantage_val, dtype=torch.float32
                )

                # Retrieve goal + scorable
                goal = self.memory.evaluations.get_goal(evaluation_id)
                scorable = ScorableFactory.from_id(
                    self.memory, target_type, target_id
                )

                if not goal or not scorable:
                    self.logger.log(
                        "GILDDataPrepWarning",
                        {
                            "message": "Could not retrieve goal or scorable.",
                            "target_id": str(target_id),
                            "target_type": target_type,
                        },
                    )
                    continue

                goal_dict = goal.to_dict()

                # Use existing q/v if present; otherwise recompute
                q_value = item.get("q_value")
                v_value = item.get("v_value")

                if q_value is None or v_value is None:
                    with torch.no_grad():
                        sicql_outputs = self.sicql_scorer(
                            goal_dict, scorable, dimension
                        )
                    q_value = sicql_outputs["q_value"].item()
                    v_value = sicql_outputs["state_value"].item()

                # Encode latent state `state_z`
                state_z = self.sicql_scorer.encode(goal_dict, scorable, dimension)

                signal = GILDSignal(
                    evaluation_id=evaluation_id,
                    goal_id=goal_id,
                    target_id=target_id,
                    target_type=target_type,
                    dimension=dimension,
                    q_value=float(q_value),
                    state_value=float(v_value),
                    advantage=advantage_tensor,
                    state_z=state_z,
                    source=item.get("source", "sicql"),
                    pi_value=(
                        float(item["pi_value"])
                        if item.get("pi_value") is not None
                        else None
                    ),
                    meta={},
                )

                prepared.append(signal)

            except Exception as e:
                log.error(
                    "GILDDataPrepItemFailed target_id %s error %s",
                    str(item.get("target_id")), str(e),
                )
                continue

        log.info(
            "GILDDataPreparationCompletedRaw prepared_items %d total_input_items %d",
            len(prepared),
            len(sicql_advantages_data),
        )
        return prepared

    # -----------------------------
    #  AWR training epoch
    # -----------------------------

    def _run_training_epoch(
        self, model: Any, signals: List[GILDSignal]
    ) -> float:
        """Run one epoch of GILD AWR training over a list of GILDSignal."""
        total_loss = 0.0
        num_batches = 0

        batch_size = self.gild_cfg.batch_size

        for i in range(0, len(signals), batch_size):
            batch_signals = signals[i : i + batch_size]

            # Convert to tensors
            batch_states, batch_advantages = GILDSignal.batch_to_tensors(
                batch_signals, device=self.device
            )

            # Optional advantage thresholding
            if self.gild_cfg.min_abs_advantage > 0.0:
                mask = batch_advantages.abs() >= self.gild_cfg.min_abs_advantage
                if mask.sum() == 0:
                    continue
                batch_advantages = batch_advantages[mask]
                batch_states = batch_states[mask]

            if batch_states.numel() == 0:
                continue

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass through the policy head only
            action_logits = model.pi_head(batch_states)
            log_probs = F.log_softmax(action_logits, dim=-1)

            # Advantage-weighted imitation
            weights = torch.exp(self.gild_cfg.beta * batch_advantages.detach())
            weights = weights / (weights.sum() + 1e-8)
            weights = weights.unsqueeze(-1)

            pi_loss = -(log_probs * weights).sum(dim=-1).mean()

            # Optional entropy regularization
            if self.gild_cfg.entropy_coef > 0.0:
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                pi_loss = pi_loss - self.gild_cfg.entropy_coef * entropy

            # Backward
            pi_loss.backward()

            # Optional gradient clipping
            if self.gild_cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.pi_head.parameters(), self.gild_cfg.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += pi_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    # --------------------
