# stephanie/agents/learning/gild_trainer.py
import json
import os
import time
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
from sqlalchemy import text

from stephanie.agents.base_agent import BaseAgent
from stephanie.data.plan_trace import ExecutionStep, PlanTrace
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.scorer.ep_hrm_scorer import \
    EpistemicPlanHRMScorer  # Adjust import
from stephanie.scoring.scorer.hrm_scorer import HRMScorer
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.training.preference_pair_builder import \
    PreferencePairBuilder


class GILDTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.beta = cfg.get("beta", 1.0)  # Temperature for advantage weighting
        self.learning_rate = cfg.get("learning_rate", 1e-4)
        self.epochs = cfg.get(
            "gild_epochs", 5
        )  # Number of passes over the data
        self.batch_size = cfg.get("batch_size", 32)
        self.model_path = cfg.get("model_path", "models")
        self.target_type = cfg.get("target_type", "plan_trace")
        self.embedding_type = self.memory.embedding.name
        self.version = cfg.get("model_version", "v1")

        # --- Paths and Data Handling ---
        # If data was dumped to file, we need the path
        self.gild_data_file_path = cfg.get(
            "gild_data_file_path"
        )  # Fallback, ideally comes from context

        # If not provided, we can set a default path        # --- Training Components ---
        self.optimizer = None  # Will be initialized when model is loaded

        self.dimensions = cfg.get("dimensions", [])
        self.pair_builder = PreferencePairBuilder(memory.session, logger)

        self.hrm_scorer = HRMScorer(cfg.get("hrm", {}), memory, logger)
        self.sicql_scorer = SICQLScorer(cfg.get("sicql", {}), memory, logger)
        self.epistemic_plan_hrm_scorer = EpistemicPlanHRMScorer(
            cfg.get("epistemic_plan_hrm", {}), memory, logger
        )

        self.logger.log(
            "GILDTrainerAgentInitialized",
            {
                "beta": self.beta,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                # Add other relevant config
            },
        )

    # Inside GILDTrainerAgent.run (conceptual structure)

    async def run(self, context: dict) -> dict:
        # --- 1. Initialize GILD Process Trace (as before) ---
        gild_trace = None
        gild_step_order_counter = 1
        goal = context.get("goal")
        goal_id = goal.get("id")
        goal_text = goal.get("goal_text")
        expert_scorer = self.epistemic_plan_hrm_scorer

        try:
            trace_id = f"gild_trace_{int(time.time() * 1000)}_{hash(str(context)) % 10000}"
            gild_trace = PlanTrace(
                trace_id=trace_id,
                goal_id=goal_id,
                goal_text=goal_text[:1000],
                plan_signature=f"GILD_SICQL_Pi_Head_Update_v1",
                input_data={
                    "gild_config": {
                        k: v
                        for k, v in self.cfg.items()
                        if k.startswith("gild_")
                    },
                    "expert_scorer": expert_scorer,
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
            self.logger.log(
                "GILDProcessTraceStarted",
                {
                    "trace_id": trace_id,
                    "goal_id": goal_id,
                },
            )

        except Exception as e:
            self.logger.log("GILDProcessTraceInitError", {"error": str(e)})
            gild_trace = None

        # --- 2. Log Execution Step: Data Preparation ---
        data_prep_step_db_id = None
        if gild_trace:
            try:
                data_prep_step = ExecutionStep(
                    step_order=gild_step_order_counter,
                    step_id=f"{trace_id}_step_{gild_step_order_counter}",
                    description="Load and prepare GILD training data.",
                    output_text="",
                    scores=None,  # Assuming no scores yet
                    meta={},
                )
                # self.execution_step_store.add(data_prep_step)
                # Assuming insert returns the ID or you can get it
                # data_prep_step_db_id = data_prep_step.id
                # gild_step_order_counter += 1
            except Exception as e:
                self.logger.log(
                    "GILDProcessTraceDataPrepStepError",
                    {"error": str(e), "trace_id": trace_id},
                )

        # --- 3. Prepare GILD Training Data (YOUR SNIPPET STARTS HERE) ---
        # This is the core logic from your uploaded snippet
        try:
            sicql_advantages_data = self.extract_sicql_advantages()
            if not sicql_advantages_data:
                raise ValueError(
                    "No GILD signals (sicql_advantages) found in context."
                )

            # --- YOUR DATA PREP LOGIC ---
            prepared_data = []
            for item in sicql_advantages_data:
                try:
                    target_id = item["target_id"]
                    target_type = item["target_type"]
                    dimension = item["dimension"]
                    evaluation_id = item["evaluation_id"]

                    goal = self.memory.evaluations.get_goal(
                        evaluation_id
                    ).to_dict()
                    scorable = ScorableFactory.from_id(
                        self.memory, target_type, target_id
                    )
                    with torch.no_grad():
                        sicql_outputs = self.sicql_scorer(
                            goal, scorable, dimension
                        )
                        state_z = sicql_outputs.get("zsa")
                        state_z = state_z.detach().to(self.device)

                    prepared_data.append(
                        {
                            **item,
                            "state_z": state_z,  # This is the crucial part
                        }
                    )
                except Exception as e:
                    self.logger.log(
                        "GILDDataPrepItemFailed",
                        {"target_id": item.get("target_id"), "error": str(e)},
                    )
                    continue  # Continue with other items

            self.logger.log(
                "GILDDataPreparationCompleted",
                {
                    "prepared_items": len(prepared_data),
                    "total_input_items": len(sicql_advantages_data),
                },
            )

            # --- Update Data Prep Execution Step with Outcome ---
            if data_prep_step_db_id:
                try:
                    # Re-query or update the step ORM object
                    data_prep_step_orm = (
                        self.memory.execution_step_store.get_by_id(
                            data_prep_step_db_id
                        )
                    )
                    if data_prep_step_orm:
                        data_prep_step_orm.output_text = f"Loaded {len(sicql_advantages_data)} signals, prepared {len(prepared_data)} training examples."
                        # Add timing or other stats to meta if needed
                        # data_prep_step_orm.extra_data["prep_time_seconds"] = ...
                        self.execution_step_store.session.commit()
                except Exception as e:
                    self.logger.log(
                        "GILDProcessTraceDataPrepStepUpdateError",
                        {"error": str(e), "step_id": data_prep_step_db_id},
                    )

            if not prepared_data:
                raise RuntimeError(
                    "No data prepared for GILD training after processing."
                )

        except Exception as e:
            self.logger.log("GILDDataPreparationError", {"error": str(e)})
            # Log error step in trace if possible
            # ... (similar to previous draft)
            context["gild_status"] = "failed_data_prep"
            context["gild_error"] = str(e)
            if gild_trace:
                gild_trace.final_output_text = f"Failed during data prep: {e}"
                gild_trace.meta["completed_at"] = (
                    datetime.now().isoformat() + "Z"
                )
                self.plan_trace_store.session.commit()
            return context

        # --- 4. GILD Training Loop (YOUR SNIPPET CONTINUES) ---
        # Determine dimensions to update
        dimensions_to_update = list(
            set(item["dimension"] for item in prepared_data)
        )
        training_results = {}

        for dimension in dimensions_to_update:
            model = self.sicql_scorer.models.get(dimension)
            if not model:
                self.logger.log(
                    "GILDTrainingModelError",
                    {
                        "message": f"SICQL model for dimension '{dimension}' not found.",
                        "trace_id": trace_id if gild_trace else "unknown",
                    },
                )
                training_results[dimension] = {
                    "status": "model_not_found",
                    "error": "Model not found",
                }
                continue

            pi_head = model.pi_head
            if not pi_head:
                self.logger.log(
                    "GILDTrainingModelError",
                    {
                        "message": f"Pi head for dimension '{dimension}' not found.",
                        "trace_id": trace_id if gild_trace else "unknown",
                    },
                )
                training_results[dimension] = {
                    "status": "pi_head_not_found",
                    "error": "Pi head not found",
                }
                continue

            # Freeze other parts, unfreeze pi_head (as in previous draft)
            # ... (Freeze logic) ...

            optimizer = torch.optim.AdamW(
                pi_head.parameters(), lr=self.cfg.get("gild_lr", 1e-4)
            )

            # Log Training Start Step
            training_start_step_db_id = None
            if gild_trace:
                try:
                    training_start_step = ExecutionStep(
                        step_order=gild_step_order_counter,
                        step_id=f"{trace_id}_step_{gild_step_order_counter}",
                        description=f"Start GILD training for dimension '{dimension}'.",
                        output_text="",
                        scores=None,  # Assuming no scores yet
                        meta={
                            "trainable_params": sum(
                                p.numel() for p in pi_head.parameters()
                            )
                        },
                    )
                    gild_step_order_counter += 1
                except Exception as e:
                    self.logger.log(
                        "GILDProcessTraceTrainingStartStepError",
                        {"error": str(e), "trace_id": trace_id},
                    )

            try:
                # 1. Collect only the samples for THIS dimension
                dim_samples = [row for row in prepared_data if row["dimension"] == dimension]
                if not dim_samples:
                    training_results[dimension] = {"status": "skipped", "reason": "no samples"}
                    continue

                # 2. Freeze everything except the π-head
                for p in model.parameters():
                    p.requires_grad = False
                for p in pi_head.parameters():
                    p.requires_grad = True

                # 3. Fresh optimizer for this head
                self.optimizer = torch.optim.AdamW(pi_head.parameters(), lr=self.learning_rate)

                # 4. Epoch loop (uses your existing _run_training_epoch)
                epoch_losses = []
                for epoch in range(self.epochs):
                    avg_loss = self._run_training_epoch(model, dim_samples)
                    epoch_losses.append(avg_loss)
                    self.logger.log(
                        "GILDEpochCompleted",
                        {"epoch": epoch, "avg_loss": avg_loss, "dimension": dimension},
                    )

                # 5. Pack up results
                final_avg_loss = epoch_losses[-1] if epoch_losses else float("inf")
                training_results[dimension] = {
                    "status": "completed",
                    "final_loss": final_avg_loss,
                    "loss_history": epoch_losses,
                }

                final_avg_loss = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses
                    else float("inf")
                )

                # Log Training End Step with results
                if gild_trace:
                    try:
                        training_end_step = ExecutionStep(
                            step_order=gild_step_order_counter,
                            step_id=f"{trace_id}_step_{gild_step_order_counter}",
                            description=f"Completed GILD training for dimension '{dimension}'.",
                            output_text=f"Final average loss: {final_avg_loss:.6f}",
                            scores=None,  # Assuming no scores yet
                            meta={"final_loss": final_avg_loss,
                                         "epochs": self.epochs,
                                         "dimension": dimension},
                        )
                        gild_step_order_counter += 1
                    except Exception as e:
                        self.logger.log(
                            "GILDProcessTraceTrainingEndStepError",
                            {"error": str(e), "trace_id": trace_id},
                        )

                # Save updated model (as in your snippet)
                # ... (save logic) ...
                training_results[dimension] = {
                    "status": "completed",
                    "final_loss": final_avg_loss,
                    "loss_history": epoch_losses,
                }

            except Exception as e:
                self.logger.log(
                    "GILDTrainingLoopError",
                    {
                        "error": str(e),
                        "dimension": dimension,
                        "traceback": traceback.format_exc(),
                    },
                )
                # Log error step
                # ... (error step logic) ...
                training_results[dimension] = {
                    "status": "failed_training",
                    "error": str(e),
                    "final_loss": epoch_losses[-1] if epoch_losses else None,
                }
                # Decide whether to continue with other dimensions or fail completely
                # For now, let's continue

        # --- 5. Assign Epistemic Quality and Finalize Trace ---
        final_status = (
            "completed"
            if all(
                res.get("status") == "completed"
                for res in training_results.values()
            )
            else "completed_with_errors"
        )
        overall_final_loss = (
            sum(
                res.get("final_loss", 0)
                for res in training_results.values()
                if res.get("status") == "completed"
            )
            / len(
                [
                    r
                    for r in training_results.values()
                    if r.get("status") == "completed"
                ]
            )
            if any(
                r.get("status") == "completed"
                for r in training_results.values()
            )
            else float("inf")
        )

        # --- Calculate Proxy Epistemic Quality ---
        max_expected_loss = 0.1
        normalized_loss_quality = (
            max(0.0, min(1.0, 1.0 - (overall_final_loss / max_expected_loss)))
            if overall_final_loss != float("inf")
            else 0.0
        )

        if gild_trace:
            try:
                gild_trace.target_epistemic_quality = normalized_loss_quality
                gild_trace.target_epistemic_quality_source = (
                    "proxy_final_loss_normalized"
                )
                gild_trace.final_output_text = f"GILD run {final_status}. Overall final average loss: {overall_final_loss:.6f}. Assigned proxy epistemic quality: {normalized_loss_quality:.4f}."
                gild_trace.meta["completed_at"] = (
                    datetime.now().isoformat() + "Z"
                )
                gild_trace.meta["final_metrics"] = {
                    "overall_final_loss": overall_final_loss,
                    "proxy_epistemic_quality": normalized_loss_quality,
                    "epochs_run": self.epochs,
                    "per_dimension_results": training_results,  # Include detailed results
                }
                self.logger.log(
                    "GILDProcessTraceFinalized",
                    {
                        "trace_id": gild_trace.trace_id,
                        "epistemic_quality": normalized_loss_quality,
                        "overall_final_loss": overall_final_loss,
                    },
                )
            except Exception as e:
                self.logger.log(
                    "GILDProcessTraceFinalizationError", {"error": str(e)}
                )
                if gild_trace:
                    gild_trace.final_output_text += (
                        f" [Trace Finalization Error: {e}]"
                    )
                    gild_trace.meta["trace_finalization_error"] = str(e)

        # --- 6. Score the Trace with Epistemic HRM (as per suggestions) ---
        quality_pred = None
        if gild_trace:
            try:


                # Score the trace (Suggestion 3)
                # TODO convert to scorable
                score = self.epistemic_plan_hrm_scorer.score(gild_trace, self.dimensions)
                quality_pred = score.aggregate()

            except Exception as e:
                self.logger.log(
                    "GILDTraceHRMScoringError",
                    {
                        "error": str(e),
                        "trace_id": gild_trace.trace_id
                        if gild_trace
                        else "unknown",
                        "traceback": traceback.format_exc(),
                    },
                )
                # Don't fail the whole process if HRM scoring fails

        # --- 7. Update Context and Return ---
        context["gild_status"] = final_status
        context["gild_overall_final_loss"] = overall_final_loss
        context["gild_training_results"] = (
            training_results  # Detailed per-dimension results
        )
        if gild_trace:
            context["gild_trace_id"] = gild_trace.trace_id
            context["gild_epistemic_quality"] = (
                normalized_loss_quality  # The proxy
            )
            if quality_pred is not None:
                context["gild_hrm_predicted_quality"] = (
                    quality_pred  # Add HRM prediction to context
                )

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

    def _load_gild_signals(self, context: dict) -> dict:
        """Load GILD signals from context or file."""
        # 1. Try loading directly from context (if not dumped)
        signals = context.get("policy_synthesis_results", {}).get(
            "gild_signals"
        )
        if signals:
            self.logger.log("GILDDataLoadedFromContext", {})
            return signals

        # 2. Check if data was dumped and load from file
        # The PolicySynthesisAgent might have put the file path in the context
        psr = context.get("policy_synthesis_results", {})
        if (
            isinstance(psr, dict)
            and psr.get("large_data_dumped")
            and "dumped_to_file" in psr
        ):
            file_path = psr["dumped_to_file"]
        else:
            # Fallback to config path
            file_path = self.gild_data_file_path

        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    signals = json.load(f)
                self.logger.log(
                    "GILDDataLoadedFromFile", {"file_path": file_path}
                )
                return signals
            except Exception as e:
                self.logger.log(
                    "GILDDataLoadFromFileFailed",
                    {"file_path": file_path, "error": str(e)},
                )

        return {}

    def _prepare_training_data(self, sicql_advantages_data: list) -> list:
        """
        Prepare data for training: reconstruct states, organize tensors.
        This is a critical step requiring access to embeddings.
        """
        prepared_data = []
        for item in sicql_advantages_data:
            try:
                target_id = item["target_id"]
                target_type = item["target_type"]
                advantage = float(item["advantage"])  # Ensure it's a float
                dimension = item["dimension"]
                evaluation_id = item[
                    "evaluation_id"
                ]  # Optional ID for tracking
                goal = self.memory.evaluations.get_goal(
                    evaluation_id
                )  # You need to implement this
                scorable = ScorableFactory.from_id(
                    self.memory, target_type, target_id
                )  # You need to None implement this

                if not goal or not scorable:
                    self.logger.log(
                        "GILDDataPrepWarning",
                        {
                            "message": "Could not retrieve text for state reconstruction",
                            "target_id": target_id,
                            "target_type": target_type,
                        },
                    )
                    continue  # Skip this item

                with torch.no_grad():  # Usually, you get the *current* model's prediction without gradients
                    sicql_outputs = self.sicql_scorer(
                        goal.to_dict(), scorable, dimension
                    )
                    # sicql_outputs is the dictionary: {"q_value": ..., "state_value": ..., ...}
                state_z = self.sicql_scorer.encode(
                    goal.to_dict(), scorable, dimension
                )
                prepared_data.append(
                    {
                        "q_value": sicql_outputs["q_value"].item(),
                        "state_value": sicql_outputs[
                            "state_value"
                        ].item(),  # Get the state value
                        "advantage": torch.tensor(
                            advantage, dtype=torch.float32
                        ),  # Tensor
                        "state_z": state_z,
                        "target_id": target_id,
                        "target_type": target_type,
                        "dimension": dimension,
                        "evaluation_id": evaluation_id,
                    }
                )
            except Exception as e:
                self.logger.log(
                    "GILDDataPrepItemFailed",
                    {"target_id": item.get("target_id"), "error": str(e)},
                )
                # Continue with other items

        self.logger.log(
            "GILDDataPreparationCompleted",
            {
                "prepared_items": len(prepared_data),
                "total_input_items": len(sicql_advantages_data),
            },
        )
        return prepared_data

    def _run_training_epoch(self, model, prepared_data: list) -> float:
        """Run one epoch of GILD training."""
        total_loss = 0.0
        num_batches = 0

        # Simple batching (you might want a proper DataLoader)
        for i in range(0, len(prepared_data), self.batch_size):
            batch = prepared_data[i : i + self.batch_size]

            # Aggregate batch data
            batch_states = torch.stack(
                [item["state_z"] for item in batch]
            )  # Shape: (batch_size, z_dim)
            batch_advantages = torch.stack(
                [item["advantage"] for item in batch]
            )  # Shape: (batch_size,)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass through the policy head only
            # model.pi_head should take state_z and output action_logits
            action_logits = model.pi_head(
                batch_states
            )  # Shape: (batch_size, action_dim)

            # --- Core GILD Update ---
            # Calculate log probabilities
            log_probs = F.log_softmax(
                action_logits, dim=-1
            )  # Shape: (batch_size, action_dim)

            # Calculate weights from advantages
            # Ensure advantages are detached and have correct shape for broadcasting
            weights = torch.exp(
                self.beta * batch_advantages.detach()
            )  # Shape: (batch_size,)
            weights = weights / (
                weights.sum() + 1e-8
            )  # Normalize weights (optional but often done)
            weights = weights.unsqueeze(
                -1
            )  # Shape: (batch_size, 1) for broadcasting

            # Calculate weighted imitation loss
            # We sum over actions (dim=-1) and mean over the batch
            pi_loss = -(log_probs * weights).sum(dim=-1).mean()  # Scalar loss

            # Backward pass
            pi_loss.backward()

            # Update parameters
            self.optimizer.step()

            total_loss += pi_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def extract_sicql_advantages(self,
        dimensions: list[str] | None = None,                         
        min_length: int = 1_000,
        limit: int | None = 10,
    ) -> list[dict[str, any]]:
        """Pull `(goal, doc)`‑level *advantage* records produced by the SICQL scorer.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            Active DB session.
        logger : Any
            Object with a `.log(event: str, payload: dict)` method.
        dimensions : list[str] | None, default ``None``
            If given, filter to this subset of HRM/SICQL dimensions.
        min_length : int, default ``1_000``
            Emit a warning if fewer than this many rows are returned.
        limit : int | None, default ``10``
            Hard cap on the number of rows.  Set to ``None`` to disable.
        """

        base_sql = """
            SELECT
                e.id   AS evaluation_id,
                e.goal_id,
                e.target_id,
                e.target_type,
                s.dimension,
                ea.q_value,
                ea.v_value,
                ea.source,
                ea.pi_value,
                ea.advantage
            FROM evaluation_attributes ea
            JOIN evaluations e ON ea.evaluation_id = e.id
            JOIN scores      s ON s.evaluation_id = e.id AND s.dimension = ea.dimension
            WHERE e.source = :source
            AND ea.advantage IS NOT NULL
        """

        params: dict[str, any] = {"source": "sicql"}

        if dimensions:
            base_sql += "\n          AND s.dimension IN :dims"
            params["dims"] = tuple(dimensions)

        base_sql += "\n        ORDER BY s.dimension"

        if limit is not None:
            base_sql += "\n        LIMIT :lim"
            params["lim"] = int(limit)

        rows = self.memory.session.execute(text(base_sql), params).fetchall()
        result = [dict(r._mapping) for r in rows]

        self.logger.log("SICQLAdvantageExtracted", {
            "total": len(result),
            "dimensions": dimensions or "all",
            "limit": limit,
        })

        if len(result) < min_length:
            self.logger.log("SICQLAdvantageWarning", {
                "message": f"Only {len(result)} records found — might be insufficient for training.",
                "min_length": min_length,
            })

        return result