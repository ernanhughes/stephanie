# stephanie/agents/gild_trace.py
from __future__ import annotations

import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from stephanie.agents.base_agent import BaseAgent
# ✅ Use the dataclass layer (same as new SSP Trainer)
from stephanie.data.plan_trace import PlanTrace, ExecutionStep

# Optional: fallback stores if memory.plan_traces repo missing
from stephanie.memory.plan_trace_store import PlanTraceStore as _FallbackPlanTraceStore  # optional
from stephanie.memory.execution_step_store import ExecutionStepStore as _FallbackExecStore  # optional


class _TraceRepoAdapter:
    """
    Small adapter that prefers memory.plan_traces.upsert(PlanTrace),
    with a safe fallback to the older stores if needed.
    """
    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger

        self.repo = getattr(memory, "plan_traces", None)
        # Fallbacks (only used if repo doesn’t exist)
        self._fallback_plan = None
        self._fallback_steps = None
        if self.repo is None and hasattr(memory, "session"):
            try:
                self._fallback_plan = _FallbackPlanTraceStore(memory.session, logger)
                self._fallback_steps = _FallbackExecStore(memory.session, logger)
            except Exception:
                self._fallback_plan = None
                self._fallback_steps = None

    def upsert_trace(self, pt: PlanTrace) -> PlanTrace:
        if self.repo and hasattr(self.repo, "upsert"):
            return self.repo.upsert(pt)
        # Fallback: best-effort upsert using stores
        if self._fallback_plan:
            try:
                # The fallback store may accept dataclass; if not, convert here as needed.
                return self._fallback_plan.upsert(pt)  # type: ignore
            except Exception as e:
                self.logger.log("GILDTraceFallbackUpsertError", {"error": str(e)})
        raise RuntimeError("No plan_traces repo or fallback available")

    def add_step(self, pt: PlanTrace, step: ExecutionStep) -> PlanTrace:
        # Preferred path: just mutate and upsert again
        pt.execution_steps = list(pt.execution_steps or []) + [step]
        return self.upsert_trace(pt)


class GILDTraceAgent(BaseAgent):
    """
    GILD Trainer that performs policy refinement and records the full run
    as a PlanTrace via memory.plan_traces.upsert(...).
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.beta = cfg.get("gild_beta", 1.0)
        self.epochs = cfg.get("gild_epochs", 5)
        self.batch_size = cfg.get("gild_batch_size", 32)
        self.lr = cfg.get("gild_lr", 1e-4)

        # Repo adapter (prefers memory.plan_traces.upsert)
        self._repo = _TraceRepoAdapter(memory, logger)

        # Device
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = "cpu"

    async def run(self, context: dict) -> dict:
        """
        Executes the GILD training process and records it as a PlanTrace.
        """
        goal_id = context.get("goal_id")
        goal_text = context.get("goal_text", "Unknown Goal")
        dimension = context.get("dimension", "Unknown Dimension")
        expert_scorer = context.get("expert_scorer", "Unknown Expert")
        pipeline_run_id = context.get("pipeline_run_id") or f"gild-{int(time.time()*1000)}"

        # ---------- 1) Create PlanTrace (status: in_progress) ----------
        trace_id = f"gild-{int(time.time()*1000)}-{abs(hash(goal_text))%1_000_000}"
        plan_trace = PlanTrace(
            trace_id=trace_id,
            pipeline_run_id=pipeline_run_id,
            goal_id=goal_id if goal_id is not None else abs(hash(goal_text)) % 1_000_000_000,
            goal_text=str(goal_text)[:1000],
            plan_signature=f"GILD_SICQL_Pi_Head_Update[{dimension}]",
            input_data={
                "gild_config": {k: v for k, v in dict(self.cfg).items() if str(k).startswith("gild_")},
                "expert_scorer": expert_scorer,
                "dimension": dimension,
            },
            execution_steps=[],
            final_output_text="",
            status="in_progress",
            created_at=datetime.utcnow().isoformat() + "Z",
            meta={
                "agent_name": self.__class__.__name__,
                "started_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        plan_trace = self._repo.upsert_trace(plan_trace)
        self.logger.log("GILDProcessTraceStarted", {"trace_id": plan_trace.trace_id})

        # ---------- 2) Data Prep Step ----------
        try:
            gild_signals = context.get("gild_signals", []) or []
            train_loader = context.get("gild_dataloader")  # optional, prebuilt

            step = ExecutionStep(
                step_id=f"{trace_id}-data",
                pipeline_run_id=pipeline_run_id,
                step_order=1,
                step_type="data_preparation",
                description="Load and prepare GILD training data.",
                agent_name=self.__class__.__name__,
                agent_role="prepare",
                input_text=f"signals={len(gild_signals)}",
                output_text="",
                scores={},
                meta={},
                start_time=time.time(),
            )
            # If you build a loader here, do it and fill meta
            if train_loader is not None:
                try:
                    blen = len(train_loader)
                except Exception:
                    blen = "N/A"
            else:
                blen = "N/A"

            step.output_text = f"Prepared signals={len(gild_signals)}, batches={blen}"
            step.end_time = time.time()
            step.duration = (step.end_time - step.start_time)
            step.status = "completed"
            plan_trace = self._repo.add_step(plan_trace, step)
            self.logger.log("GILDDataPrepared", {"num_signals": len(gild_signals), "batches": blen})

        except Exception as e:
            self._append_error_step(plan_trace, pipeline_run_id, trace_id, "data_preparation_error", str(e))
            return self._fail_trace(plan_trace, context, f"Failed during data prep: {e}")

        # ---------- 3) Training Loop (Pi head only) ----------
        model = context.get("sicql_model")
        if not model or not hasattr(model, "pi_head") or model.pi_head is None:
            err = "SICQL model or pi_head missing in context."
            self._append_error_step(plan_trace, pipeline_run_id, trace_id, "model_missing", err)
            return self._fail_trace(plan_trace, context, err)

        pi_head: nn.Module = model.pi_head
        # Freeze everything except pi_head
        try:
            for p in model.parameters():
                p.requires_grad = False
            for p in pi_head.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(pi_head.parameters(), lr=float(self.lr))
        except Exception as e:
            self._append_error_step(plan_trace, pipeline_run_id, trace_id, "optimizer_error", str(e))
            return self._fail_trace(plan_trace, context, f"Optimizer init failed: {e}")

        epoch_losses: List[float] = []
        try:
            start_step = ExecutionStep(
                step_id=f"{trace_id}-train-start",
                pipeline_run_id=pipeline_run_id,
                step_order=2,
                step_type="training_start",
                description="Start GILD policy update loop.",
                agent_name=self.__class__.__name__,
                agent_role="train",
                input_text=f"epochs={self.epochs}, lr={self.lr}",
                output_text="Beginning training",
                scores={},
                meta={
                    "frozen_params": int(sum(not p.requires_grad for p in model.parameters())),
                    "trainable_params_pi": int(sum(p.numel() for p in pi_head.parameters())),
                },
                start_time=time.time(),
                status="completed",
                end_time=time.time(),
                duration=0.0,
            )
            start_step.duration = start_step.end_time - start_step.start_time
            plan_trace = self._repo.add_step(plan_trace, start_step)

            # Optional: no loader, skip training gracefully
            if not train_loader:
                self.logger.log("GILDTrainingSkipped", {"reason": "no_train_loader"})
            else:
                for epoch in range(int(self.epochs)):
                    model.train()
                    total_pi_loss = 0.0
                    num_batches = 0

                    for batch in train_loader:
                        # Expect these keys; adapt to your pipeline
                        ctx_emb = batch["context_emb"].to(self.device)
                        doc_emb = batch["doc_emb"].to(self.device)
                        expert_score = batch.get("expert_score")  # optional depending on your setup
                        if expert_score is not None and hasattr(expert_score, "to"):
                            expert_score = expert_score.to(self.device)

                        with torch.no_grad():
                            outputs: Dict[str, Any] = model(ctx_emb, doc_emb)
                            current_pi_logits = outputs.get("action_logits")
                            zsa = outputs.get("zsa", ctx_emb + doc_emb)

                        predicted_pi_logits = pi_head(zsa)

                        # NOTE: Placeholder loss; replace with proper AWR:
                        pi_loss = F.mse_loss(predicted_pi_logits, current_pi_logits.detach())

                        optimizer.zero_grad()
                        pi_loss.backward()
                        optimizer.step()

                        total_pi_loss += float(pi_loss.item())
                        num_batches += 1

                    avg_epoch_loss = (total_pi_loss / max(1, num_batches))
                    epoch_losses.append(avg_epoch_loss)
                    self.logger.log("GILDEpochCompleted", {"epoch": epoch, "avg_loss": avg_epoch_loss})

                    epoch_step = ExecutionStep(
                        step_id=f"{trace_id}-epoch-{epoch}",
                        pipeline_run_id=pipeline_run_id,
                        step_order=3 + epoch,
                        step_type="training_epoch",
                        description=f"Completed epoch {epoch}",
                        agent_name=self.__class__.__name__,
                        agent_role="train",
                        input_text=None,
                        output_text=f"avg_pi_loss={avg_epoch_loss:.6f}",
                        scores={},
                        meta={"epoch": epoch, "avg_loss": avg_epoch_loss},
                        start_time=time.time(),
                        status="completed",
                        end_time=time.time(),
                        duration=0.0,
                    )
                    epoch_step.duration = epoch_step.end_time - epoch_step.start_time
                    plan_trace = self._repo.add_step(plan_trace, epoch_step)

            final_avg_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
            end_step = ExecutionStep(
                step_id=f"{trace_id}-train-end",
                pipeline_run_id=pipeline_run_id,
                step_order=3 + int(self.epochs),
                step_type="training_end",
                description="GILD training loop completed.",
                agent_name=self.__class__.__name__,
                agent_role="train",
                input_text=None,
                output_text=f"final_avg_pi_loss={final_avg_loss:.6f}",
                scores={},
                meta={"final_avg_loss": final_avg_loss, "epochs_run": int(self.epochs)},
                start_time=time.time(),
                status="completed",
                end_time=time.time(),
                duration=0.0,
            )
            end_step.duration = end_step.end_time - end_step.start_time
            plan_trace = self._repo.add_step(plan_trace, end_step)

        except Exception as e:
            self._append_error_step(plan_trace, pipeline_run_id, trace_id, "training_error", f"{e}\n{traceback.format_exc()}")
            return self._fail_trace(plan_trace, context, f"Failed during training: {e}")

        # ---------- 4) Epistemic Quality + finalize ----------
        max_expected_loss = 0.1  # tune for your scale
        final_avg_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        normalized_loss_quality = max(0.0, min(1.0, 1.0 - (final_avg_loss / max_expected_loss)))

        plan_trace.final_output_text = (
            f"GILD run complete. Final avg Pi loss={final_avg_loss:.6f}; "
            f"proxy_epistemic_quality={normalized_loss_quality:.4f}"
        )
        plan_trace.meta = dict(plan_trace.meta or {})
        plan_trace.meta.update({
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "final_metrics": {
                "final_avg_loss": final_avg_loss,
                "proxy_epistemic_quality": normalized_loss_quality,
                "epochs_run": int(self.epochs),
            }
        })
        plan_trace.status = "completed"
        # If your dataclass includes these fields, set them; otherwise omit:
        try:
            # Some versions of PlanTrace include these fields:
            plan_trace.target_epistemic_quality = normalized_loss_quality  # type: ignore[attr-defined]
            plan_trace.target_epistemic_quality_source = "proxy_final_loss_normalized"  # type: ignore[attr-defined]
        except Exception:
            pass

        plan_trace = self._repo.upsert_trace(plan_trace)
        self.logger.log("GILDProcessTraceFinalized", {
            "trace_id": plan_trace.trace_id,
            "epistemic_quality": normalized_loss_quality,
            "final_loss": final_avg_loss
        })

        # ---------- 5) Context out ----------
        context["gild_status"] = "completed"
        context["gild_final_loss"] = final_avg_loss
        context["gild_model_updated"] = True
        context["gild_trace_id"] = plan_trace.trace_id
        context["gild_epistemic_quality"] = normalized_loss_quality
        return context

    # -------------------- helpers --------------------

    def _append_error_step(self, plan_trace: PlanTrace, run_id: str | int, trace_id: str, err_type: str, msg: str):
        try:
            step = ExecutionStep(
                step_id=f"{trace_id}-{err_type}",
                pipeline_run_id=run_id,
                step_order=99,
                step_type=err_type,
                description=f"Error: {err_type}",
                agent_name=self.__class__.__name__,
                agent_role="error",
                input_text=None,
                output_text=str(msg),
                scores={},
                meta={},
                start_time=time.time(),
                status="failed",
                end_time=time.time(),
                duration=0.0,
            )
            self._repo.add_step(plan_trace, step)
        except Exception as e:
            self.logger.log("GILDTraceAppendErrorStepFailed", {"error": str(e)})

    def _fail_trace(self, plan_trace: PlanTrace, context: dict, reason: str) -> dict:
        try:
            plan_trace.final_output_text = str(reason)
            plan_trace.status = "failed"
            plan_trace.meta = dict(plan_trace.meta or {})
            plan_trace.meta["completed_at"] = datetime.utcnow().isoformat() + "Z"
            self._repo.upsert_trace(plan_trace)
        except Exception as e:
            self.logger.log("GILDTraceFailPersistError", {"error": str(e)})

        context["gild_status"] = "failed"
        context["gild_error"] = reason
        return context
