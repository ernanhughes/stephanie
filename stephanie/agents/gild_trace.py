# Assume this code is in a file like:
# stephanie/agents/gild/gild_trainer_with_trace.py (or similar path)

import time
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.execution_step_store import ExecutionStepStore
from stephanie.memory.plan_trace_store import PlanTraceStore
from stephanie.models.plan_trace import ExecutionStepORM, PlanTraceORM


class GILDTraceAgent(BaseAgent): 
    """
    GILD Trainer that not only performs policy refinement but also records
    its own process and outcome as a PlanTrace for self-analysis and improvement.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # --- Initialize Database Stores ---
        self.plan_trace_store: PlanTraceStore = PlanTraceStore(memory.session, logger)
        self.execution_step_store: ExecutionStepStore = ExecutionStepStore(memory.session, logger)
        # --- GILD Specific Config (example, adjust as needed) ---
        self.beta = cfg.get("gild_beta", 1.0)
        self.epochs = cfg.get("gild_epochs", 5)
        self.batch_size = cfg.get("gild_batch_size", 32)
        # Add other GILD hyperparameters as needed

    async def run(self, context: dict) -> dict:
        """
        Executes the GILD training process and records it as a PlanTrace.
        """ 
        # --- 1. Initialize GILD Process Trace ---
        gild_trace_orm = None
        gild_step_order_counter = 1
        goal_id = context.get("goal_id") # Assuming goal_id is passed
        goal_text = context.get("goal_text", "Unknown Goal") # Get goal text if available
        dimension = context.get("dimension", "Unknown Dimension") # Get dimension being updated
        expert_scorer = context.get("expert_scorer", "Unknown Expert") # e.g., "llm", "hrm"

        try:
            trace_id = f"gild_trace_{int(time.time() * 1000)}_{hash(str(context)) % 10000}" # Simple unique ID
            gild_trace_orm = PlanTraceORM(
                trace_id=trace_id,
                goal_id=goal_id,
                goal_text=goal_text[:1000], # Truncate if very long
                plan_signature=f"GILD_SICQL_Pi_Head_Update_{dimension}_v1", # Descriptive signature
                input_data={ # Store relevant GILD config and context
                    "gild_config": {k: v for k, v in self.cfg.items() if k.startswith("gild_")},
                    "expert_scorer": expert_scorer,
                    "dimension": dimension,
                    # Add other relevant context data if needed
                },
                final_output_text="", # Will be updated later
                target_epistemic_quality=None, # Will be calculated and set later
                target_epistemic_quality_source=None, # Will be set later
                meta={
                    "agent_name": self.__class__.__name__,
                    "started_at": datetime.now().isoformat() + 'Z'
                }
            )
            # Save the initial trace ORM to get its ID
            self.plan_trace_store.session.add(gild_trace_orm)
            self.plan_trace_store.session.flush() # Get ID without committing
            gild_trace_db_id = gild_trace_orm.id
            self.logger.log("GILDProcessTraceStarted", {"trace_id": trace_id, "db_id": gild_trace_db_id})

        except Exception as e:
            self.logger.log("GILDProcessTraceInitError", {"error": str(e)})
            # If trace init fails, log it but continue GILD training without tracing
            gild_trace_orm = None
            gild_trace_db_id = None


        # --- 2. Log Execution Step: Data Preparation ---
        try:
            if gild_trace_orm:
                data_prep_step = ExecutionStepORM(
                    plan_trace_id=gild_trace_db_id,
                    step_order=gild_step_order_counter,
                    step_id=f"{trace_id}_step_{gild_step_order_counter}",
                    description="Load and prepare GILD training data.",
                    output_text="", # Will populate after data loading
                    meta={} # Add data loading stats here later if needed
                )
                self.execution_step_store.insert(data_prep_step)
                gild_step_order_counter += 1

            # --- Original GILD Data Loading Logic (Conceptual) ---
            # Assume context contains 'gild_signals' or similar data structure
            gild_signals = context.get("gild_signals", [])
            # Process signals into DataLoader
            # This involves extracting embeddings, expert scores, calculating advantages, creating weights
            # train_loader = self._prepare_gild_dataloader(gild_signals) # Implement this helper
            # For this example, let's assume train_loader is ready
            train_loader = context.get("gild_dataloader") # Placeholder
            
            if gild_trace_orm:
                 # Update the data prep step with outcome
                 # Find the step ORM (assuming you have a way to get it, or re-query)
                 # For simplicity here, assume we just update the text
                 data_prep_step.output_text = f"Loaded {len(gild_signals)} training signals. Created DataLoader with {len(train_loader) if train_loader else 'N/A'} batches."
                 # In a real scenario, you'd update the ORM object and commit
                 self.execution_step_store.session.commit() # Or use store method if available

            self.logger.log("GILDDataPrepared", {"num_signals": len(gild_signals)})

        except Exception as e:
             self.logger.log("GILDDataPreparationError", {"error": str(e)})
             if gild_trace_orm:
                 # Log error step
                 error_step = ExecutionStepORM(
                     plan_trace_id=gild_trace_db_id,
                     step_order=gild_step_order_counter,
                     step_id=f"{trace_id}_step_{gild_step_order_counter}",
                     description="Error during data preparation.",
                     output_text=f"Error: {str(e)}",
                     meta={}
                 )
                 self.execution_step_store.insert(error_step)
                 gild_step_order_counter += 1
             # Decide if to continue or fail based on error severity
             # For now, let's assume critical and return
             context["gild_status"] = "failed"
             context["gild_error"] = str(e)
             if gild_trace_orm:
                 gild_trace_orm.final_output_text = f"Failed during data prep: {e}"
                 gild_trace_orm.meta["completed_at"] = datetime.now().isoformat() + 'Z'
                 self.plan_trace_store.session.commit()
             return context


        # --- 3. GILD Training Loop ---
        model = context.get("sicql_model") # Assume the model is passed in context
        pi_head = model.pi_head # Assume access to the specific head being trained
        optimizer = torch.optim.AdamW(pi_head.parameters(), lr=self.cfg.get("gild_lr", 1e-4))
        
        if not model or not pi_head:
            error_msg = "SICQL model or Pi head not provided in context."
            self.logger.log("GILDModelError", {"error": error_msg})
            context["gild_status"] = "failed"
            context["gild_error"] = error_msg
            if gild_trace_orm:
                gild_trace_orm.final_output_text = error_msg
                gild_trace_orm.meta["completed_at"] = datetime.now().isoformat() + 'Z'
                self.plan_trace_store.session.commit()
            return context

        # Freeze other parts of the model
        for param in model.parameters():
            param.requires_grad = False
        for param in pi_head.parameters():
            param.requires_grad = True

        self.logger.log("GILDTrainingStarted", {"epochs": self.epochs, "batches": len(train_loader) if train_loader else 'N/A'})
        
        try:
            if gild_trace_orm:
                training_start_step = ExecutionStepORM(
                    plan_trace_id=gild_trace_db_id,
                    step_order=gild_step_order_counter,
                    step_id=f"{trace_id}_step_{gild_step_order_counter}",
                    description="Start GILD policy update loop.",
                    output_text=f"Beginning training for {self.epochs} epochs.",
                    meta={"frozen_params": sum(not p.requires_grad for p in model.parameters()),
                          "trainable_params_pi": sum(p.numel() for p in pi_head.parameters())}
                )
                self.execution_step_store.insert(training_start_step)
                gild_step_order_counter += 1

            # --- Training Statistics Tracking ---
            epoch_losses = []
            # Training loop
            for epoch in range(self.epochs):
                model.train()
                total_pi_loss = 0.0
                num_batches = 0
                # Log start of epoch step? (Optional, might be too granular)
                
                if train_loader: # Check if loader exists
                    for batch_idx, batch in enumerate(train_loader):
                        # Move data to device
                        # Assume batch structure from your data prep
                        ctx_emb = batch["context_emb"].to(self.device)
                        doc_emb = batch["doc_emb"].to(self.device)
                        expert_score = batch["expert_score"].to(self.device) # LLM or HRM score
                        # Get current SICQL prediction (Q, V, Pi)
                        with torch.no_grad(): # Don't need gradients for base model parts
                             outputs = model(ctx_emb, doc_emb)
                             current_q = outputs["q_value"]
                             current_v = outputs["state_value"]
                             current_pi_logits = outputs["action_logits"]

                        # Calculate advantage (expert - V or Q - V depending on setup)
                        # Assuming advantage is pre-calculated and passed, or calculate here
                        # advantage = expert_score - current_v # Example
                        advantage = batch.get("advantage", torch.zeros_like(expert_score)).to(self.device)
                        
                        # Calculate weights
                        weights = torch.exp(self.beta * advantage)
                        weights = weights / weights.sum() # Normalize if needed (depends on AWR variant)
                        
                        # --- GILD Pi Loss (Advantage Weighted Regression) ---
                        # Pass through Pi head only (it's the only part with grad enabled)
                        # Need to ensure input to pi_head matches its expected input
                        # This might be z_context, or a combination, depending on your model
                        # Let's assume zsa (state-action rep) is needed, derived from ctx_emb, doc_emb
                        # You might have an encoder or way to get zsa
                        # zsa = model.encoder(ctx_emb, doc_emb) # Example
                        # For now, assume zsa or equivalent input is available or derivable
                        # Let's assume the model's forward also returns zsa or it's accessible
                        # If not, you need to calculate it here based on your SICQL model structure.
                        # Placeholder: Assume zsa is part of outputs or can be derived
                        zsa = outputs.get("zsa", ctx_emb + doc_emb) # Simplified placeholder
                        
                        predicted_pi_logits = pi_head(zsa) # Get new logits from *trainable* pi_head
                        
                        # Calculate AWR loss
                        # Negative log prob of action * weight
                        # Assume 'action' or target policy is implicit in expert_score/advantage calculation
                        # Or, if you have expert actions, use them.
                        # Simplified version: treat expert_score as a target for the policy distribution
                        # More accurately, you'd use the expert actions or derive a target distribution.
                        # Let's assume a simplified MSE loss between logits for illustration.
                        # A proper AWR would involve log_prob of expert actions.
                        pi_loss = F.mse_loss(predicted_pi_logits, current_pi_logits.detach()) # Example, likely incorrect AWR
                        # Correct AWR usually involves: -log_prob(expert_action) * weight
                        # You need to adapt this based on your exact GILD implementation.
                        
                        # Apply weights (importance sampling)
                        # This often involves summing weighted losses across the batch
                        # weighted_loss = (weights * pi_loss).mean() # Example approach
                        # Or, if pi_loss is already summed/averaged, apply weight differently.
                        # Let's assume pi_loss is calculated per sample and needs weighting.
                        # Reshape weights if necessary
                        # weights = weights.view(-1, 1) if weights.dim() == 1 else weights
                        # weighted_losses = weights * pi_loss # Element-wise if pi_loss is per sample
                        # final_pi_loss = weighted_losses.sum() # Or mean?
                        # The exact weighting depends on your AWR implementation details.
                        # For this draft, let's use a simple average loss placeholder.
                        final_pi_loss = pi_loss.mean() # Placeholder
                        
                        # --- Backward and Optimize ---
                        optimizer.zero_grad()
                        final_pi_loss.backward()
                        optimizer.step()
                        
                        total_pi_loss += final_pi_loss.item()
                        num_batches += 1
                        
                        # Optional: Log batch loss step? (Very granular)
                    
                    avg_epoch_loss = total_pi_loss / num_batches if num_batches > 0 else 0.0
                    epoch_losses.append(avg_epoch_loss)
                    self.logger.log("GILDEpochCompleted", {"epoch": epoch, "avg_loss": avg_epoch_loss})
                    
                    # Log end of epoch step? (Optional)
                    if gild_trace_orm:
                         epoch_step = ExecutionStepORM(
                             plan_trace_id=gild_trace_db_id,
                             step_order=gild_step_order_counter,
                             step_id=f"{trace_id}_step_{gild_step_order_counter}",
                             description=f"Completed GILD training epoch {epoch}.",
                             output_text=f"Average Pi loss: {avg_epoch_loss:.6f}",
                             meta={"epoch": epoch, "avg_loss": avg_epoch_loss}
                         )
                         self.execution_step_store.insert(epoch_step)
                         gild_step_order_counter += 1

            # --- Finalize Training ---
            final_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            self.logger.log("GILDTrainingCompleted", {"final_avg_loss": final_avg_loss})
            
            if gild_trace_orm:
                training_end_step = ExecutionStepORM(
                    plan_trace_id=gild_trace_db_id,
                    step_order=gild_step_order_counter,
                    step_id=f"{trace_id}_step_{gild_step_order_counter}",
                    description="GILD policy update loop completed.",
                    output_text=f"Training finished. Final average Pi loss: {final_avg_loss:.6f}",
                    meta={"final_avg_loss": final_avg_loss, "epochs_run": self.epochs}
                )
                self.execution_step_store.insert(training_end_step)
                gild_step_order_counter += 1

        except Exception as e:
             self.logger.log("GILDTrainingLoopError", {"error": str(e), "traceback": traceback.format_exc()})
             if gild_trace_orm:
                 error_step = ExecutionStepORM(
                     plan_trace_id=gild_trace_db_id,
                     step_order=gild_step_order_counter,
                     step_id=f"{trace_id}_step_{gild_step_order_counter}",
                     description="Error during GILD training loop.",
                     output_text=f"Error: {str(e)}",
                     meta={}
                 )
                 self.execution_step_store.insert(error_step)
                 gild_step_order_counter += 1
             # Handle training error
             context["gild_status"] = "failed_training"
             context["gild_error"] = str(e)
             if gild_trace_orm:
                 gild_trace_orm.final_output_text = f"Failed during training: {e}"
                 gild_trace_orm.meta["completed_at"] = datetime.now().isoformat() + 'Z'
                 self.plan_trace_store.session.commit()
             return context


        # --- 4. Assign Epistemic Quality and Finalize Trace ---
        try:
            # --- Calculate Proxy Epistemic Quality ---
            # Example: 1.0 - normalized_loss (simple proxy)
            # You need a reasonable range for loss. Assume 0.0 to 0.1 is good (1.0 to 0.0 quality)
            # Adjust normalization factor as needed based on observed loss ranges.
            max_expected_loss = 0.1
            normalized_loss_quality = max(0.0, min(1.0, 1.0 - (final_avg_loss / max_expected_loss)))
            
            # Other proxies could be: policy change magnitude, improvement on a small val set, etc.
            
            if gild_trace_orm:
                gild_trace_orm.target_epistemic_quality = normalized_loss_quality
                gild_trace_orm.target_epistemic_quality_source = "proxy_final_loss_normalized"
                gild_trace_orm.final_output_text = f"GILD run completed successfully. Final average Pi loss: {final_avg_loss:.6f}. Assigned proxy epistemic quality: {normalized_loss_quality:.4f}."
                gild_trace_orm.meta["completed_at"] = datetime.now().isoformat() + 'Z'
                gild_trace_orm.meta["final_metrics"] = {
                    "final_avg_loss": final_avg_loss,
                    "proxy_epistemic_quality": normalized_loss_quality,
                    "epochs_run": self.epochs
                }
                # Commit all changes to the database
                self.plan_trace_store.session.commit()
                self.logger.log("GILDProcessTraceFinalized", {
                    "trace_id": gild_trace_orm.trace_id,
                    "db_id": gild_trace_db_id,
                    "epistemic_quality": normalized_loss_quality,
                    "final_loss": final_avg_loss
                })

        except Exception as e:
             self.logger.log("GILDProcessTraceFinalizationError", {"error": str(e)})
             # Even if finalization fails, the training might have succeeded.
             # Log the error but don't necessarily fail the whole context update.
             if gild_trace_orm:
                 gild_trace_orm.final_output_text += f" [Trace Finalization Error: {e}]"
                 gild_trace_orm.meta["trace_finalization_error"] = str(e)
                 self.plan_trace_store.session.commit() # Try to commit error info

        # --- 5. Update Context and Return ---
        # Indicate success in context
        context["gild_status"] = "completed"
        context["gild_final_loss"] = final_avg_loss
        context["gild_model_updated"] = True # Or reference to the updated model
        if gild_trace_orm:
            context["gild_trace_id"] = gild_trace_orm.trace_id
            context["gild_trace_db_id"] = gild_trace_db_id
            context["gild_epistemic_quality"] = normalized_loss_quality

        self.logger.log("GILDTrainerWithTraceCompleted", {
            "status": context["gild_status"],
            "final_loss": context.get("gild_final_loss"),
            "trace_recorded": gild_trace_orm is not None
        })
        
        return context

# Note: This is a conceptual adaptation. You will need to:
# 1. Fill in the exact data loading/preparation logic (`_prepare_gild_dataloader` or direct use of `gild_signals`).
# 2. Correctly implement the Advantage Weighted Regression (AWR) loss calculation based on your GILD setup
#    (this often involves expert actions or distributions, not just MSE of logits).
# 3. Ensure correct access to the SICQL model components (encoder, pi_head) and their inputs/outputs.
# 4. Adjust database session handling (commits, rollbacks on error) according to your memory/store patterns.
# 5. Potentially add more detailed logging steps within the training loop if needed for trace analysis.
# 6. Handle potential errors and edge cases more robustly.
# 7. Ensure imports point to the correct locations in your project structure.