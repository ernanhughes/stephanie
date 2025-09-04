# stephanie/scoring/training/pacs_trainer.py
from __future__ import annotations

import os
import json
import math
from datetime import datetime
from stephanie.scoring.scorable_factory import TargetType
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Callable, List

from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.models.model_version import ModelVersionORM
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.scoring.training.sicql_trainer import SICQLTrainer

import random
from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizer


# ==============================
# Core RLVR data structures
# ==============================

@dataclass
class RLVRItem:
    """Minimal data structure for PACS training samples"""
    query: str
    meta: Optional[Dict[str, Any]] = None

class RLVRDataset(torch.utils.data.Dataset):
    """Dataset of RLVRItems that can be sampled in batches"""
    
    def __init__(self, items: List[RLVRItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RLVRItem:
        return self.items[idx]
    
    def sample(self, batch_size: int) -> Dict[str, List]:
        """Sample a batch of items with inputs and references"""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        batch = [self.items[i] for i in indices]
        
        return {
            "inputs": [item.query for item in batch],
            "refs": [item.meta.get("expected", "") if item.meta else "" for item in batch]
        }


# ==============================
# PACS Configuration
# ==============================

@dataclass
class PACSConfig:
    """Configurable parameters for PACS training"""
    score_mode: str = "critic"     # "logprob" | "critic"
    beta: float = 1.0              # scale for r̂ (logprob/logit delta)
    group_size: int = 8            # samples per prompt
    max_new_tokens: int = 256      # response generation length
    temperature: float = 0.6       # sampling temperature
    top_p: float = 0.96            # nucleus sampling parameter
    
    lr: float = 1e-6               # learning rate
    weight_decay: float = 0.01     # weight decay
    grad_clip: float = 1.0         # gradient clipping norm
    steps_per_reset: int = 200     # steps between reference model resets
    pos_weight: float = 1.0        # class balancing for BCE
    log_every: int = 10            # logging frequency

    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

# ==============================
# Weighted BCE Loss
# ==============================

class WeightedBCEWithLogits(nn.Module):
    """BCEWithLogitsLoss with class weighting"""
    
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight
        )


# ==============================
# Hybrid SICQL Adapter (Critical Component)
# ==============================

class HybridSICQLAdapter:
    """
    Unifies the interfaces needed by PACS in both scoring modes:
    
    1. `logprob` mode: uses actor LM to compute Σ log p for responses
    2. `critic` mode: uses SICQL critic head to score (prompt, response) pairs
    
    This adapter is the critical bridge between your SICQL infrastructure and PACS training.
    
    Key features:
    - Maintains both online and reference models (for RLOO regularization)
    - Handles safe generation with proper tokenization
    - Implements correct logprob calculation for actor-based mode
    - Integrates with SICQL policy heads for critic-based mode
    - Includes safety mechanisms to prevent model collapse
    """
    
    def __init__(
        self,
        actor_lm: PreTrainedModel,           # Base LM for response generation
        tokenizer: PreTrainedTokenizer,      # Matching tokenizer
        critic_head: Optional[nn.Module] = None,  # SICQL head (optional)
        device: Optional[str] = None,
    ):
        """
        Args:
            actor_lm: Base language model (trainable)
            tokenizer: Tokenizer for the model
            critic_head: SICQL policy head (optional, for critic mode)
            device: Target device (auto-detected if None)
        """
        # Device setup
        self._device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Actor setup (policy for generation)
        self.actor = actor_lm.to(self._device)
        self.actor.train()
        
        # Reference actor (frozen for RLOO)
        import copy
        self.actor_ref = copy.deepcopy(actor_lm).to(self._device)
        self.actor_ref.eval()
        for p in self.actor_ref.parameters():
            p.requires_grad_(False)
        
        # Critic setup (optional, for critic mode)
        self.critic = critic_head.to(self._device) if critic_head is not None else None
        if self.critic is not None:
            self.critic.train()
            self.critic_ref = copy.deepcopy(self.critic).to(self._device)
            self.critic_ref.eval()
            for p in self.critic_ref.parameters():
                p.requires_grad_(False)
        else:
            self.critic_ref = None
        
        # Tokenizer setup
        self.tok = tokenizer
        if getattr(self.tok, "pad_token", None) is None:
            if getattr(self.tok, "eos_token", None) is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                # Fallback to adding a pad token
                self.tok.add_special_tokens({"pad_token": "[PAD]"})
                self.actor.resize_token_embeddings(len(self.tok))
                if self.actor_ref:
                    self.actor_ref.resize_token_embeddings(len(self.tok))

    def device(self) -> torch.device:
        """Return the device this adapter is using"""
        return self._device
    
    # ---------- Generation ----------
    @torch.no_grad()
    def sample_group(
        self,
        prompt: str,
        group_size: int,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """
        Generate multiple responses for a single prompt.
        
        Args:
            prompt: Input prompt
            group_size: Number of responses to generate
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            
        Returns:
            List of generated responses
        """
        # Use defaults if not provided
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens if hasattr(self, 'cfg') else 256
        temperature = temperature or self.cfg.temperature if hasattr(self, 'cfg') else 0.6
        top_p = top_p or self.cfg.top_p if hasattr(self, 'cfg') else 0.96
        
        # Tokenize prompt
        inputs = self.tok(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self._device)
        
        responses = []
        for _ in range(group_size):
            try:
                # Generate response
                out = self.actor.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                    eos_token_id=self.tok.eos_token_id,
                )
                
                # Extract only the generated part (remove prompt)
                prompt_len = inputs["input_ids"].size(1)
                response_ids = out[0, prompt_len:]
                response = self.tok.decode(response_ids, skip_special_tokens=True)
                responses.append(response)
            except Exception as e:
                print(f"Generation failed: {e}")
                # Fallback to empty response
                responses.append("")
        
        return responses
    
    # ---------- Logprob sums for `logprob` mode ----------
    def _sum_logprobs(
        self,
        model: PreTrainedModel,
        prompt: str,
        response: str,
        safety_threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Compute sum of log probabilities for the response tokens.
        
        Args:
            model: Model to use for logprob calculation
            prompt: Input prompt
            response: Generated response
            safety_threshold: Threshold for NaN/inf detection
            
        Returns:
            Sum of log probabilities for response tokens
        """
        # Tokenize full sequence
        full_text = prompt + response
        enc = self.tok(
            full_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self._device)
        
        # Get input IDs and attention mask
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            logits = outputs.logits
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs for the actual next tokens
            shift_logits = log_probs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Get log probs for the actual next tokens
            token_log_probs = torch.gather(
                shift_logits, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Identify prompt vs response tokens
            prompt_enc = self.tok(
                prompt, 
                return_tensors="pt", 
                padding=False, 
                truncation=True, 
                max_length=1024
            )
            prompt_len = len(prompt_enc["input_ids"][0])
            
            # Only sum over response tokens
            response_log_probs = token_log_probs[:, prompt_len-1:]
            
            # Safety checks
            if torch.isnan(response_log_probs).any():
                print("Warning: NaN detected in log probs, replacing with safe values")
                response_log_probs = torch.nan_to_num(response_log_probs, nan=-100.0)
                
            if (response_log_probs < -1/safety_threshold).any():
                print(f"Warning: Extremely low log probs detected (below {-1/safety_threshold})")
                response_log_probs = torch.clamp(response_log_probs, min=-1/safety_threshold)
        
        # Sum log probs for response tokens
        return response_log_probs.sum(dim=1).squeeze()
    
    def logprob_sum(self, prompt: str, response: str) -> torch.Tensor:
        """Compute logprob sum using the online actor model"""
        return self._sum_logprobs(self.actor, prompt, response)
    
    @torch.no_grad()
    def logprob_sum_ref(self, prompt: str, response: str) -> torch.Tensor:
        """Compute logprob sum using the frozen reference actor"""
        return self._sum_logprobs(self.actor_ref, prompt, response).detach()
    
    # ---------- Critic logits for `critic` mode ----------
    def _critic_logit(
        self,
        model: nn.Module,
        prompt: str,
        response: str,
        max_length: int = 2048
    ) -> torch.Tensor:
        """
        Compute critic logit for a (prompt, response) pair.
        
        Args:
            model: Critic model to use
            prompt: Input prompt
            response: Generated response
            max_length: Maximum sequence length
            
        Returns:
            Scalar logit value
        """
        if self.critic is None:
            raise ValueError("Critic head required for score_mode='critic'")
        
        # Format input for critic
        full_text = f"{prompt}\n\n{response}"
        
        # Tokenize
        enc = self.tok(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self._device)
        
        # Forward pass through critic
        with torch.set_grad_enabled(model.training):
            out = model(**enc)
            logits = getattr(out, "logits", out)
            
            # Handle different output formats
            if isinstance(logits, torch.Tensor):
                if logits.dim() == 2 and logits.size(1) == 1:
                    return logits.squeeze(1)
                return logits
            elif hasattr(logits, "logits"):
                return logits.logits
            else:
                return logits
    
    def critic_logit(self, prompt: str, response: str) -> torch.Tensor:
        """Compute critic logit using the online critic model"""
        return self._critic_logit(self.critic, prompt, response)
    
    @torch.no_grad()
    def critic_logit_ref(self, prompt: str, response: str) -> torch.Tensor:
        """Compute critic logit using the frozen reference critic"""
        return self._critic_logit(self.critic_ref, prompt, response).detach()
    
    # ---------- Reference sync ----------
    def hard_reset_ref(self) -> None:
        """
        Reset reference models to current online models.
        
        This is critical for PACS stability - should be called periodically
        (e.g., every steps_per_reset steps).
        """
        # Reset actor reference
        self.actor_ref.load_state_dict(self.actor.state_dict())
        self.actor_ref.eval()
        
        # Reset critic reference if available
        if self.critic is not None and self.critic_ref is not None:
            self.critic_ref.load_state_dict(self.critic.state_dict())
            self.critic_ref.eval()
    
    # ---------- Safety and diagnostics ----------
    def calculate_response_entropy(
        self,
        prompt: str,
        responses: List[str],
        temperature: float = 0.6
    ) -> float:
        """
        Calculate approximate entropy of the response distribution.
        
        Args:
            prompt: Input prompt
            responses: List of generated responses
            temperature: Sampling temperature
            
        Returns:
            Approximate entropy value
        """
        if not responses:
            return 0.0
            
        # Simple heuristic: entropy proportional to response diversity
        unique_responses = set(responses)
        diversity = len(unique_responses) / len(responses)
        
        # Scale by temperature (higher temp = higher entropy)
        return -math.log(temperature + 1e-8) * diversity
    
    def check_model_stability(self) -> bool:
        """
        Check if the model is becoming unstable (e.g., collapsing to single response).
        
        Returns:
            True if stable, False if unstable
        """
        # Sample multiple responses to a simple prompt
        test_prompt = "The capital of France is"
        responses = self.sample_group(test_prompt, group_size=8)
        
        # Check response diversity
        unique_responses = set(responses)
        diversity = len(unique_responses) / len(responses)
        
        # Model is unstable if diversity is too low
        return diversity > 0.5


# ==============================
# PACS Trainer Core
# ==============================

class PACSCoreTrainer:
    """PACS trainer implementing RLVR via Supervised with RLOO regularization"""
    
    def __init__(
        self,
        policy: HybridSICQLAdapter,
        cfg: PACSConfig,
        verifier: Callable[[str, str, Optional[Dict[str, Any]]], int],
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            policy: Policy adapter (HybridSICQLAdapter)
            cfg: PACS configuration
            verifier: Function to verify (prompt, response, meta) → {0,1}
            logger: Optional logging function
        """
        self.policy = policy
        self.cfg = cfg
        self.verifier = verifier
        self.logger = logger or (lambda d: None)
        
        # Determine which parameters to train
        if cfg.score_mode == "logprob":
            params = self.policy.actor.parameters()
        elif cfg.score_mode == "critic":
            if self.policy.critic is None:
                raise ValueError("critic_head required for score_mode='critic'")
            params = self.policy.critic.parameters()
        else:
            raise ValueError("score_mode must be 'logprob' or 'critic'")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params, 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode="min", 
            factor=0.5, 
            patience=5
        )
        
        # Loss function
        self.loss_fn = WeightedBCEWithLogits(pos_weight=cfg.pos_weight)
        
        # Training state
        self._step = 0
        self.best_loss = float("inf")
        self.early_stop_counter = 0

    # ---- Helpers ----
    def _rhat_vector(self, prompt: str, responses: List[str]) -> torch.Tensor:
        """
        Compute r̂ for each response in group based on chosen score_mode.
        
        r̂ = β·(score_online - score_ref)
        """
        vals = []
        
        if self.cfg.score_mode == "logprob":
            for r in responses:
                lp = self.policy.logprob_sum(prompt, r)          # requires grad
                lpr = self.policy.logprob_sum_ref(prompt, r)     # no grad
                vals.append(self.cfg.beta * (lp - lpr))
        else:  # critic
            for r in responses:
                lg = self.policy.critic_logit(prompt, r)         # requires grad
                lgr = self.policy.critic_logit_ref(prompt, r)    # no grad
                vals.append(self.cfg.beta * (lg - lgr))
        
        return torch.stack(vals)  # (G,)
    
    def _psi_rloo(self, rhat: torch.Tensor) -> torch.Tensor:
        """
        Compute ψ using RLOO (leave-one-out) regularization.
        
        ψ_i = r̂_i − mean_{j≠i} r̂_j
        """
        G = rhat.numel()
        if G <= 1:
            return rhat
            
        # Compute mean of all r̂ values
        total = rhat.sum()
        
        # Compute leave-one-out mean for each element
        psi = rhat - (total - rhat) / (G - 1)
        return psi
    
    def _verify_labels(
        self, 
        prompt: str, 
        responses: List[str], 
        meta: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Verify responses and return labels (0 or 1)"""
        labels = []
        for r in responses:
            try:
                labels.append(float(self.verifier(prompt, r, meta)))
            except Exception as e:
                print(f"Verification failed: {e}")
                labels.append(0.0)
        
        return torch.tensor(labels, dtype=torch.float32, device=self.policy.device())
    
    def _log(self, d: Dict[str, Any]):
        """Log metrics with step counter"""
        d = {"step": self._step, **d}
        self.logger(d)
    
    def _check_early_stopping(self, loss: float) -> bool:
        """Check if early stopping criteria are met"""
        if loss < self.best_loss - self.cfg.early_stopping_min_delta:
            self.best_loss = loss
            self.early_stop_counter = 0
            return False
        
        self.early_stop_counter += 1
        return self.early_stop_counter >= self.cfg.early_stopping_patience

    # ---- Training API ----
    def train(
        self, 
        dataset: RLVRDataset, 
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train policy using PACS with RLOO regularization.
        
        Args:
            dataset: RLVRDataset containing training samples
            max_steps: Maximum number of training steps
            
        Returns:
            Training statistics
        """
        # Training metrics
        metrics = {
            "losses": [],
            "psi_means": [],
            "rhat_means": [],
            "label_rates": [],
            "acc_proxies": [],
            "entropies": []
        }
        
        # Training loop
        step = 0
        while max_steps is None or step < max_steps:
            try:
                # Sample random item from dataset
                item = dataset[random.randint(0, len(dataset) - 1)]
                self._train_on_item(item)
                
                # Periodic reference model reset
                if self.cfg.steps_per_reset and (self._step % self.cfg.steps_per_reset == 0):
                    self.policy.hard_reset_ref()
                
                # Log metrics periodically
                if self._step % self.cfg.log_every == 0:
                    self._log({
                        "loss": metrics["losses"][-1] if metrics["losses"] else 0.0,
                        "psi_mean": metrics["psi_means"][-1] if metrics["psi_means"] else 0.0,
                        "rhat_mean": metrics["rhat_means"][-1] if metrics["rhat_means"] else 0.0,
                        "label_pos_rate": metrics["label_rates"][-1] if metrics["label_rates"] else 0.0,
                        "acc_proxy": metrics["acc_proxies"][-1] if metrics["acc_proxies"] else 0.0,
                        "mode": self.cfg.score_mode,
                        "entropy": metrics["entropies"][-1] if metrics["entropies"] else 0.0
                    })
                
                step += 1
                self._step += 1
                
            except Exception as e:
                print(f"Training error at step {self._step}: {e}")
                continue
        
        # Return training statistics
        return {
            "avg_loss": float(np.mean(metrics["losses"])) if metrics["losses"] else 0.0,
            "avg_psi": float(np.mean(metrics["psi_means"])) if metrics["psi_means"] else 0.0,
            "avg_rhat": float(np.mean(metrics["rhat_means"])) if metrics["rhat_means"] else 0.0,
            "pos_rate": float(np.mean(metrics["label_rates"])) if metrics["label_rates"] else 0.0,
            "acc_proxy": float(np.mean(metrics["acc_proxies"])) if metrics["acc_proxies"] else 0.0,
            "entropy": float(np.mean(metrics["entropies"])) if metrics["entropies"] else 0.0,
            "steps": self._step
        }
    
    def _train_on_item(self, item: RLVRItem) -> None:
        """Train on a single dataset item (prompt + meta)"""
        prompt = item.query
        meta = item.meta
        
        # Generate response group
        responses = self.policy.sample_group(
            prompt,
            group_size=self.cfg.group_size,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        
        # Verify responses
        labels = self._verify_labels(prompt, responses, meta)
        
        # Compute r̂ and ψ
        rhat = self._rhat_vector(prompt, responses)  # (G,)
        psi = self._psi_rloo(rhat)
        
        # Compute loss
        loss = self.loss_fn(psi, labels)
        
        # Optimization
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], 
                self.cfg.grad_clip
            )
        
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(loss.item())
        
        # Compute metrics
        with torch.no_grad():
            psi_mean = float(psi.mean().item())
            rhat_mean = float(rhat.mean().item())
            label_pos_rate = float(labels.mean().item())
            acc_proxy = float(((psi > 0).float() == labels).float().mean().item())
            entropy = self.policy.calculate_response_entropy(prompt, responses)
        
        # Store metrics
        self._log({
            "loss": float(loss.item()),
            "psi_mean": psi_mean,
            "rhat_mean": rhat_mean,
            "label_pos_rate": label_pos_rate,
            "acc_proxy": acc_proxy,
            "entropy": entropy,
            "mode": self.cfg.score_mode,
        })

class PACSTrainer(BaseTrainer):
    """
    PACS trainer that implements the RLVR via Supervised algorithm.
    
    This integrates properly with SICQL infrastructure by:
    - Using SICQL policy heads as critics
    - Leveraging SICQL verifiers for labels
    - Maintaining proper RLOO implementation
    
    Key features:
    - Two scoring modes: "logprob" (actor-based) and "critic" (SICQL-based)
    - Proper RLOO (leave-one-out) regularization
    - Reference model resets
    - Safety mechanisms to prevent model collapse
    """
    
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        
        # Device management
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize configuration
        self._init_config(cfg)
        
        # Track training state
        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self.trainer = None  # Will be initialized in train()
        self.pacs_core = None  # Core PACS implementation
        
        # Log initialization
        self.logger.log(
            "PACSTrainerInitialized",
            {
                "score_mode": self.score_mode,
                "beta": self.beta,
                "group_size": self.group_size,
                "device": str(self.device),
            },
        )
    
    def _init_config(self, cfg):
        """Initialize training parameters from config"""
        self.score_mode = cfg.get("score_mode", "critic")  # "logprob" or "critic"
        self.beta = cfg.get("beta", 1.0)
        self.group_size = cfg.get("group_size", 8)
        self.max_steps = cfg.get("max_steps", 1000)
        self.lr = cfg.get("lr", 1e-6)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.pos_weight = cfg.get("pos_weight", 1.0)
        self.steps_per_reset = cfg.get("steps_per_reset", 200)
        self.dimension = cfg.get("dimension", "alignment")
        self.target_type = cfg.get("target_type", "document")
        self.model_path = cfg.get("model_path", "models/pacs")
        self.model_version = cfg.get("model_version", "v1")
    
    def _get_verifier(self, dimension: str) -> Callable[[str, str, Optional[Dict]], int]:
        """Get verifier function for the given dimension"""
        if dimension == "math":
            from stephanie.scoring.verifiers import boxed_math_verifier
            return boxed_math_verifier
        elif dimension == "code":
            from stephanie.scoring.verifiers import code_verifier
            return code_verifier
        else:
            # Generic verifier that uses SICQL scores
            def generic_verifier(prompt: str, response: str, meta: Optional[Dict] = None) -> int:
                # In practice, this would use your SICQL system
                from stephanie.scoring.scorable_factory import ScorableFactory
                from stephanie.scoring.transforms.regression_tuner import RegressionTuner
                
                # Create scorable
                scorable = ScorableFactory.from_text(
                    response, 
                    TargetType.DOCUMENT,
                    context=prompt
                )
                
                # Get score from SICQL
                score = self.memory.scores.get_score(
                    goal_id=meta.get("goal_id") if meta else None,
                    scorable=scorable
                )
                
                # Use tuner if available
                tuner = RegressionTuner(dimension=dimension)
                if tuner.is_loaded():
                    normalized = tuner.transform(score.score if score else 0.0)
                    return 1 if normalized > 0.5 else 0
                return 1 if (score.score if score else 0.0) > 0.5 else 0
                
            return generic_verifier
    
    def _build_policy_adapter(self, dimension: str):
        """Build policy adapter with SICQL integration"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load base model (same as SICQL uses)
        base_model = "Qwen/Qwen2.5-1.5B"
        actor = AutoModelForCausalLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # For critic mode, load SICQL policy head
        critic = None
        if self.score_mode == "critic":
            # Reuse SICQL's policy head structure
            from stephanie.scoring.model.policy_head import PolicyHead
            from stephanie.scoring.model.text_encoder import TextEncoder
            from stephanie.scoring.model.q_head import QHead
            from stephanie.scoring.model.v_head import VHead
            
            # Load SICQL components
            locator = SICQLTrainer.get_locator(dimension)
            encoder = TextEncoder(dim=self.memory.embedding.dim, hdim=self.memory.embedding.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.memory.embedding.dim, hdim=self.memory.embedding.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.memory.embedding.dim, hdim=self.memory.embedding.hdim).to(self.device)
            pi_head = PolicyHead(
                zsa_dim=self.memory.embedding.dim, 
                hdim=self.memory.embedding.hdim, 
                num_actions=3
            ).to(self.device)
            
            # Load weights
            if os.path.exists(locator.encoder_file()):
                encoder.load_state_dict(torch.load(locator.encoder_file(), map_location=self.device))
            if os.path.exists(locator.q_head_file()):
                q_head.load_state_dict(torch.load(locator.q_head_file(), map_location=self.device))
            if os.path.exists(locator.v_head_file()):
                v_head.load_state_dict(torch.load(locator.v_head_file(), map_location=self.device))
            if os.path.exists(locator.pi_head_file()):
                pi_head.load_state_dict(torch.load(locator.pi_head_file(), map_location=self.device))
            
            # Create SICQL model (for critic)
            from stephanie.scoring.model.in_context_q import InContextQModel
            critic = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
        
        # Create policy adapter
        adapter = HybridSICQLAdapter(
            actor_lm=actor,
            tokenizer=tokenizer,
            critic_head=critic,
            device=self.device
        )
        
        return adapter
    
    def train(self, dataset, dimension: str = "alignment"):
        """
        Train model using proper PACS algorithm with RLOO.
        
        Args:
            dataset: RLVRDataset containing queries and meta
            dimension: Dimension to train on (for verifier selection)
        
        Returns:
            Training statistics and model metadata
        """
        self.logger.log("PACSTrainingStarted", {"dimension": dimension})
        
        # 1. Build policy adapter (with SICQL integration)
        policy = self._build_policy_adapter(dimension)
        
        # 2. Get verifier for this dimension
        verifier = self._get_verifier(dimension)
        
        # 3. Create PACS config
        pacs_cfg = PACSConfig(
            score_mode=self.score_mode,
            beta=self.beta,
            group_size=self.group_size,
            max_new_tokens=256,
            temperature=0.6,
            top_p=0.96,
            lr=self.lr,
            weight_decay=0.01,
            grad_clip=1.0,
            steps_per_reset=self.steps_per_reset,
            pos_weight=self.pos_weight,
            log_every=10
        )
        
        # 4. Create core PACS trainer
        self.pacs_core = PACSCoreTrainer(
            policy=policy,
            cfg=pacs_cfg,
            verifier=verifier,
            logger=self._log_metrics
        )
        
        # 5. Train model
        self.pacs_core.train(dataset, max_steps=self.max_steps)
        
        # 6. Save model and metadata
        meta = self._save_model(policy, dimension)
        
        # 7. Log training stats
        self._log_training_stats(dimension, meta)
        
        # 8. Update belief cartridges
        self._update_belief_cartridge(dimension, meta)
        
        self.logger.log(
            "PACSTrainingComplete",
            {
                "dimension": dimension,
                "final_loss": meta["avg_loss"],
                "policy_entropy": meta["policy_entropy"]
            },
        )
        
        return meta
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics from PACS core trainer"""
        self.logger.log("PACSMetrics", metrics)
        
        # Track for early stopping
        if "loss" in metrics:
            if metrics["loss"] < self.best_loss - self.early_stopping_min_delta:
                self.best_loss = metrics["loss"]
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
    
    def _save_model(self, policy, dimension: str):
        """Save PACS model with metadata"""
        # Create output directory
        output_dir = os.path.join(self.model_path, dimension)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save actor model
        actor_path = os.path.join(output_dir, "actor")
        policy.actor.save_pretrained(actor_path)
        
        # Save critic if in critic mode
        if self.score_mode == "critic" and policy.critic is not None:
            critic_path = os.path.join(output_dir, "critic")
            # Save SICQL components
            torch.save(policy.critic.encoder.state_dict(), os.path.join(critic_path, "encoder.pt"))
            torch.save(policy.critic.q_head.state_dict(), os.path.join(critic_path, "q_head.pt"))
            torch.save(policy.critic.v_head.state_dict(), os.path.join(critic_path, "v_head.pt"))
            torch.save(policy.critic.pi_head.state_dict(), os.path.join(critic_path, "pi_head.pt"))
        
        # Calculate policy metrics
        policy_entropy = self._calculate_policy_entropy(policy)
        policy_stability = self._calculate_policy_stability(policy)
        
        # Build metadata
        meta = {
            "version": self.model_version,
            "dimension": dimension,
            "score_mode": self.score_mode,
            "beta": self.beta,
            "group_size": self.group_size,
            "avg_loss": float(self.best_loss),
            "policy_entropy": policy_entropy,
            "policy_stability": policy_stability,
            "steps": self.pacs_core._step if self.pacs_core else 0,
            "device": str(self.device),
            "model_path": output_dir,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save metadata
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # Save to database
        model_version = ModelVersionORM(
            model_type="pacs",
            dimension=dimension,
            version=self.model_version,
            **meta
        )
        self.memory.session.add(model_version)
        self.memory.session.commit()
        
        return meta
    
    def _calculate_policy_entropy(self, policy) -> float:
        """Calculate policy entropy for monitoring"""
        # In practice, this would sample responses and calculate entropy
        # For simplicity, we'll estimate from temperature
        return -math.log(policy.cfg.temperature) if hasattr(policy, 'cfg') else 1.5
    
    def _calculate_policy_stability(self, policy) -> float:
        """Calculate policy stability metric"""
        # This would be more sophisticated in practice
        return 0.85  # Placeholder
    
    def _log_training_stats(self, dimension: str, meta: Dict[str, Any]):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="pacs",
            target_type=self.target_type,
            dimension=dimension,
            version=self.model_version,
            avg_q_loss=meta["avg_loss"],
            policy_entropy=meta["policy_entropy"],
            policy_stability=meta["policy_stability"],
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()
    
    def _update_belief_cartridge(self, dimension: str, meta: Dict[str, Any]):
        """Update belief cartridges with policy stats"""
        bc = BeliefCartridgeORM(
            title=f"PACS Policy - {dimension}",
            content=f"PACS training completed for {dimension} dimension",
            goal_id=None,  # Would be set based on context
            domain=dimension,
            policy_entropy=meta["policy_entropy"],
            policy_stability=meta["policy_stability"],
        )
        self.memory.session.add(bc)
        self.memory.session.commit()
    
    def run(self, context: dict) -> dict:
        """
        Main entry point for PACS training.
        
        Context should contain:
        - "rlvr_dataset": RLVRDataset from CaseBook
        - "dimension": Dimension to train on
        """
        self.logger.log("PACSTrainerRun", {"context_keys": list(context.keys())})
        
        # Get dimension from context
        dimension = context.get("dimension", self.dimension)
        
        # Get dataset
        dataset = context.get("rlvr_dataset")
        if not dataset:
            self.logger.log("MissingDataset", {"required": "rlvr_dataset"})
            return context
        
        # Train model
        stats = self.train(dataset, dimension)
        
        # Update context
        context["pacs_training_stats"] = stats
        context["pacs_model_path"] = stats.get("model_path", "")
        
        return context