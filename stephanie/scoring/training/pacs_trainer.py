# stephanie/scoring/training/pacs_trainer.py
from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.model_version import ModelVersionORM
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.scorer.sicql_scorer import SICQLScorer
from stephanie.scoring.training.base_trainer import BaseTrainer

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
    Adapter unifying actor (HF LM) and critic (SICQL InContextQModel).
    
    - Actor: HuggingFace causal LM (text in → text out, uses tokenizer)
    - Critic: SICQL InContextQModel (embeddings in → Q/V/π outputs)
    """

class HybridSICQLAdapter:
    def __init__(self, memory, actor_lm, tokenizer, critic_head=None, device=None):
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.memory = memory

        # --- Actor setup ---
        self.actor = actor_lm.to(self._device)
        self.actor.train()

        # Reference actor (frozen)
        self.actor_ref = copy.deepcopy(actor_lm).to(self._device)
        self.actor_ref.eval()
        for p in self.actor_ref.parameters():
            p.requires_grad_(False)

        # --- Critic setup (supports both SICQL + HuggingFace critics) ---
        self.critic = critic_head.to(self._device) if critic_head is not None else None
        self.critic_ref = None
        self.critic_type = None

        if self.critic is not None:
            # Detect critic type
            if hasattr(self.critic, "q_head") and hasattr(self.critic, "encoder"):
                # Looks like SICQL InContextQModel
                self.critic_type = "sicql"
                self.critic.train()

                self.critic_ref = type(self.critic)(
                    encoder=copy.deepcopy(self.critic.encoder),
                    q_head=copy.deepcopy(self.critic.q_head),
                    v_head=copy.deepcopy(self.critic.v_head),
                    pi_head=copy.deepcopy(self.critic.pi_head),
                    embedding_store=self.memory.embedding,
                    device=self._device,
                ).to(self._device)
                self.critic_ref.load_state_dict(self.critic.state_dict())
                self.critic_ref.eval()
                for p in self.critic_ref.parameters():
                    p.requires_grad_(False)

            else:
                # Assume HuggingFace-style sequence classification model
                self.critic_type = "hf"
                self.critic.train()
                self.critic_ref = copy.deepcopy(self.critic).to(self._device)
                self.critic_ref.eval()
                for p in self.critic_ref.parameters():
                    p.requires_grad_(False)

        # --- Tokenizer setup ---
        self.tok = tokenizer
        if getattr(self.tok, "pad_token", None) is None:
            if getattr(self.tok, "eos_token", None) is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "[PAD]"})
                self.actor.resize_token_embeddings(len(self.tok))
                self.actor_ref.resize_token_embeddings(len(self.tok))

        self.memory = memory

    def device(self):
        return self._device

    # ---------- Actor (HF LM) ----------
    @torch.no_grad()
    def sample_group(self, prompt, group_size, max_new_tokens=256, temperature=0.6, top_p=0.95):
        """Generate multiple responses from the HuggingFace LM."""
        inputs = self.tok(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self._device)

        responses = []
        for _ in range(group_size):
            out = self.actor.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            prompt_len = inputs["input_ids"].size(1)
            response_ids = out[0, prompt_len:]
            response = self.tok.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
        return responses

    def logprob_sum(self, prompt, response):
        """Compute logprob sum with online actor LM."""
        return self._sum_logprobs(self.actor, prompt, response)

    @torch.no_grad()
    def logprob_sum_ref(self, prompt, response):
        """Compute logprob sum with frozen actor reference."""
        return self._sum_logprobs(self.actor_ref, prompt, response).detach()

    def _sum_logprobs(self, model, prompt, response):
        """Sum log probabilities for actor outputs."""
        full_text = prompt + response
        enc = self.tok(full_text, return_tensors="pt", truncation=True, max_length=2048).to(self._device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        shift_logits = log_probs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        token_log_probs = torch.gather(shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        prompt_len = len(self.tok(prompt, return_tensors="pt")["input_ids"][0])
        response_log_probs = token_log_probs[:, prompt_len - 1:]
        return response_log_probs.sum(dim=1).squeeze()

    # ---------- Critic (SICQL InContextQModel) ----------
    def _critic_logit(self, model, prompt, response):
        """Compute Q-value logit from SICQL critic (InContextQModel)."""
        if model is None:
            raise ValueError("Critic head required for critic mode")
        if self.memory is None:
            raise ValueError("Memory with embedding system required for critic")

        # Get embeddings
        prompt_emb_np = self.memory.embedding.get_or_create(prompt)
        response_emb_np = self.memory.embedding.get_or_create(response)
        prompt_emb = torch.tensor(prompt_emb_np, device=self._device, dtype=torch.float32).unsqueeze(0)
        response_emb = torch.tensor(response_emb_np, device=self._device, dtype=torch.float32).unsqueeze(0)

        out = model(prompt_emb, response_emb)
        return out["q_value"].squeeze()

    def critic_logit(self, prompt, response):
        return self._critic_logit(self.critic, prompt, response)

    @torch.no_grad()
    def critic_logit_ref(self, prompt, response):
        return self._critic_logit(self.critic_ref, prompt, response).detach()

    # ---------- Ref sync ----------
    def hard_reset_ref(self):
        """Reset references to current models."""
        self.actor_ref.load_state_dict(self.actor.state_dict())
        self.actor_ref.eval()
        if self.critic and self.critic_ref:
            self.critic_ref.load_state_dict(self.critic.state_dict())
            self.critic_ref.eval()



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
        online_training: bool = False
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
        self.online_training = online_training

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
        # Use BCEWithLogitsLoss to maintain numerical stability. 
        # The paper's Equation 1 shows the loss is computed with σ(ψ), 
        # but PyTorch's BCEWithLogitsLoss combines sigmoid + BCE in a numerically stable way.
        self.loss_fn = WeightedBCEWithLogits(pos_weight=cfg.pos_weight)
        
        # Training state
        self._step = 0
        self.best_loss = float("inf")
        self.early_stop_counter = 0

    # ---- Helpers ----
    def _rhat_vector(self, prompt: str, responses: List[str]) -> torch.Tensor:
        """
        Compute r̂ for each response in group based on chosen score_mode.
        In both modes:
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

                if self.online_training:
                    self._train_on_item(item, metrics)
                else:
                    self._train_on_item_offline(item, metrics)
                
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
    

    def _train_on_item_offline(self, item: RLVRItem, metrics: Dict[str, list]) -> None:
        """
        Train directly on dataset rewards (offline supervised PACS).
        
        Uses RLVRItem.prompt, RLVRItem.response, and RLVRItem.reward
        as the training signal without generating new responses.
        """
        prompt = item.prompt
        response = item.response
        reward_val = float(item.reward)

        # Convert reward into tensor
        reward = torch.tensor([reward_val], device=self.policy.device(), dtype=torch.float32)

        # --- Compute r̂ (online vs reference) ---
        if self.cfg.score_mode == "logprob":
            lp = self.policy.logprob_sum(prompt, response)          # requires grad
            lpr = self.policy.logprob_sum_ref(prompt, response)     # no grad
            rhat = self.cfg.beta * (lp - lpr)
        else:  # critic mode
            lg = self.policy.critic_logit(prompt, response)         # requires grad
            lgr = self.policy.critic_logit_ref(prompt, response)    # no grad
            rhat = self.cfg.beta * (lg - lgr)

        psi = rhat.unsqueeze(0)  # keep batch dimension

        # --- Compute supervised loss ---
        loss = self.loss_fn(psi, reward)

        # --- Optimization ---
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], 
                self.cfg.grad_clip
            )
        self.optimizer.step()
        self.scheduler.step(loss.item())

        # --- Metrics ---
        with torch.no_grad():
            psi_val = float(psi.item())
            pred = (torch.sigmoid(psi) > 0.5).float()
            acc_proxy = float((pred == reward).float().mean().item())

            metrics["losses"].append(float(loss.item()))
            metrics["psi_means"].append(psi_val)
            metrics["rhat_means"].append(float(rhat.item()))
            metrics["label_rates"].append(reward_val)
            metrics["acc_proxies"].append(acc_proxy)

        # --- Log ---
        self._log({
            "mode": f"{self.cfg.score_mode}-offline",
            "loss": float(loss.item()),
            "psi": psi_val,
            "rhat": float(rhat.item()),
            "reward": reward_val,
            "acc_proxy": acc_proxy,
            "prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else "")
        })

    
    def _train_on_item(self, item: RLVRItem, metrics: Dict[str, list]) -> None:
        """Train on a single dataset item and update metrics"""
        prompt = item.prompt
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
        
        # Detach psi for component calculations
        psi_detached = psi.detach()
        psi_sigmoid = torch.sigmoid(psi_detached)
        
        # Calculate actor component magnitude
        with torch.no_grad():
            # Actor component: policy improvement term
            # [l(q, o; πθ)]∇θ log πθ(o|q)
            cross_entropy_loss = -(labels * torch.log(psi_sigmoid + 1e-8) + 
                                (1 - labels) * torch.log(1 - psi_sigmoid + 1e-8))
            actor_component = cross_entropy_loss.mean().item()
            
            # Critic component: reward estimation term
            # (R(q, o) - σ(ψ(q, o; πθ)))∇θψ(q, o; πθ)
            critic_raw = (labels - psi_sigmoid).mean().item()  # Prediction error
            critic_grad = psi.mean().item()  # Gradient magnitude
            critic_component = critic_raw * critic_grad  # Full critic update
            
            # Calculate coupling ratio (absolute values to handle sign)
            coupling_ratio = abs(actor_component) / (abs(critic_component) + 1e-8)
            
            # Track policy entropy for exploration analysis
            policy_entropy = -torch.mean(psi_sigmoid * torch.log(psi_sigmoid + 1e-8) + 
                                        (1 - psi_sigmoid) * torch.log(1 - psi_sigmoid + 1e-8)).item()
            sequence_entropy = self.policy.calculate_response_entropy(prompt, responses)

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
            preds = (torch.sigmoid(psi) > 0.5).float()
            acc_proxy = float((preds == labels).float().mean().item())
            entropy = self.policy.calculate_response_entropy(prompt, responses)
        
        grad_analysis = self._analyze_gradients(psi, labels)

        metrics["actor_components"].append(grad_analysis["actor_component"])
        metrics["critic_raws"].append(grad_analysis["critic_raw"])
        metrics["critic_grads"].append(grad_analysis["critic_grad"])
        metrics["critic_components"].append(grad_analysis["critic_component"])
        metrics["coupling_ratios"].append(grad_analysis["coupling_ratio"])
        metrics["binary_entropies"].append(grad_analysis["binary_entropy"])
        metrics["losses"].append(float(loss.item()))
        metrics["psi_means"].append(psi_mean)
        metrics["rhat_means"].append(rhat_mean)
        metrics["label_rates"].append(label_pos_rate)
        metrics["acc_proxies"].append(acc_proxy)
        metrics["entropies"].append(entropy)
        metrics["policy_entropies"].append(policy_entropy)
        metrics["sequence_entropies"].append(sequence_entropy)
        metrics["response_entropy"].append(
            self.policy.calculate_response_entropy(prompt, responses)
        )

        # Log metrics
        self._log({
            "loss": float(loss.item()),
            "psi_mean": psi_mean,
            "rhat_mean": rhat_mean,
            "label_pos_rate": label_pos_rate,
            "acc_proxy": acc_proxy,
            "entropy": entropy,
            "mode": self.cfg.score_mode,
            "actor_component": actor_component,
            "critic_component": critic_component,
            "coupling_ratio": coupling_ratio,
            **grad_analysis,
            "policy_entropy": self.policy.calculate_response_entropy(prompt, responses)
        })

    def _analyze_gradients(self, psi: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Analyze gradient components as in PACS paper Equation 6"""
        # Compute sigmoid once for efficiency
        psi_sigmoid = torch.sigmoid(psi)
        
        # 1. ACTOR component: policy improvement term
        # [l(q, o; πθ)]∇θ log πθ(o|q)
        cross_entropy_loss = -(labels * torch.log(psi_sigmoid + 1e-8) + 
                            (1 - labels) * torch.log(1 - psi_sigmoid + 1e-8))
        actor_component = cross_entropy_loss.mean().item()
        
        # 2. CRITIC component breakdown
        prediction_error = (labels - psi_sigmoid)
        critic_raw = prediction_error.mean().item()  # R - σ(ψ)
        critic_grad = psi.mean().item()  # ∇θψ
        critic_component = (prediction_error * psi).mean().item()  # Full critic update
        
        # 3. Coupling ratio (normalized for stability)
        coupling_ratio = abs(actor_component) / (abs(critic_component) + 1e-8)
        
        # 4. Binary entropy (prediction confidence)
        binary_entropy = -torch.mean(
            psi_sigmoid * torch.log(psi_sigmoid + 1e-8) + 
            (1 - psi_sigmoid) * torch.log(1 - psi_sigmoid + 1e-8)
        ).item()
        
        return {
            "actor_component": actor_component,
            "critic_raw": critic_raw,
            "critic_grad": critic_grad,
            "critic_component": critic_component,
            "coupling_ratio": coupling_ratio,
            "binary_entropy": binary_entropy
        }

# Add to imports at the top

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
    
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        
        # Device management
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize configuration
        self._init_config(cfg)
        
        # Initialize ScorableClassifier for prompt type classification
        try:
            self.classifier = ScorableClassifier(
                memory=self.memory,
                logger=self.logger,
                config_path="config/domain/seeds.yaml",
                metric="cosine"
            )
            self.logger.log("ScorableClassifierInitialized", {
                "message": "Successfully initialized ScorableClassifier for prompt type classification"
            })
        except Exception as e:
            self.logger.log("ScorableClassifierError", {
                "error": str(e),
                "message": "Failed to initialize ScorableClassifier, falling back to simple classification"
            })
            self.classifier = None
        
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
                from stephanie.scoring.transforms.regression_tuner import \
                    RegressionTuner

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
            scorer = SICQLScorer(self.cfg, self.memory, self.logger)
            critic = scorer.get_model(self.dimension)
        # Create policy adapter
        adapter = HybridSICQLAdapter(
            memory=self.memory,
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
            logger=self._log_metrics,
            online_training=False
        ) 
        
        # 5. Train model
        training_stats = self.pacs_core.train(dataset, max_steps=self.max_steps)
        
        # 6. Save model and metadata
        meta = self._save_model(policy, dimension, training_stats, dataset=dataset)

        # 7. Log training stats
        self._log_training_stats(dimension, meta)
        
        
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
    
    def _save_model(self, policy, dimension: str, training_stats: Dict[str, Any], dataset=None) -> Dict[str, Any]:
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
            os.makedirs(critic_path, exist_ok=True) 
            # Save SICQL components
            torch.save(policy.critic.encoder.state_dict(), os.path.join(critic_path, "encoder.pt"))
            torch.save(policy.critic.q_head.state_dict(), os.path.join(critic_path, "q_head.pt"))
            torch.save(policy.critic.v_head.state_dict(), os.path.join(critic_path, "v_head.pt"))
            torch.save(policy.critic.pi_head.state_dict(), os.path.join(critic_path, "pi_head.pt"))
        
        # Calculate policy metrics
        policy_entropy = self._calculate_policy_entropy(policy, dataset, tokenizer=policy.tok)
        policy_stability = self._calculate_policy_stability(policy, dataset=dataset)

        # Build metadata
        meta = {
            "version": self.model_version,
            "dimension": dimension,
            "score_mode": self.score_mode,
            "policy_entropy_calc": policy_entropy,
            "policy_stability_calc": policy_stability,
            "actor_model": "Qwen2.5-1.5B",
            "critic_model": "SICQL" if self.score_mode == "critic" else None,
            "beta": self.beta,
            "group_size": self.group_size,
            "avg_loss": float(training_stats["avg_loss"]),
            "policy_entropy": training_stats["entropy"],
            "policy_stability": training_stats["acc_proxy"],
            "steps": training_stats["steps"],
            "device": str(self.device),
            "model_path": output_dir,
            "timestamp": datetime.now().isoformat(),
            "config": self.cfg if isinstance(self.cfg, dict) else self.cfg.__dict__
        }

        # Split into direct columns vs. extra_data
        db_fields = {
            "model_type": "pacs",
            "target_type": self.target_type,
            "dimension": meta["dimension"],
            "version": meta["version"],
            "score_mode": meta["score_mode"],
            "model_path": meta["model_path"],
        }

        extra_data = {k: v for k, v in meta.items() if k not in db_fields and k not in ["model_path", "dimension", "version", "score_mode"]}

            
        # Save metadata
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Insert into DB
        model_version = ModelVersionORM(
            **db_fields,
            extra_data=extra_data
        )
        self.memory.session.add(model_version)
        self.memory.session.commit()
        
        return meta
    
    def _calculate_policy_entropy(self, policy, dataset, tokenizer, sample_size: int = 100) -> float:
        """
        Calculate average entropy of the actor policy over a subset of dataset.
        """
        import random
        subset = random.sample(dataset, min(sample_size, len(dataset)))

        entropies = []
        for item in subset:
            prompt, response = item.prompt, item.response
            if not response:
                continue

            logits = self._get_logits(policy, prompt, response, tokenizer, device=policy._device)
            # take only the response part
            resp_len = len(tokenizer.encode(response))
            resp_logits = logits[:, -resp_len:, :]

            probs = F.softmax(resp_logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            entropies.append(entropy)

        return float(sum(entropies) / len(entropies)) if entropies else 0.0

    def _get_logits(self, policy, prompt: str, response: str, tokenizer, device="cuda"):
        """Tokenize prompt+response and run through actor LM to get logits."""
        inputs = tokenizer(prompt + response, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = policy.actor(**inputs)
            logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
        return logits

    def _calculate_policy_stability(self, policy, dataset, sample_size: int = 100) -> float:
        subset = random.sample(dataset, min(sample_size, len(dataset)))
        rewards, kl_divs = [], []

        for item in subset:
            prompt, response = item.prompt, item.response
            if not response:
                continue

            # Current policy logits
            logits_cur = self._get_logits(policy, prompt, response, policy.tok, device=policy._device)
            probs_cur = F.softmax(logits_cur, dim=-1)

            # Reference policy logits
            logits_ref = self._get_logits(policy, prompt, response, policy.tok, device=policy._device)
            probs_ref = F.softmax(logits_ref, dim=-1)

            kl = F.kl_div(probs_ref.log(), probs_cur, reduction="batchmean").item()
            kl_divs.append(kl)

            if policy.critic is not None:
                reward = policy.critic_logit(prompt, response).item()
                rewards.append(reward)

        reward_var = np.var(rewards) if rewards else 0.0
        kl_mean = np.mean(kl_divs) if kl_divs else 0.0
        return 1.0 / (1.0 + reward_var + kl_mean)
    
    def _log_training_stats(self, dimension: str, meta: Dict[str, Any]):
        """Log training stats to database"""
        training_stats = TrainingStatsORM(
            model_type="pacs",
            target_type=self.target_type,
            embedding_type=self.memory.embedding.name,
            dimension=dimension,
            version=self.model_version,
            avg_q_loss=meta["avg_loss"],
            policy_entropy=meta["policy_entropy"],
            policy_stability=meta["policy_stability"],
        )
        self.memory.session.add(training_stats)
        self.memory.session.commit()
    
    def _update_belief_cartridge(self, context: dict, dimension: str, meta: Dict[str, Any]):
        """Update belief cartridges with policy stats and proper goal ID"""
        # Extract goal_id from context
        goal_id = context.get("goal", {}).get("id")
        if not goal_id:
            self.logger.log("MissingGoalID", {"dimension": dimension})
        
        bc = BeliefCartridgeORM(
            title=f"PACS Policy - {dimension}",
            content=f"PACS training completed for {dimension} dimension",
            goal_id=goal_id,  # FIXED: now properly set
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