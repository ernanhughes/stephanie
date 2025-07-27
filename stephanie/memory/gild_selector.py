# stephanie/memory/gild_selector.py
import json
from typing import Dict, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session
from torch import nn
from torch.nn import functional as F

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.model_locator import ModelLocator

from stephanie.

class GILDSelector:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = {}  # Cache for frequent selections
        self.uncertainty_threshold = 0.3  # From config

    def select_best_scorer(self, goal: dict, scorable: Scorable) -> Dict[str, any]:
        """
        Select best scorer based on:
        1. Historical performance
        2. Policy drift from expert demonstrations
        3. Epistemic uncertainty
        """
        cache_key = f"{goal.get('id')}_{scorable.id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get current policy
        current_policy = self._get_current_policy(goal, scorable)
        if not current_policy:
            return {"scorer": "mrq", "policy_logits": [0.5, 0.3, 0.2], "source": "default"}

        # Get expert demonstration
        expert_policy = self._get_expert_policy(goal, scorable)
        if not expert_policy:
            return {"scorer": "ebt", "policy_logits": current_policy["action_logits"], "source": "current"}

        # Calculate policy drift
        policy_drift = self._calculate_policy_drift(current_policy["action_logits"], expert_policy["action_logits"])
        self.logger.log("PolicyDrift", {
            "document_id": scorable.id,
            "policy_drift": policy_drift,
            "current_policy": current_policy["action_logits"],
            "expert_policy": expert_policy["action_logits"]
        })

        # Select based on policy drift
        if policy_drift < self.uncertainty_threshold:
            best_action = torch.argmax(torch.tensor(current_policy["action_logits"])).item()
        else:
            best_action = torch.argmax(torch.tensor(expert_policy["action_logits"])).item()

        scorer_map = ["ebt", "svm", "mrq"]
        selected_scorer = scorer_map[best_action]
        
        # Cache for reuse
        result = {
            "scorer": selected_scorer,
            "policy_logits": current_policy["action_logits"],
            "expert_logits": expert_policy["action_logits"],
            "policy_drift": policy_drift,
            "source": "gild_selector"
        }
        
        self.cache[cache_key] = result
        return result

    def _get_current_policy(self, goal: dict, scorable: Scorable) -> Optional[Dict]:
        """Get current policy from belief cartridges"""
        try:
            cartridge = self.session.query(BeliefCartridgeORM).filter_by(
                goal_id=goal.get("id"),
                target_id=scorable.id
            ).first()
            
            if cartridge and cartridge.policy_logits:
                return {
                    "action_logits": np.array(json.loads(cartridge.policy_logits)),
                    "entropy": cartridge.entropy,
                    "stability": cartridge.policy_stability
                }
            return None
            
        except Exception as e:
            self.logger.log("CurrentPolicyLoadFailed", {"error": str(e)})
            return None

    def _get_expert_policy(self, goal: dict, scorable: Scorable) -> Optional[Dict]:
        """Get expert demonstration from LLM or historical data"""
        try:
            # First try LLM as expert
            if goal.get("llm_score"):
                return {
                    "action_logits": goal["llm_score"]["policy_logits"],
                    "source": "llm"
                }
            
            # Fallback to historical best
            query = (
                self.session.query(EvaluationAttributeORM)
                .join(EvaluationORM)
                .filter(
                    EvaluationAttributeORM.dimension == "alignment",
                    EvaluationORM.target_id == scorable.id,
                    EvaluationORM.evaluator_name == "ebt"
                )
                .order_by(EvaluationAttributeORM.created_at.desc())
                .first()
            )
            
            if query and query.policy_logits:
                return {
                    "action_logits": np.array(json.loads(query.policy_logits)),
                    "source": "historical"
                }
            return None
            
        except Exception as e:
            self.logger.log("ExpertPolicyLoadFailed", {"error": str(e)})
            return None

    def _calculate_policy_drift(self, current_logits, expert_logits):
        """Calculate drift between current and expert policy"""
        if not current_logits or not expert_logits:
            return float('inf')
            
        current = torch.tensor(current_logits)
        expert = torch.tensor(expert_logits["action_logits"])
        
        # Ensure same shape
        if current.shape != expert.shape:
            current = current.unsqueeze(0) if current.dim() == 1 else current
            expert = expert.unsqueeze(0) if expert.dim() == 1 else expert
            
        # Calculate KL divergence
        current_probs = F.softmax(current, dim=-1)
        expert_probs = F.softmax(expert, dim=-1)
        
        kl_div = torch.sum(
            expert_probs * torch.log(expert_probs / (current_probs + 1e-8) + 1e-8)
        ).item()
        
        # Normalize to [0, 1] scale
        return kl_div / (torch.sum(expert_probs * torch.log(expert_probs + 1e-8)).item() + 1e-8)

    def _calculate_policy_entropy(self, policy_logits):
        """Calculate entropy of policy distribution"""
        probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-8)).item()

    def get_policy_similarity(self, policy1, policy2):
        """Get similarity between two policies"""
        p1 = F.softmax(torch.tensor(policy1), dim=-1)
        p2 = F.softmax(torch.tensor(policy2), dim=-1)
        
        # Calculate cosine similarity
        similarity = torch.dot(p1, p2) / (torch.norm(p1) * torch.norm(p2) + 1e-8)
        return similarity.item()

    def _select_best_action(self, policy_logits):
        """Select best action based on policy logits"""
        if not policy_logits:
            return 0  # Default to ebt
            
        logits = torch.tensor(policy_logits)
        probs = F.softmax(logits, dim=-1)
        
        # Use policy entropy to decide strategy
        entropy = self._calculate_policy_entropy(policy_logits)
        if entropy < 0.5:
            # Confident policy - exploit
            return torch.argmax(probs).item()
        else:
            # Uncertain policy - explore
            return torch.multinomial(probs, 1).item()

    def _validate_tensor(self, tensor, name):
        """Validate tensor shape and device"""
        if tensor is None:
            self.logger.log("InvalidTensor", {
                "tensor_name": name,
                "reason": "tensor is None"
            })
            return False
            
        if torch.isnan(tensor).any():
            self.logger.log("NaNInTensor", {
                "tensor_name": name,
                "tensor": tensor.tolist()
            })
            return False
            
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dim if missing
            
        if tensor.dim() == 2 and tensor.size(1) == 1:
            tensor = tensor.repeat(1, 3)  # Expand to 3 actions if needed
            
        return tensor

    def get_scorer_stats(self):
        """Get statistics about scorer performance"""
        query = text("""
        SELECT 
            evaluator_name,
            embedding_type,
            COUNT(*) AS example_count,
            AVG(policy_stability) AS avg_stability,
            AVG(policy_entropy) AS avg_entropy
        FROM belief_cartridges
        GROUP BY evaluator_name, embedding_type
        """)
        
        return pd.DataFrame([dict(row._mapping) for row in self.session.execute(query).fetchall()])

    def _calculate_policy_consistency(self, policy_logits):
        """Calculate how consistent a policy is across samples"""
        if not policy_logits:
            return 0.0
            
        probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        if probs.dim() == 2:
            probs = probs.mean(dim=0)
            
        # Calculate consistency as max probability
        return probs.max().item()

    def get_scorer_recommendation(self, goal, scorable):
        """Get scorer recommendation based on epistemic signals"""
        policy = self.select_best_scorer(goal, scorable)
        
        # Map policy to recommendation
        if policy["policy_drift"] < 0.2:
            return {"scorer": policy["scorer"], "confidence": 0.9}
        elif policy["policy_drift"] < 0.5:
            return {"scorer": policy["scorer"], "confidence": 0.7}
        else:
            return {"scorer": "mrq", "confidence": 0.5}