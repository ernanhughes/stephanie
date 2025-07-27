# stephanie/agents/belief_cartridge_builder.py
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

import numpy as np
import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.memory.evaluation_attribute_store import \
    EvaluationAttributeStore
from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.evaluation_attribute import EvaluationAttributeORM
from stephanie.models.score import ScoreORM
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.metrics import compute_uncertainty


class BeliefCartridgeBuilder(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "novelty"])
        self.embedding_types = cfg.get("embedding_types", ["hnet", "mxbai", "hf"])
        self.scorer_types = cfg.get("scorer_types", ["mrq", "ebt", "svm", "llm"])
        self.use_sicql = cfg.get("use_sicql", False)
        self.track_efficiency = cfg.get("track_efficiency", True)
        self.uncertainty_threshold = cfg.get("uncertainty_threshold", 0.3)
        self.evaluation_store = EvaluationAttributeStore(memory.session, logger)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def build_cartridge(self, context: dict) -> dict:
        goal = context.get(GOAL)
        document = context.get("document")
        
        if not document or not goal:
            return context

        # Create scorable object
        scorable = Scorable(
            text=document.get("text"),
            target_type=TargetType.DOCUMENT,
            id=document.get("id")
        )
        
        # Initialize belief cartridge
        cartridge = BeliefCartridgeORM(
            id=document.get("id"),
            title=document.get("title", ""),
            content=document.get("text", ""),
            goal_id=goal.get("id"),
            domain=goal.get("domain", "default"),
            created_at=datetime.utcnow()
        )
        
        # Score across all dimensions and scorers
        scoring_results = await self._score_document(goal, scorable)
        
        # Generate idea (if applicable)
        if self.cfg.get("generate_idea", True):
            scorable.embedding_type = "hnet"  # Default for idea generation
            idea = await self._generate_idea(goal, scorable)
            cartridge.idea = idea
            
        # Save to database
        await self._save_to_db(cartridge, scoring_results, goal, scorable)
        
        context["belief_cartridge"] = cartridge.to_dict()
        return context

    async def _score_document(self, goal, scorable: Scorable) -> Dict[str, Dict]:
        """Score document across all dimensions, embeddings, and scorers"""
        results = {
            "scores": {},
            "dimension_scores": defaultdict(dict),
            "embedding_comparison": defaultdict(lambda: defaultdict(dict)),
            "metadata": {
                "embedding_types": self.embedding_types,
                "scorers": self.scorer_types,
                "dimensions": self.dimensions
            }
        }
        
        # Track execution time for efficiency calculation
        start_time = time.time()
        
        # Score across all combinations
        for embedding_type in self.embedding_types:
            scorable.embedding_type = embedding_type
            
            for scorer_type in self.scorer_types:
                scorer = self._get_scorer(scorer_type, embedding_type)
                if not scorer:
                    continue
                
                # Score all dimensions
                scores = await scorer.score(goal, scorable)
                
                # Store results
                results["scores"][f"{embedding_type}_{scorer_type}"] = scores
                results["dimension_scores"][embedding_type][scorer_type] = scores
                
                # Track per-dimension scores for comparison
                for dim, score in scores.items():
                    results["embedding_comparison"][dim][f"{embedding_type}_{scorer_type}"] = score
        
        # Calculate efficiency metrics
        if self.track_efficiency:
            results["metadata"]["execution_time"] = time.time() - start_time
            results["metadata"]["efficiency"] = self._compute_efficiency(
                results["scores"], 
                goal.get("llm_score", {})
            )
        
        return results

    def _get_scorer(self, scorer_type: str, embedding_type: str):
        """Get appropriate scorer based on type and embedding"""
        if scorer_type == "sicql" and self.use_sicql:
            return self.memory.sicql_scorer.get(embedding_type)
        elif scorer_type == "mrq":
            return self.memory.mrq_scorer.get(embedding_type)
        elif scorer_type == "ebt":
            return self.memory.ebt_scorer.get(embedding_type)
        elif scorer_type == "svm":
            return self.memory.svm_scorer.get(embedding_type)
        elif scorer_type == "llm":
            return self.memory.llm_scorer
        return None

    async def _generate_idea(self, goal, scorable: Scorable) -> str:
        """Generate structured idea using GILD-enhanced scoring"""
        try:
            # Use best scorer according to efficiency metrics
            best_scorer = self._select_best_scorer(goal, scorable)
            scores = await best_scorer.score(goal, scorable)
            
            # Apply GILD policy if available
            if hasattr(best_scorer, "apply_gild_policy"):
                scores = best_scorer.apply_gild_policy(scores)
            
            # Generate idea using EBT refinement
            if self.cfg.get("use_ebt_refinement", True):
                ebt_scorer = self._get_scorer("ebt", scorable.embedding_type)
                scorable.text = await ebt_scorer.refine(scorable.text)
            
            return self._format_idea(goal, scorable, scores)
            
        except Exception as e:
            self.logger.log("IdeaGenerationFailed", {"error": str(e)})
            return ""

    def _select_best_scorer(self, goal, scorable: Scorable):
        """Select best scorer based on historical performance"""
        if self.cfg.get("use_gild_selector", True):
            gild_selector = self.memory.gild_selector
            best_scorer_type = gild_selector.select_scorer(goal, scorable)
            return self._get_scorer(best_scorer_type, scorable.embedding_type)
        
        # Fallback to default scorer
        return self._get_scorer(self.cfg.get("default_scorer", "mrq"), scorable.embedding_type)

    async def _save_to_db(self, cartridge: BeliefCartridgeORM, scoring_results, goal, scorable: Scorable):
        """Save belief cartridge and all scoring results"""
        try:
            # Save main cartridge
            self.memory.session.add(cartridge)
            self.memory.session.flush()
            
            # Save evaluations for each scorer
            for scorer_key, scores in scoring_results["scores"].items():
                embedding_type, scorer_type = scorer_key.split("_", 1)
                evaluation = self._create_evaluation(
                    cartridge.id, goal, scorable, embedding_type, scorer_type
                )
                self.memory.session.add(evaluation)
                
                # Save detailed scores
                for dim, score in scores.items():
                    prompt_hash = ScoreORM.compute_prompt_hash(
                        goal_text=goal.get("goal_text", ""),
                        document_text=scorable.text
                    )
                    score_orm = self._create_score_orm(
                        evaluation.id, dim, score, scorer_type, prompt_hash
                    )
                    self.memory.session.add(score_orm)
                    
                # Save SICQL-specific attributes if available
                if scorer_type == "sicql" and self.use_sicql:
                    await self._save_sicql_attributes(evaluation.id, scores, scorable)

            # Save efficiency metrics
            efficiency = scoring_results["metadata"].get("efficiency", {})
            for scorer_key, score in efficiency.items():
                if "composite" not in scorer_key:
                    continue
                _, scorer_type = scorer_key.split("_")
                cartridge.efficiency_score = score["efficiency"]
                cartridge.efficiency_details = efficiency.get(scorer_type, {})

            self.memory.session.commit()
            
        except Exception as e:
            self.memory.session.rollback()
            self.logger.log("CartridgeSaveFailed", {
                "error": str(e),
                "cartridge_id": cartridge.id
            })
            raise

    def _create_evaluation(self, cartridge_id, goal, scorable, embedding_type, scorer_type):
        """Create evaluation record with metadata"""
        return EvaluationORM(
            goal_id=goal.get("id"),
            target_id=scorable.id,
            target_type=scorable.target_type,
            evaluator_name=scorer_type,
            model_name=f"{scorable.target_type}_{scorer_type}_v1",
            embedding_type=embedding_type,
            cartridge_id=cartridge_id
        )

    def _create_score_orm(self, evaluation_id, dimension, score, scorer_type, prompt_hash):
        """Create score ORM object with metadata"""
        return ScoreORM(
            evaluation_id=evaluation_id,
            dimension=dimension,
            score=score,
            rationale=f"{scorer_type} scorer",
            energy=score,  # For compatibility
            source=scorer_type,
            target_type=TargetType.DOCUMENT,
            prompt_hash=prompt_hash
        )

    async def _save_sicql_attributes(self, evaluation_id: int, scores: Dict[str, float], scorable: Scorable):
        """Save SICQL-specific metrics to evaluation attributes"""
        try:
            sicql_scorer = self.memory.sicql_scorer.get(scorable.embedding_type)
            if not sicql_scorer:
                return
                
            # Get SICQL outputs
            sicql_outputs = await sicql_scorer.get_detailed_scores(scorable)
            
            # Save Q/V values and policy metrics
            for dim in self.dimensions:
                attribute = EvaluationAttributeORM(
                    evaluation_id=evaluation_id,
                    dimension=dim,
                    source="sicql",
                    q_value=sicql_outputs[dim].get("q_value"),
                    v_value=sicql_outputs[dim].get("v_value"),
                    advantage=sicql_outputs[dim].get("advantage"),
                    policy_logits=sicql_outputs[dim].get("policy_logits"),
                    uncertainty=compute_uncertainty(
                        sicql_outputs[dim].get("q_value"),
                        sicql_outputs[dim].get("v_value")
                    ),
                    entropy=sicql_outputs[dim].get("entropy")
                )
                self.evaluation_store.insert(attribute)
                
                # Log high uncertainty cases
                if attribute.uncertainty > self.uncertainty_threshold:
                    self.logger.log("HighUncertainty", {
                        "document_id": scorable.id,
                        "dimension": dim,
                        "uncertainty": attribute.uncertainty
                    })
                    
        except Exception as e:
            self.logger.log("SICQLAttributeSaveFailed", {
                "error": str(e),
                "evaluation_id": evaluation_id
            })

    def _compute_efficiency(self, scores: Dict[str, Dict], llm_scores: Dict) -> Dict:
        """Compute composite efficiency scores for scorer selection"""
        efficiency_scores = {}
        
        for scorer_key, dim_scores in scores.items():
            if "llm" in scorer_key:
                continue
                
            # Calculate alignment with LLM scores
            alignment_gains = {}
            for dim, score in dim_scores.items():
                llm_score = llm_scores.get(dim, score)
                alignment_gains[dim] = 1 - abs(score - llm_score) / 100
                
            # Calculate composite efficiency
            avg_gain = np.mean(list(alignment_gains.values()))
            execution_time = 0.1  # Placeholder from metadata
            
            efficiency_scores[f"composite_{scorer_key}"] = {
                "efficiency": avg_gain / (execution_time + 0.01),
                "alignment_gains": alignment_gains,
                "execution_time": execution_time,
                "scorer_key": scorer_key
            }
            
        return efficiency_scores

    def _format_idea(self, goal, scorable: Scorable, scores: Dict) -> str:
        """Format idea with scoring metadata"""
        return f"""
        IDEA: {scorable.text[:200]}...
        DIMENSION SCORES:
        {" ".join(f"{dim}: {score:.1f}" for dim, score in scores.items())}
        METADATA:
        embedding_type: {scorable.embedding_type}
        scorer: {self.cfg.get("default_scorer", "mrq")}
        efficiency: {scores.get("efficiency", "N/A")}
        GOAL ALIGNMENT: {scores.get("alignment", 0.0)}
        """