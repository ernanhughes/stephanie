
# stephanie/agents/selection/gild_selector.py

import logging
from collections import defaultdict

import scipy.stats

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class GILDSelectorAgent(BaseAgent):
    """
    Selects the most efficient scoring strategy per goal using GILD-like greedy logic.
    Ingests precomputed efficiency scores from CostBenefitAnalyzerAgent.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger or logging.getLogger(__name__))

        self.strategy = cfg.get("selection_strategy", "greedy")  
        self.min_efficiency = cfg.get("min_efficiency", 0.1)
        self.top_k = cfg.get("top_k", 1)
        self.require_llm = cfg.get("require_llm", True)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        try:
            # Load scored examples from the database (e.g., GILDScoringExample list)
            gild_examples = self.memory.scoring.load_gild_examples()
            # Aggregate into efficiency scores
            scores = self._compute_efficiency_scores(gild_examples)
            self.logger.log("GILDSelector", scores)
            selected = self._select_scorers(scores)
            context["efficiency_scores"] = scores
            context["selected_scorers"] = selected

            stats = self.memory.scoring.get_scorer_stats()
            self.logger.log("GILDSelector", {"scoring_stats": stats})
            context["scoring_stats"] = stats

            goal_id = goal.get("id")
            report = self.memory.scoring.generate_comparison_report(goal_id)
            self.logger.log("GILDSelector", {"goal_id": goal_id, "report": report})

            return context

        except Exception as e:
            return context

    def _compute_efficiency_scores(self, examples: list) -> dict:
        from collections import defaultdict

        import numpy as np

        scorer_fields = [
            "hnet_ebt_score", "huggingface_ebt_score", "ollama_ebt_score",
            "hnet_svm_score", "huggingface_svm_score", "ollama_svm_score",
            "hnet_mrq_score", "huggingface_mrq_score", "ollama_mrq_score"
        ]

        grouped = defaultdict(list)
        for ex in examples:
            for scorer_name, score in ex.scores.items():
                if scorer_name != "llm" and score is not None and ex.llm_score is not None:
                    # Normalize to [0, 1]
                    norm_score = score / 100.0
                    norm_llm = ex.llm_score / 100.0

                    # Use inverse absolute error
                    efficiency = max(0.0, 1.0 - abs(norm_score - norm_llm))
                    grouped[scorer_name].append(efficiency)

        

        result = {}
        for scorer, effs in grouped.items():
            mean = np.mean(effs)
            ci = scipy.stats.t.interval(
                confidence=0.95, 
                df=len(effs)-1, 
                loc=mean, 
                scale=scipy.stats.sem(effs)
            )
            result[scorer] = {
                "efficiency": float(mean),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "sample_count": len(effs)
            }
        return result

    def _select_scorers(self, scores: dict) -> list[str]:
        """
        Select top evaluators based on efficiency score using GILD-style strategy.
        """
        filtered = {
            name: data
            for name, data in scores.items()
            if data["efficiency"] >= self.min_efficiency
        }

        # if self.require_llm and "llm" not in filtered:
        #     self.logger.log("LLM missing or below efficiency threshold; using fallback.")
        #     filtered["llm"] = scores.get("llm", {"efficiency": 0.01})

        result = []
        if self.strategy == "greedy":
            result = [max(filtered, key=lambda k: filtered[k]["efficiency"])]
            self.logger.log("Greedy selection", result)
        elif self.strategy == "top_k":
            sorted_items = sorted(
                filtered.items(), key=lambda x: x[1]["efficiency"], reverse=True
            )
            result = [k for k, _ in sorted_items[:self.top_k]]
            self.logger.log("Top K selection", result)      
        elif self.strategy == "weighted":
            # Normalize efficiencies into weights
            total = sum(data["efficiency"] for data in filtered.values())
            result = [{
                name: data["efficiency"] / total
                for name, data in filtered.items()
            }]
            self.logger.log("Weighted selection", result)
        return result

    # In GILDSelectorAgent._compute_efficiency_scores
    def _compute_efficiency_scores(self, examples: list) -> dict:
        from datetime import datetime

        import numpy as np
        
        scorer_fields = [f"{emb}_{scorer}_score" for emb in ["hnet", "huggingface", "ollama"] 
                        for scorer in ["ebt", "svm", "mrq"]]
        
        grouped = defaultdict(list)
        for ex in examples:
            age_days = (datetime.now() - ex.created_at).days  # Add created_at to GILDScoringExample
            time_weight = 0.9 ** age_days  # Exponential decay
            
            for scorer_name in scorer_fields:
                score = getattr(ex, scorer_name)
                if score is not None and ex.llm_score is not None:
                    norm_score = score / 100.0
                    norm_llm = ex.llm_score / 100.0
                    efficiency = max(0.0, 1.0 - abs(norm_score - norm_llm)) * time_weight
                    grouped[scorer_name].append(efficiency)
        
        return {scorer: {"efficiency": float(np.mean(effs))} for scorer, effs in grouped.items()}
