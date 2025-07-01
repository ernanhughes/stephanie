# stephanie/agents/score_analysis_agent.py
import matplotlib.pyplot as plt
import pandas as pd

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE_RUN_ID
from stephanie.models import EvaluationORM
from stephanie.scoring.score_analyzer import ScoreAnalyzer


class ScoreAnalysisAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "ScoreAnalysisAgent"})

    async def run(self, context: dict) -> dict:
        # pipeline_run_id = 1061
        pipeline_run_id = context.get("pipeline_run_id")
        self.logger.log("ScoreAnalysisStarted", {"pipeline_run_id": pipeline_run_id})

        # Fetch all EvaluationORM entries
        raw_scores = self.memory.evaluations.get_by_pipeline_run_id(pipeline_run_id)
        if not raw_scores:
            self.logger.log("ScoreAnalysisEmpty", {"pipeline_run_id": pipeline_run_id})
            return context

        # Flatten the dimension scores from the nested dict structure
        data = []
        for row in raw_scores:
            hypothesis_id = row.get("hypothesis_id")
            stage = row.get("stage")
            scores_dict = row.get("scores", {})
            dims = scores_dict.get("dimensions")
            for dim_name, dim_data in dims.items():
                data.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "metrics": stage,
                        "dimension": dim_name,
                        "score": dim_data.get("score"),
                        "rationale": dim_data.get("rationale"),
                        "weight": dim_data.get("weight"),
                    }
                )

        df = pd.DataFrame(data)

        # Analyze
        analyzer = ScoreAnalyzer(df)
        desc = analyzer.describe_scores()
        print("\n📊 Score Summary:\n", desc)

        pca_components, variance_ratio = analyzer.perform_pca()
        print("\n🔍 PCA Variance Explained:\n", variance_ratio)

        # Plot
        analyzer.plot_pca_clusters(n_clusters=3)

        self.logger.log("ScoreAnalysisCompleted", {
            "pipeline_run_id": pipeline_run_id,
            "pca_variance_ratio": variance_ratio.tolist(),
        })

        return context
