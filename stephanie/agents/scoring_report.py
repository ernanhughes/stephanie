# stephanie/agents/analysis/scoring_store_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.scoring_store import ScoringStore
from stephanie.utils.visualization_utils import save_dataframe_plot  # Optional

class ScoringReportAgent(BaseAgent):
    def __init__(self, cfg, memory, logger=None):
        super().__init__(cfg, memory, logger)
        self.store = ScoringStore(memory.session, logger)
        self.goal_id = cfg.get("goal_id", 4148)

    async def run(self, context: dict) -> dict:
        """Main entry point for generating reports from ScoringStore."""
        dimension = context.get("dimension", "alignment")

        # Load GILD examples
        examples = self.store.load_gild_examples()
        self.logger.log("LoadedGILDExamples", {"count": len(examples)})

        # Generate scorer stats
        stats = self.store.get_scorer_stats()
        self.logger.log("ScorerStats", stats)

        # Generate comparison report
        comparison_df = self.store.generate_comparison_report(goal_id=self.goal_id)
        comparison_df.to_csv(f"reports/scorer_comparison_{self.goal_id}.csv", index=False)
        self.logger.log("ComparisonReportGenerated", {"rows": len(comparison_df)})

        # Temporal bias over time
        bias_df = self.store.get_temporal_analysis(dimension)
        bias_df.to_csv(f"reports/temporal_bias_{dimension}.csv", index=False)

        # GILD feedback
        feedback_df = self.store.get_gild_training_feedback()
        feedback_df.to_csv(f"reports/gild_feedback.csv", index=False)

        # GILD effectiveness
        gain_df = self.store.get_gild_effectiveness()
        gain_df.to_csv(f"reports/gild_gain.csv", index=False)

        # Optional visualization
        self.store.plot_scorer_comparison(dimension)

        return {
            "scorer_stats": stats,
            "comparison_summary": comparison_df.head().to_dict(),
            "gild_feedback_summary": feedback_df.head().to_dict(),
            "temporal_bias": bias_df.head().to_dict(),
            "gain": gain_df.head().to_dict()
        }
