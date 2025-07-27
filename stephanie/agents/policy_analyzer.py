# In your analysis agent
import os
from datetime import datetime

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.policy_analyzer import PolicyAnalyzer


class PolicyAnalysisAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "novelty"])
        self.analyzer = PolicyAnalyzer(memory.session, logger)
        self.output_dir = cfg.get("report_output_dir", "logs/policy_reports")

        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    async def run(self, context: dict) -> dict:
        reports = {}
        pipeline_run_id = context.get("pipeline_run_id", None)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for dim in self.dimensions:
            report = self.analyzer.generate_policy_report(dim, pipeline_run_id=pipeline_run_id)
            reports[dim] = report

            # Log insights
            for insight in report.get("insights", []):
                self.logger.log("PolicyInsight", {"dimension": dim, "insight": insight})

            # Generate markdown and save
            markdown = self.analyzer.generate_markdown_summary(report)
            file_path = os.path.join(self.output_dir, f"{dim}_{timestamp}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            # Optionally print path to console or log
            self.logger.log("PolicyReportSaved", {"dimension": dim, "path": file_path})

            # Generate visualization if needed
            if self.cfg.get("generate_visualization", True):
                viz_paths = self.analyzer.visualize_policy(dim)
                if viz_paths:
                    report["visualizations"] = viz_paths

        context["policy_analysis"] = reports
        return context
