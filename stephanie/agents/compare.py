import json

from stephanie.agents.base_agent import BaseAgent
from stephanie.reports.comparison import ComparisonReporter


class CompareAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here

        reporter = ComparisonReporter(session=self.memory.session)


        report = reporter.compare_runs(3670, 3669)

        print(f"🎯 Goal: {report['goal']}")
        print(f"Preferred Run: {report['delta']['preferred']}")
        print(f"Score Difference: {report['delta']['score_diff']:.2f}")
        print(f"Embedding Quality: {report['delta']['embedding_quality']}")
        print(f"Convergence: {report['delta']['convergence']}")

        print("\n📊 Stage-by-Stage Comparison:")
        for stage in report["delta"]["stage_performance"]:
            print(
                f"{stage['stage']}: {stage['run_a_score']} vs {stage['run_b_score']} → {stage['rationale']}"
            )

        print("\n📈 Summary of Differences:"
              f"\n- Preferred Run: {report['delta']['preferred']}"
              f"\n- Score Difference: {report['delta']['score_diff']:.2f}"
              f"\n- Embedding Quality: {report['delta']['embedding_quality']}"
              f"\n- Convergence: {report['delta']['convergence']}")

        with open(self.get_output(), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Full report saved to {self.get_output()}")

        return context
    
    def get_output(self):
        return "output.json"