# stephanie/cli/compare.py
import json

import click
from omegaconf import OmegaConf

from stephanie.logs.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.reports.comparison import ComparisonReporter


@click.command()
@click.argument("run_a_id", type=str)
@click.argument("run_b_id", type=str)
@click.option("--output", "-o", default="comparison_report.json")
@click.option("--verbose", "-v", is_flag=True)
def compare_runs(run_a_id: str, run_b_id: str, output: str, verbose: bool):
    """
    CLI command to compare two pipeline runs.
    Usage: python -m stephanie.cli.compare_runs <run_a_id> <run_b_id>
    """
    # Generate report
    try:

        # Load config
        cfg = OmegaConf.load("configs/config.yaml")
        log_path = cfg.logger.log_path

        # Initialize logger
        logger = JSONLogger(log_path)

        # Initialize memory tool
        memory = MemoryTool(cfg, logger)

        # Initialize reporter
        reporter = ComparisonReporter(session=memory.get_session())


        report = reporter.compare_runs(run_a_id, run_b_id)

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

        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Full report saved to {output}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise