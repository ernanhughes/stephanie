"""Run a bounded, model-free Hydra/Supervisor persistence smoke test.

This deliberately writes one Goal, PipelineRun, PlanTrace and ExecutionStep to
the database selected by ``config/code.yaml``. It does not create tables. Pass
``--execute`` explicitly to perform those writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stephanie.core.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.orm.base import database_url_from_config
from stephanie.supervisor import Supervisor


def compose_smoke_config(fixture_dir: str):
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir.resolve())
    ):
        cfg = compose(config_name="code")

    OmegaConf.set_struct(cfg, False)
    # Programmatic compose does not set HydraConfig.runtime.choices.
    cfg.services_profile = "all"
    cfg.paths.prompts = str((repo_root / "prompts").resolve())
    cfg.pipeline.name = "stn01_loader_smoke"
    cfg.pipeline.tag = "stn01"
    cfg.pipeline.description = "One-stage deterministic runtime and trace smoke check"
    cfg.pipeline.stages = [
        {
            "name": "critic_loader",
            "description": "Load one deterministic JSONL fixture",
            "cls": "stephanie.agents.loader.LoaderAgent",
            "enabled": True,
            "iterations": 1,
        }
    ]
    cfg.agents.critic_loader.log_dirs = [fixture_dir]
    cfg.agents.critic_loader.log_glob = "*.jsonl"
    cfg.agents.critic_loader.shuffle = False
    cfg.agents.critic_loader.max_records = 10
    cfg.plan_monitor.enabled = True
    cfg.plan_monitor.score_enabled = False
    cfg.plan_monitor.index_embeddings = False
    cfg.plan_monitor.save_output = False
    cfg.bus.enabled = False
    cfg["symbolic"] = {"enabled": False}
    cfg.report.generate_report = False
    cfg.goal.goal_text = f"STN01 deterministic runtime smoke {uuid4()}"
    cfg.goal.goal_type = "test"
    cfg.goal.goal_category = "stn01"
    cfg.goal.strategy = "loader_smoke"
    cfg.goal.difficulty = "low"
    return cfg


async def execute_smoke(cfg) -> dict:
    memory = None
    supervisor = None
    try:
        memory = MemoryTool(cfg=cfg, logger=JSONLogger("logs/stn01_smoke.jsonl"))
        identity = memory.database_identity()
        if not identity["matches"]:
            raise RuntimeError("Configured, ORM and direct database identities differ")

        supervisor = Supervisor(cfg=cfg, memory=memory, logger=memory.logger)
        result = await supervisor.run_pipeline_config(
            {"goal": OmegaConf.to_container(cfg.goal, resolve=True)}
        )
        trace_id = str(result["plan_trace_id"])
        run_id = result["pipeline_run_id"]
        trace = memory.plan_traces.get_by_trace_id(trace_id)
        run = memory.pipeline_runs.get_by_id(run_id)
        summary = {
            "database": identity["orm"]["database"],
            "database_identity_matches": identity["matches"],
            "pipeline": cfg.pipeline.name,
            "stages_executed": [stage.name for stage in supervisor.pipeline_stages],
            "service_profile": cfg.services_profile,
            "services_used": sorted(supervisor.container._services),
            "pipeline_run_id": run_id,
            "run_uuid": run.run_id if run else None,
            "run_persisted": run is not None,
            "trace_id": trace_id,
            "trace_persisted": trace is not None,
            "trace_status": (trace.meta or {}).get("status") if trace else None,
            "trace_step_count": len(trace.execution_steps) if trace else None,
            "loaded_scorables": len(result.get("scorables", [])),
        }
        if (
            not run
            or not trace
            or summary["trace_status"] != "completed"
            or summary["trace_step_count"] != 1
        ):
            raise RuntimeError(f"STN01 persistence acceptance failed: {summary}")
        return summary
    finally:
        try:
            if supervisor is not None:
                await supervisor.container.shutdown()
        finally:
            if memory is not None:
                memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write one smoke Goal/PipelineRun/PlanTrace to the configured database",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        print("\nNo database connection or writes performed; rerun with --execute to run.")
        return 0

    fixture_parent = Path(
        os.environ.get(
            "STEPHANIE_SMOKE_TEMP_DIR",
            str(Path.home() / "AppData" / "Local" / "Temp" / "opencode"),
        )
    )
    fixture_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stephanie-stn01-", dir=fixture_parent) as temp_dir:
        fixture = Path(temp_dir) / "loader.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "problem_id": "stn01-loader-fixture",
                    "scorable": {
                        "external_id": "stn01-loader-fixture",
                        "text": "Deterministic STN01 loader pipeline fixture.",
                        "meta": {"is_correct": True},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = compose_smoke_config(temp_dir)
        if not database_url_from_config(cfg.db).password:
            raise SystemExit("Set STEPHANIE_DB_PASSWORD before --execute")
        print(json.dumps(asyncio.run(execute_smoke(cfg)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
