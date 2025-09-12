# stephanie/cli/run_for_document.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from stephanie.agents.paper_improver.orchestrator import run_paper_section
from stephanie.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.tools.plan_from_document import (build_plan_from_memory,
                                                save_plan)
from stephanie.tools.spec_from_document import build_spec_from_text, save_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--document-id", type=int, required=True)
    ap.add_argument("--workdir", default="./runs")
    ap.add_argument("--backend", default="torch", choices=["torch","numpy"])
    ap.add_argument("--create-pr", action="store_true")
    args = ap.parse_args()

    # NOTE: you’ll provide `get_app_memory()` from your app context or a thin loader
    def load_config(path="/config/config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    cfg = load_config()
    logger = JSONLogger("logs/sis.jsonl")
    memory = MemoryTool(cfg=cfg, logger=logger)


    plan = build_plan_from_memory(memory, args.document_id)
    doc = memory.documents.get(args.document_id)
    spec = build_spec_from_text((doc.text or "") + " " + (plan.get("section_title") or ""))

    # paths
    run_dir = Path(args.workdir) / f"doc_{args.document_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = save_plan(plan, str(run_dir / "plan.json"))
    spec_path = save_spec(spec, str(run_dir / "spec.json"))

    report = run_paper_section(
        spec_path=str(spec_path),
        plan_path=str(plan_path),
        workdir=str(run_dir),
        backend=args.backend,
        create_pr=args.create_pr,
        repo_root="../..",
    )
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
