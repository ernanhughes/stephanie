"""STN06 preflight: freeze, preregister and dry-run without paying for inference.

Usage (read-only by default)::

    venv\\Scripts\\python.exe scripts\\stn06_preflight.py --out docs/stephanie-next/stn06-preflight.json

To persist the frozen preregistration and an experiment-scoped authorization
into the existing ExperimentStore (still no model call, no MemCube mutation)::

    venv\\Scripts\\python.exe scripts\\stn06_preflight.py \\
        --persist --authorization-id stn06-exp-scope-001 --reviewer operator \\
        --notes "Authorize the frozen STN05 experience for the STN06 pilot scope only."

The withdrawn MemCube is never reactivated and no Worker/OpenCode request is sent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


class NullLogger:
    log_path = None

    def log(self, *args, **kwargs):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="STN06 deterministic preflight")
    parser.add_argument("--config", default="sis_stn06_local")
    parser.add_argument("--scope", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--out", default="")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--authorization-id", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=args.config)

    from stephanie.memory.memory_tool import MemoryTool
    from stephanie.services.execution_experience import ExecutionExperienceService
    from stephanie.services.stn06_experiment import Stn06ExperimentService

    memory = MemoryTool(cfg, NullLogger())
    try:
        experience_service = ExecutionExperienceService(memory=memory, cfg=cfg, logger=NullLogger())
        service = Stn06ExperimentService(
            memory=memory, cfg=cfg, logger=NullLogger(), experience_service=experience_service,
        )
        report = {
            "config": args.config,
            "enabled": service._settings().get("enabled", False),
            "freeze": service.freeze_experience(),
            "authorization": None,
            "persisted": None,
            "dry_run": None,
        }
        if args.persist:
            if not (args.authorization_id and args.reviewer and args.notes):
                raise SystemExit("--persist requires --authorization-id, --reviewer and --notes")
            authorization = service.authorize_scope(
                reviewer=args.reviewer,
                authorization_id=args.authorization_id,
                notes=args.notes,
            )
            report["authorization"] = authorization
            report["persisted"] = service.persist_preregistration(authorization=authorization)
        report["dry_run"] = service.dry_run(scope=args.scope, verify_oracles=True)
    finally:
        memory.close()

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    summary = {
        "enabled": report["enabled"],
        "manifest_hash": report["dry_run"]["manifest_hash"],
        "preregistration_hash": report["dry_run"]["preregistration_hash"],
        "frozen_experience_id": report["freeze"]["experience_id"],
        "candidate_hash": report["freeze"]["candidate_hash"],
        "globally_retrievable": report["freeze"]["globally_retrievable"],
        "model_calls_made": report["dry_run"]["model_calls_made"],
        "trials_persisted": report["dry_run"]["trials_persisted"],
        "oracle_all_verified": report["dry_run"]["oracle_verification"]["all_verified"],
        "normal_retrieval_withdrawn_unavailable": (
            (report["dry_run"]["normal_retrieval"] or {}).get("withdrawn_unavailable")
        ),
        "planned_executions": report["dry_run"]["planned_executions"],
        "out": args.out or None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
