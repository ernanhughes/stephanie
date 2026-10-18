"""STN06 reconciliation readback.

Reads the persisted STN04/STN05 records through the normal MemoryTool path and
prints a single JSON document so the conflicting STN05 status reports can be
reconciled against the real ChatStore, CaseBook and MemCube history.

Read-only. It never mutates a record.
"""

from __future__ import annotations

import json
import os
import sys

from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

CONVERSATION_ID = 1667
EXECUTION_ID = "sex_8991baa0c2724f35878292c83651588f"
EXPERIENCE_ID = "exp_33fd63f133b44b21275681e5af0aabb7"
MEMCUBE_ID = f"{EXPERIENCE_ID}:v1"


class NullLogger:
    log_path = None

    def log(self, *args, **kwargs):
        return None


def build_report(memory) -> dict:
    record = memory.chats.get_external_execution(CONVERSATION_ID, EXECUTION_ID)
    candidate_row = memory.casebooks.get_execution_experience_candidate(EXPERIENCE_ID)
    cube = memory.memcubes.get_by_id(MEMCUBE_ID)

    candidate = (candidate_row or {}).get("candidate") or {}
    extra = (cube.extra_data or {}) if cube is not None else {}
    report = {
        "conversation_id": CONVERSATION_ID,
        "execution_id": EXECUTION_ID,
        "execution": {
            "status": (record or {}).get("status"),
            "human_execution_review": ((record or {}).get("human_review") or {}).get("decision"),
            "human_review_history_len": len((record or {}).get("human_review_history") or []),
            "worker_acceptance_id": (record or {}).get("acceptance_id"),
            "experience_id": (record or {}).get("experience_id"),
            "experience_status": (record or {}).get("experience_status"),
        },
        "casebook": {
            "found": candidate_row is not None,
            "casebook_id": (candidate_row or {}).get("casebook_id"),
            "candidate_review_status": candidate.get("candidate_review_status"),
            "candidate_review_decision": (candidate.get("candidate_review") or {}).get("decision"),
            "candidate_hash": candidate.get("candidate_hash"),
            "reuse_eligible": candidate.get("reuse_eligible"),
            "promotion_status": candidate.get("promotion_status"),
            "deactivation_history": candidate.get("deactivation_history") or [],
            "lesson_text": (candidate.get("lesson_hypothesis") or {}).get("text"),
            "lesson_validation_status": (candidate.get("lesson_hypothesis") or {}).get("validation_status"),
            "applicability": (candidate.get("lesson_hypothesis") or {}).get("applicability"),
            "limitations": (candidate.get("lesson_hypothesis") or {}).get("limitations"),
            "source": candidate.get("source"),
        },
        "memcube": {
            "found": cube is not None,
            "id": cube.id if cube is not None else None,
            "source": cube.source if cube is not None else None,
            "sensitivity": cube.sensitivity if cube is not None else None,
            "reuse_eligible": extra.get("reuse_eligible"),
            "promotion_status": extra.get("promotion_status"),
        },
        "active_retrieval_eligibility": bool(
            cube is not None
            and extra.get("reuse_eligible") is True
            and extra.get("promotion_status") == "ACTIVE"
        ),
    }
    return report


def main() -> int:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="sis_stn04_local")
    from stephanie.memory.memory_tool import MemoryTool

    memory = MemoryTool(cfg, NullLogger())
    try:
        report = build_report(memory)
    finally:
        memory.close()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
