# stephanie/utils/report_utils.py
from __future__ import annotations

from datetime import datetime


def get_stage_details(stage, status="⏳ starting") -> dict:
    return {
        "stage": stage.name,
        "agent": stage.cls.split(".")[-1],
        "status": status,
        "summary": stage.description,
        "metrics": {},   # filled later
        "outputs": {},   # filled later
        "start_time": datetime.now().strftime("%H:%M:%S"),
        "end_time": "",
        "error": "",
    }


def update_stage_report(report: dict, context: dict):
    report["status"] = context.get("status", "")
    report["end_time"] = datetime.now().strftime("%H:%M:%S")

    # Optionally capture metrics/outputs from context
    report["metrics"] = context.get("metrics", {})
    report["outputs"] = context.get("outputs", {})
    report["error"] = context.get("error", "")
