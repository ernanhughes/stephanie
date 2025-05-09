# co_ai/logs/json_logger.py
import json
from datetime import datetime, timezone
from pathlib import Path


class JSONLogger:
    EVENT_ICONS = {
        "PipelineStart": "🔬",
        "PipelineSuccess": "✅",
        "PipelineError": "❌",
        "IterationStart": "🔄",
        "IterationEnd": "🔚",
        "GeneratedHypotheses": "🧪",
        "ReflectedHypotheses": "🔎",
        "RankedHypotheses": "🏅",
        "EvolvedHypotheses": "🧬",
        "MetaReviewSummary": "📝",
        "debug": "💬"
    }
     
    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        icon = self.EVENT_ICONS.get(event_type, "📦")  # Default icon
        print(f"{icon} Logging event: {event_type} with data: {str(data)[:100]}")
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "data": data
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except TypeError as e:
            print(f"[Logger] Skipping non-serializable log: {e}")
            print(f"[Logger] Problematic record: {log_entry}")

