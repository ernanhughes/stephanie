import json
from datetime import datetime, timezone
from pathlib import Path
from co_ai.logs.icons import get_event_icon

class JSONLogger:

    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, data: dict):
        icon = get_event_icon(event_type)
        print(f"{icon} [{event_type}] {str(data)[:100]}")

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                json.dump(log_entry, f, default=str)
                f.write("\n")
        except (TypeError, ValueError) as e:
            print("❌ [Logger] Failed to serialize log entry.")
            print(f"🛠️  Event Type: {event_type}")
            print(f"🪵  Error: {e}")
            print(f"🧱  Data: {repr(data)[:200]}")
