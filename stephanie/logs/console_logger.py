import sys
from datetime import datetime, timezone

from stephanie.logs.icons import get_event_icon


class ConsoleLogger:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout  # default to stdout

    def log(self, event_type: str, data: dict):
        icon = get_event_icon(event_type)
        truncated = str(data)[:100]

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            print(f"{icon} [{event_type}] {truncated}", file=self.stream)
            print(f"🕒  {log_entry['timestamp']}", file=self.stream)
        except Exception as e:
            print("❌ [ConsoleLogger] Failed to print log entry.", file=sys.stderr)
            print(f"🛠️  Event Type: {event_type}", file=sys.stderr)
            print(f"🪵  Error: {e}", file=sys.stderr)
            print(f"🧱  Data: {repr(data)[:200]}", file=sys.stderr)
