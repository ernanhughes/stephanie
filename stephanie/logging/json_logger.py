# stephanie/logs/json_logger.py
import json
from datetime import datetime, timezone
from pathlib import Path

from stephanie.logging.icons import get_event_icon


class JSONLogger:
    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str, extra: dict = None):
        """
        Log an informational message with optional extra data.
        
        Args:
            message: The log message
            extra: Additional data to include in the log
        """
        self.log("info", {"message": message, "extra": extra or {}})

    def exception(self, message: str, extra: dict = None):
        """
        Log an exception with a message and optional extra data.
        
        Args:
            message: The error message
            extra: Additional data to include in the log
        """
        self.log("exception", {"message": message, "extra": extra or {}})

    def error(self, message: str, extra: dict = None):
        """
        Log an error message with optional extra data.

        Args:
            message: The error message
            extra: Additional data to include in the log
        """
        self.log("error", {"message": message, "extra": extra or {}})

    def warning(self, message: str, extra: dict = None):
        """
        Log a warning message with optional extra data.
        
        Args:
            message: The warning message
            extra: Additional data to include in the log
        """
        self.log("warning", {"message": message, "extra": extra or {}})

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

    def get_logs_by_type(self, event_type: str) -> list:
        """
        Retrieve all logs of a specific type from the log file

        Args:
            event_type: The type of event to filter by

        Returns:
            List of matching log entries
        """
        if not self.log_path.exists():
            return []

        logs = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry["event_type"] == event_type:
                            logs.append(entry)
                    except json.JSONDecodeError:
                        continue  # Skip invalid lines
        except Exception as e:
            print(f"❌ [Logger] Failed to read logs: {str(e)}")
            return []

        return logs

    def get_all_logs(self) -> list:
        """
        Retrieve all logs from the file

        Returns:
            List of all log entries
        """
        if not self.log_path.exists():
            return []

        logs = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"❌ [Logger] Failed to read logs: {str(e)}")
            return []

        return logs
