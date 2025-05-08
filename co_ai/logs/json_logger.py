# co_ai/logs/json_logger.py
import json
from datetime import datetime
from pathlib import Path


class JSONLogger:
    def __init__(self, log_path="logs/pipeline_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record):
        timestamp = datetime.utcnow().isoformat()
        log_entry = {"timestamp": timestamp, **record}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
