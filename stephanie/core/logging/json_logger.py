# stephanie/logs/json_logger.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List

from stephanie.core.logging.icons import get_event_icon


class JSONLineFormatter(logging.Formatter):
    """
    Formats records as one JSON object per line.
    Includes timestamp, level, event_type, and either `data` or a plain message.
    """
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event_type": getattr(record, "event_type", record.levelname.lower()),
        }

        # If caller passed a structured `data` object via `extra={"data": ...}`
        data = getattr(record, "data", None)
        if data is not None:
            payload["data"] = data
        else:
            payload["message"] = record.getMessage()

        # Capture exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Optionally include extra bound fields
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict) and extra_fields:
            payload.update(extra_fields)

        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Pretty console formatting that mimics your current stdout line:
    'ICON [event_type] preview-of-data'
    """
    def format(self, record: logging.LogRecord) -> str:
        event_type = getattr(record, "event_type", record.levelname.lower())
        icon = get_event_icon(event_type)
        data = getattr(record, "data", None)
        preview = str(data)[:100] if data is not None else record.getMessage()[:100]
        return f"{icon} [{event_type}] {preview}"


class JSONLogger:
    """
    Backward-compatible facade:
      - self.log(event_type, data)  -> INFO level structured event
      - self.info/error/warning/exception(message, extra)
      - JSONL persisted via RotatingFileHandler
      - Console output with icons
    """

    def __init__(
        self,
        log_path: str = "logs/pipeline_log.jsonl",
        *,
        level: int = logging.INFO,
        rotate_bytes: int = 10_000_000,   # ~10MB per file
        rotate_backups: int = 5,
        logger_name: str = "stephanie",
        enable_console: bool = True,
        enable_jsonl: bool = True,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        self._logger.propagate = False  # prevent double logging if root has handlers

        # Avoid duplicate handlers in hot-reload contexts
        if not self._logger.handlers:
            if enable_console:
                ch = logging.StreamHandler()
                ch.setLevel(level)
                ch.setFormatter(ConsoleFormatter())
                self._logger.addHandler(ch)

            if enable_jsonl:
                fh = RotatingFileHandler(
                    filename=str(self.log_path),
                    maxBytes=rotate_bytes,
                    backupCount=rotate_backups,
                    encoding="utf-8",
                )
                fh.setLevel(level)
                fh.setFormatter(JSONLineFormatter())
                self._logger.addHandler(fh)

    # ---------- Back-compat structured event API ----------

    def log(self, event_type: str, data: dict | None = None):
        """
        Structured event (default INFO).
        Keeps your existing call sites intact: self.logger.log("PlannerReuseGenerated", {...})
        """
        # 🚫 Skip empty logs
        if not data:
            return
        self._logger.debug("", extra={"event_type": event_type, "data": data})

    def info(self, message: str, extra: dict | None = None):
        # 🚫 Skip empty info calls
        if not message and not extra:
            return
        self._logger.debug(message, extra={"event_type": "info", "data": extra or {}})

    def warning(self, message: str, extra: dict | None = None):
        if not message and not extra:
            return
        self._logger.warning(message, extra={"event_type": "warning", "data": extra or {}})

    def error(self, message: str, extra: dict | None = None):
        self._logger.error(message, extra={"event_type": "error", "data": extra or {}})

    def exception(self, message: str, extra: dict | None = None):
        # exc_info=True captures stack trace
        self._logger.exception(message, extra={"event_type": "exception", "data": extra or {}})

    # ---------- JSONL readers (unchanged semantics) ----------

    def get_logs_by_type(self, event_type: str) -> List[dict]:
        if not self.log_path.exists():
            return []
        out: List[dict] = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("event_type") == event_type:
                            out.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"❌ [Logger] Failed to read logs: {str(e)}")
        return out

    def get_all_logs(self) -> List[dict]:
        if not self.log_path.exists():
            return []
        out: List[dict] = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"❌ [Logger] Failed to read logs: {str(e)}")
        return out
