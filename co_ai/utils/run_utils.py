import os
import re
import uuid
from datetime import datetime, timezone

from omegaconf import DictConfig


def generate_run_id(goal: str) -> str:
    # Extract keywords from goal
    keywords = re.findall(r'\b\w{5,}\b', goal.lower())  # words with 5+ letters
    keywords = keywords[:2] if keywords else ['run']

    # Sanitize and slugify
    slug = "_".join(keywords)
    slug = re.sub(r'[^a-z0-9_]+', '', slug)

    # Add timestamp and short UUID
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    short_uuid = uuid.uuid4().hex[:6]

    return f"{slug}_{timestamp}_{short_uuid}"

def get_log_file_path(run_id:str, cfg: DictConfig) -> str:
    # Get the path to the log file
    if cfg.logging.logger.get("log_file", None):
        print(f"Log file path: {cfg.logging.logger.log_file}")
        return cfg.logging.logger.log_file
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    safe_run_id = re.sub(r"[\\W_]+", "_", run_id)  # remove/replace unsafe chars
    log_filename = f"{safe_run_id}_{timestamp}.jsonl"
    os.makedirs(cfg.logging.logger.log_path, exist_ok=True)
    log_file_path = os.path.join(cfg.logging.logger.log_path, log_filename)
    print(f"Log file path: {log_file_path}")
    return log_file_path

