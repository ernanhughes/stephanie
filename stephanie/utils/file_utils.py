# stephanie/utils/file_utils.py
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime

from pathlib import Path

log = logging.getLogger(__name__)

def save_to_timestamped_file(
    data: str,
    file_prefix: str = "config",
    file_extension: str = "yaml",
    output_dir: str = "logs",
) -> str:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_prefix}_{timestamp}.{file_extension}"
    filepath = output_dir_path / filename

    filepath.write_text(data, encoding="utf-8")

    # Absolute path
    full_path = filepath.resolve()

    # 1) Plain path (sometimes clickable)
    log.info("🔧 Saved file to %s", full_path)

    return str(full_path)


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_text_from_file(file_path: str) -> str:
    """Get text from a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def write_text_to_file(path: str, text: str):
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        log.debug(f"✅ Successfully wrote to {path}")
    except Exception as e:
        log.error(f"❌ Failed to write to {path}: {e}")


def save_json(data, path: str):
    """Save data to a JSON file"""
    import json

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        log.debug(f"✅ Successfully saved JSON to {path}")
    except Exception as e:
        log.error(f"❌ Failed to save JSON to {path}: {e}")


def load_json(path: str):
    """Load data from a JSON file"""
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.debug(f"✅ Successfully loaded JSON from {path}")
        return data
    except FileNotFoundError:
        log.error(f"❌ File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"❌ Failed to decode JSON from {path}: {e}")
        return None


def file_hash(path: str) -> str:
    """
    Compute SHA256 hash of a file's contents.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        SHA256 hash of file contents
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

