# stephanie/utils/file_utils.py
from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime

from omegaconf import OmegaConf

_logger = logging.getLogger(__name__)

def save_to_timestamped_file(data, file_prefix:str = "config", file_extension: str = "yaml", output_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamped_name = f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
    filepath = os.path.join(output_dir, timestamped_name)
    with open(filepath, "w", encoding="utf-8") as f:  
        f.write(data)
    _logger.info(f"🔧 Saved config to {filepath}")
    return filepath


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
        print(f"✅ Successfully wrote to {path}")
    except Exception as e:
        print(f"❌ Failed to write to {path}: {e}")


def save_json(data, path: str):
    """Save data to a JSON file"""
    import json

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Successfully saved JSON to {path}")
    except Exception as e:
        print(f"❌ Failed to save JSON to {path}: {e}")


def load_json(path: str):
    """Load data from a JSON file"""
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Successfully loaded JSON from {path}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to decode JSON from {path}: {e}")
        return None


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash for the given text using the specified algorithm.

    Args:
        text (str): The input text to hash.
        algorithm (str): Hash algorithm, e.g., 'sha256', 'sha1', or 'md5'.

    Returns:
        str: The hexadecimal digest of the hash.
    """
    if not text:
        return ""

    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()
