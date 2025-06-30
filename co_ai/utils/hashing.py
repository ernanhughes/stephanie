# co_ai/utils/hashing.py

import hashlib
import json
from typing import Any


def hash_dict(data: dict[str, Any], sort_keys: bool = True, exclude_keys: list = None) -> str:
    """
    Generate a SHA-256 hash of a dictionary.

    Useful for:
    - Caching
    - Deduplication
    - Versioning prompts, configs, traces
    - Context-aware symbolic rules

    Args:
        data (dict): Dictionary to hash.
        sort_keys (bool): Whether to sort keys for consistent output.
        exclude_keys (list): Optional list of keys to exclude before hashing.

    Returns:
        str: Hex digest of the hash.
    """
    if exclude_keys is None:
        exclude_keys = []

    # Filter out excluded keys
    filtered_data = {
        k: v for k, v in data.items()
        if k not in exclude_keys
    }

    # Convert to a canonical JSON string
    canonical_str = json.dumps(filtered_data, sort_keys=sort_keys, ensure_ascii=True)

    # Generate SHA-256 hash
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()