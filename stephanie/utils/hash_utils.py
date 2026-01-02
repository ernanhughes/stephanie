# stephanie/utils/hash_utils.py
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, List

log = logging.getLogger(__name__)

def _hash(data: bytes, algo: str = "sha256") -> str:
    """Internal helper that does the actual hashing."""
    hasher = hashlib.new(algo)
    hasher.update(data)
    return hasher.hexdigest()


def hash_bytes(data: bytes, algo: str = "sha256") -> str:
    """
    Hash raw bytes with the given algorithm.

    Usage:
        digest = hash_bytes(b"...")
    """
    if data is None:
        data = b""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"hash_bytes expected bytes-like, got {type(data)}")
    # normalize to bytes
    if isinstance(data, bytearray) or isinstance(data, memoryview):
        data = bytes(data)
    return _hash(data, algo=algo)


def hash_dict(
    data: dict[str, Any], sort_keys: bool = True, exclude_keys: list = None
) -> str:
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
    if isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []

        # Filter out excluded keys
        filtered_data = {k: v for k, v in data.items() if k not in exclude_keys}

        # Convert to a canonical JSON string
        canonical_str = json.dumps(filtered_data, sort_keys=sort_keys, ensure_ascii=True)
    else:
        log.warning(f"hash_dict expected dict, got {type(data)}; using str()")
        canonical_str = str(data)
    # Generate SHA-256 hash
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate a hash for the given text using the specified algorithm.

    Args:
        text (str): The input text to hash.
        algorithm (str): Hash algorithm, e.g., 'sha256', 'sha1', or 'md5'.

    Returns:
        str: The hexadecimal digest of the hash.

    Usage:
        digest = hash_text("hello world")
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError(f"hash_text expected str, got {type(text)}")
    return _hash(text.encode("utf-8"), algo=algorithm)

def hash_list(names: List[str]) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update((n + "\n").encode("utf-8"))
    return h.hexdigest()[:16]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

