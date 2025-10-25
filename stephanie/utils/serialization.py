# stephanie/utils/serialization.py

"""
serialization.py
================
Utilities for efficient data serialization and compression.

This module provides:
- JSON-based serialization with custom object handling
- Zstandard compression for efficient network transmission
- Complementary decompression functionality
- Configurable compression levels for different use cases
- Error handling for serialization/deserialization

The implementation prioritizes:
- Speed: Fast compression/decompression for real-time telemetry
- Efficiency: Good compression ratios for telemetry payloads
- Reliability: Robust error handling and data integrity
"""

from __future__ import annotations

import json
import zlib
import zstandard as zstd
import logging
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

from stephanie.data.plan_trace import ExecutionStep, PlanTrace

log = logging.getLogger("stephanie.utils.serialization")

# Default compression level (1-22 for zstd, 0-9 for zlib)
# Level 3 provides good balance of speed/compression for telemetry
DEFAULT_COMPRESSION_LEVEL = 3

# Using zstandard as primary compressor (better speed/compression tradeoff than zlib)
_compression_ctx = zstd.ZstdCompressor(level=DEFAULT_COMPRESSION_LEVEL)


def compress_data(
    data: Any, compression_level: Optional[int] = None, use_zstd: bool = True
) -> bytes:
    """
    Compress data for efficient network transmission.

    Args:
        data: Data to compress (typically a dict or serializable object)
        compression_level: Optional compression level (1-22 for zstd, 0-9 for zlib)
        use_zstd: Whether to use zstandard (faster) or zlib (more compatible)

    Returns:
        Compressed bytes ready for network transmission

    Usage:
        payload = {"type": "telemetry", "data": {...}}
        compressed = compress_data(payload)
        await js.publish("subject", compressed)
    """
    try:
        # Serialize to JSON first
        json_data = json.dumps(data).encode("utf-8")

        # Apply compression
        if use_zstd:
            # Use pre-configured zstd compressor for speed
            return _compression_ctx.compress(json_data)
        else:
            # Fallback to zlib if zstd not available
            level = (
                compression_level
                if compression_level is not None
                else DEFAULT_COMPRESSION_LEVEL
            )
            return zlib.compress(json_data, level=level)

    except (TypeError, ValueError) as e:
        log.error(
            f"Serialization error in compress_data: {str(e)}", exc_info=True
        )
        raise
    except Exception as e:
        log.error(
            f"Compression error in compress_data: {str(e)}", exc_info=True
        )
        raise


def decompress_data(compressed_data: bytes, use_zstd: bool = True) -> Any:
    """
    Decompress data that was compressed with compress_data.

    Args:
        compressed_data: Compressed bytes to decompress
        use_zstd: Whether zstd was used for compression

    Returns:
        Original deserialized object

    Usage:
        async for msg in subscription:
            data = decompress_data(msg.data)
            process_telemetry(data)
    """
    try:
        # Decompress first
        if use_zstd:
            try:
                decompressed = _compression_ctx.decompress(compressed_data)
            except zstd.ZstdError:
                # Fall back to zlib if zstd fails (for backward compatibility)
                log.warning("Zstd decompression failed, trying zlib...")
                decompressed = zlib.decompress(compressed_data)
        else:
            decompressed = zlib.decompress(compressed_data)

        # Deserialize from JSON
        return json.loads(decompressed.decode("utf-8"))

    except (zstd.ZstdError, zlib.error) as e:
        log.error(
            f"Decompression error in decompress_data: {str(e)}", exc_info=True
        )
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        log.error(
            f"Deserialization error in decompress_data: {str(e)}",
            exc_info=True,
        )
        raise
    except Exception as e:
        log.error(
            f"Unexpected error in decompress_data: {str(e)}", exc_info=True
        )
        raise


def get_compression_ratio(original_data: Any, compressed_data: bytes) -> float:
    """
    Calculate the compression ratio for telemetry optimization.

    Args:
        original_data: Original data object
        compressed_data: Compressed bytes

    Returns:
        Compression ratio (higher = better compression)

    Usage:
        payload = {...}
        compressed = compress_data(payload)
        ratio = get_compression_ratio(payload, compressed)
        log.debug(f"Compression ratio: {ratio:.2f}x")
    """
    try:
        original_size = len(json.dumps(original_data).encode("utf-8"))
        compressed_size = len(compressed_data)
        return original_size / max(1, compressed_size)
    except Exception as e:
        log.error(f"Error calculating compression ratio: {str(e)}")
        return 1.0


def is_compressed(data: bytes) -> bool:
    """
    Check if data appears to be compressed (simple heuristic).

    Args:
        data: Data to check

    Returns:
        True if data appears compressed, False otherwise
    """
    # Simple heuristic: compressed data often starts with specific magic bytes
    if len(data) < 4:
        return False

    # Zstandard magic number: 28 B5 2F FD
    if data[:4] == b"\x28\xb5\x2f\xfd":
        return True

    # zlib header (CMF and FLG)
    if len(data) >= 2 and (
        data[0] == 0x78 and (data[1] in [0x01, 0x5E, 0x9C, 0xDA])
    ):
        return True

    # If data is significantly smaller than typical JSON would be
    try:
        # If it were JSON, minimum size for a simple object would be at least 2 bytes {}
        if len(data) < 2:
            return True
        # If data is binary-looking (not UTF-8 text)
        try:
            data.decode("utf-8")
            return False
        except UnicodeDecodeError:
            return True
    except Exception:
        return True

    return False


def to_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format, especially handling OmegaConf objects.

    Args:
        obj: Any object to convert

    Returns:
        JSON-serializable version of the object

    Example:
        >>> cfg = OmegaConf.create({"model": {"name": "llama", "layers": 24}})
        >>> serializable = to_serializable(cfg)
        >>> isinstance(serializable, dict)
        True
    """
    if obj is None:
        return None
    elif isinstance(obj, (int, float, bool, str)):
        return obj
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif OmegaConf.is_config(obj):
        # This is an OmegaConf object (DictConfig or ListConfig)
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    elif hasattr(obj, "to_dict"):
        # Custom objects with to_dict method
        return to_serializable(obj.to_dict())
    elif hasattr(obj, "tolist"):
        # Objects with tolist method (like numpy arrays)
        return obj.tolist()
    else:
        try:
            # Try to convert to string as last resort
            return str(obj)
        except:
            return "non-serializable-object"


def default_serializer(obj):
    """Handle serialization of complex objects including NumPy types"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, (ExecutionStep, PlanTrace)):
        return obj.to_dict()
    elif hasattr(obj, "_get_node"):  # OmegaConf DictConfig
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
    raise TypeError(f"Type {type(obj)} not serializable")
