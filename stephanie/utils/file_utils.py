# stephanie/utils/file_utils.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from stephanie.data.plan_trace import ExecutionStep, PlanTrace

log = logging.getLogger(__name__)


def save_to_timestamped_file(
    data: str,
    file_prefix: str = "config",
    file_extension: str = "yaml",
    output_dir: str = "logs",
    last_filename: str | None = None,
) -> str:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_prefix}_{timestamp}.{file_extension}"
    filepath = output_dir_path / filename

    write_text_to_file(str(filepath), data)

    # NEW: stable last copy (e.g. "last_report.md", "blog_config_last.yaml")
    if last_filename:
        try:
            write_last_copy(
                source_path=filepath,
                last_path=Path(output_dir) / last_filename,
            )
        except Exception:
            log.warning(
                "Failed to write last copy for %s", filepath, exc_info=True
            )

    return str(filepath)


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def save_yaml_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", ".yaml")
    with open(report_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)
    log.info(f"✅ Result saved to: {report_path}")


def default_serializer(obj):
    from dataclasses import asdict, is_dataclass
    from datetime import date, datetime
    from enum import Enum
    from pathlib import Path

    import numpy as np

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return (
            obj.value
            if isinstance(obj.value, (str, int, float, bool))
            else str(obj.value)
        )

    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Prefer explicit model encoders first
    if isinstance(obj, ExecutionStep):
        return obj.to_dict()
    if isinstance(obj, PlanTrace):
        return obj.to_dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    # Then dataclasses (generic)
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)

    # Hydra DictConfig
    if hasattr(obj, "_get_node"):
        return OmegaConf.to_container(obj, resolve=True, enum_to_str=True)

    raise TypeError(f"Type {type(obj)} not serializable")


def save_context_result(log_path: str, result: dict):
    report_path = log_path.replace(".jsonl", "_context.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            result, f, ensure_ascii=False, indent=2, default=default_serializer
        )
    log.info(f"✅ JSON result saved to: {report_path}")


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


def write_last_copy(
    *,
    source_path: str | Path,
    last_path: str | Path,
    prefer_link: bool = False,
) -> Path:
    """
    Create/replace a stable 'last' copy of an artifact.

    Tries, in order:
      1) symlink (best, but may require privileges on Windows)
      2) hardlink (fast, same filesystem)
      3) copy2 + atomic replace (works everywhere)

    Returns the final last_path.
    """
    src = Path(source_path)
    dst = Path(last_path)

    if not src.exists():
        raise FileNotFoundError(
            f"write_last_copy: source does not exist: {src}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing destination (file or symlink)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    except Exception:
        # If unlink fails (e.g. permission edge-case), try best-effort overwrite later
        pass

    # 1) Symlink (best if available)
    if prefer_link:
        try:
            # Use absolute path to avoid surprises with CWD changes
            dst.symlink_to(src.resolve())
            return dst
        except Exception:
            pass

        # 2) Hardlink (fast, but same filesystem)
        try:
            os.link(str(src), str(dst))
            return dst
        except Exception:
            pass

    # 3) Copy + atomic replace
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=f".{dst.name}.",
            suffix=".tmp",
            dir=str(dst.parent),
        )
        os.close(tmp_fd)
        tmp_fd = None

        shutil.copy2(str(src), str(tmp_path))
        os.replace(str(tmp_path), str(dst))
        tmp_path = None
        return dst
    finally:
        # Cleanup temp if something went wrong
        try:
            if tmp_fd is not None:
                os.close(tmp_fd)
        except Exception:
            pass
        try:
            if tmp_path is not None and Path(tmp_path).exists():
                Path(tmp_path).unlink()
        except Exception:
            pass
