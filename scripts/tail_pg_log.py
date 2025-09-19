#!/usr/bin/env python3
"""
Tail the last N lines of a PostgreSQL log on Windows.

Usage examples:
  python tail_pg_log.py --path "e:\\Program Files\\PostgreSQL\\16\\data\\log"
  python tail_pg_log.py --path "e:\\Program Files\\PostgreSQL\\16\\data\\log"
  python tail_pg_log.py --path "e:\\logs\\postgresql-2025-09-19_000000.log" -n 100
"""

import argparse
import os
import sys
import glob
from datetime import datetime

def pick_latest_log_from_dir(dir_path: str) -> str | None:
    # Common Postgres patterns; extend if your naming differs
    patterns = [
        os.path.join(dir_path, "postgresql-*.log"),
        os.path.join(dir_path, "pg_log", "postgresql-*.log"),  # some installs
        os.path.join(dir_path, "*.log"),  # fallback
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    # Pick the most recently modified file
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def tail(file_path: str, n: int = 100, chunk_size: int = 8192) -> list[str]:
    """Efficiently read the last n lines of a text file (Windows-safe)."""
    # Open in binary to count bytes reliably, then decode once at the end
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size == 0:
            return []

        data = bytearray()
        bytes_needed = 1  # force at least one read
        lines_found = 0
        pos = file_size

        while pos > 0 and lines_found <= n:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(read_size)
            data[:0] = chunk  # prepend
            lines_found = data.count(b"\n")
            # Increase chunk size gradually for huge files
            if lines_found <= n and chunk_size < 2**20:
                chunk_size *= 2

        # Decode with replacement to avoid errors on partial/mixed encodings
        text = data.decode(errors="replace")
        lines = text.splitlines()
        return lines[-n:]

def main():
    parser = argparse.ArgumentParser(description="Print last N lines of a PostgreSQL log on Windows.")
    parser.add_argument("--path", required=True, help="Path to log file OR directory containing logs.")
    parser.add_argument("-n", "--lines", type=int, default=100, help="Number of lines to print (default: 100).")
    args = parser.parse_args()

    target_path = args.path
    if not os.path.exists(target_path):
        print(f"ERROR: Path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(target_path):
        log_file = pick_latest_log_from_dir(target_path)
        if not log_file:
            print(f"ERROR: No .log files found in directory: {target_path}", file=sys.stderr)
            sys.exit(2)
        print(f"# Using latest log: {log_file}")
    else:
        log_file = target_path

    try:
        lines = tail(log_file, n=args.lines)
    except PermissionError:
        print(f"ERROR: Permission denied reading file: {log_file}", file=sys.stderr)
        sys.exit(3)
    except OSError as e:
        print(f"ERROR: Failed to read file: {log_file}\n{e}", file=sys.stderr)
        sys.exit(4)

    # Nice header with timestamp
    print(f"# --- Last {args.lines} lines of {log_file} @ {datetime.now().isoformat(timespec='seconds')} ---")
    for line in lines:
        print(line)

if __name__ == "__main__":
    main()
