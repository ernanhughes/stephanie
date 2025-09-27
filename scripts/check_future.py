#!/usr/bin/env python3
"""
Checks every Python file under a root (default: ./stephanie) for required lines.
- By default, verifies: "from __future__ import annotations"
- You can add more required literal lines or regex patterns.
- Optional --fix will insert the missing *future import* in the correct spot:
  after shebang, encoding comment, and top-level module docstring.

Examples:
  python tools/check_required_lines.py
  python tools/check_required_lines.py --root stephanie --require "from __future__ import annotations"
  python tools/check_required_lines.py --regex "^from __future__ import (.*\\bannotations\\b.*)$"
  python tools/check_required_lines.py --fix   # auto-insert future import when missing
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import List, Tuple

ENCODING_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="stephanie", help="Root directory to scan (default: stephanie)")
    p.add_argument("--require", action="append", default=["from __future__ import annotations"],
                   help="Literal line that must appear somewhere in the file. Repeatable.")
    p.add_argument("--regex", action="append", default=[],
                   help="Regex pattern that must match somewhere in the file (text). Repeatable.")
    p.add_argument("--fix", action="store_true",
                   help="If missing future import, insert it in the correct position.")
    p.add_argument("--exclude", action="append", default=[".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"],
                   help="Directories to exclude (repeatable).")
    return p.parse_args()

def should_exclude(path: str, excludes: List[str]) -> bool:
    for ex in excludes:
        if f"{os.sep}{ex}{os.sep}" in path or path.endswith(os.sep + ex) or path == ex:
            return True
    return False

def has_literal_line(text: str, line: str) -> bool:
    # search exact line boundary-insensitive to leading/trailing whitespace
    target = line.strip()
    for ln in text.splitlines():
        if ln.strip() == target:
            return True
    return False

def has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE) is not None

def detect_docstring_end_lineno(src: str) -> int:
    """
    If the file has a top-level module docstring, return its end lineno (1-based).
    Else return 0.
    """
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return 0
    if not getattr(mod, "body", None):
        return 0
    first = mod.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        # Python 3.8+ has end_lineno
        return getattr(first, "end_lineno", first.lineno)
    return 0

def find_insertion_index(lines: List[str]) -> int:
    """
    Compute the best insertion index (0-based) for future imports:
    - after shebang (#!...)
    - after encoding comment (# -*- coding: utf-8 -*- / # coding: utf-8)
    - after module docstring (triple-quoted string as the first statement)
    """
    i = 0
    # shebang
    if lines and lines[0].startswith("#!"):
        i = 1

    # possible encoding comment (must be within first two lines per PEP 263)
    # check raw bytes; but we have str, so a quick heuristic:
    enc_limit = min(2, len(lines))
    for j in range(i, enc_limit):
        if "coding:" in lines[j] or "coding=" in lines[j]:
            i = j + 1

    # module docstring: use AST to find end line
    src = "".join(lines)
    end_lineno = detect_docstring_end_lineno(src)
    if end_lineno:
        i = max(i, end_lineno)  # end_lineno is 1-based

    return i

def insert_future_import(text: str, future_line: str) -> str:
    """
    Insert the given future import after shebang/encoding/docstring.
    Avoid duplicate insertion.
    """
    if has_literal_line(text, future_line):
        return text
    lines = text.splitlines(keepends=True)
    idx = find_insertion_index(lines)
    ins = future_line.strip() + "\n"
    # ensure a blank line after imports if needed
    lines[idx:idx] = [ins]
    return "".join(lines)

def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.root)

    missing: List[Tuple[str, List[str]]] = []
    fixed_count = 0
    scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        if should_exclude(dirpath + os.sep, args.exclude):
            # prune walk
            dirnames[:] = [d for d in dirnames if not should_exclude(os.path.join(dirpath, d), args.exclude)]
            continue

        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                # try binary then decode with fallback
                try:
                    with open(path, "rb") as fb:
                        b = fb.read()
                    # naive utf-8 fallback; if it fails, skip
                    text = b.decode("utf-8", errors="ignore")
                except Exception:
                    continue

            scanned += 1
            failed: List[str] = []

            for lit in args.require:
                if not has_literal_line(text, lit):
                    failed.append(f'Missing required line: {lit!r}')

            for rx in args.regex:
                if not has_regex(text, rx):
                    failed.append(f'Missing pattern: /{rx}/')

            # Optional fix: only for the canonical future import
            if args.fix and any("from __future__ import annotations" in msg for msg in failed):
                new_text = insert_future_import(text, "from __future__ import annotations")
                if new_text != text:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    fixed_count += 1
                    # re-check the literal requirement afterwards
                    failed = [m for m in failed if "from __future__ import annotations" not in m]

            if failed:
                missing.append((path, failed))

    # Report
    if missing:
        print(f"\nChecked {scanned} files under {root}.")
        print(f"{len(missing)} files missing requirements:")
        for path, errs in missing:
            print(f" - {path}")
            for e in errs:
                print(f"    · {e}")
    else:
        print(f"All good. Checked {scanned} files; all requirements satisfied.")

    if args.fix:
        print(f"\nAuto-fix inserted 'from __future__ import annotations' into {fixed_count} file(s).")

    return 1 if missing else 0

if __name__ == "__main__":
    sys.exit(main())
