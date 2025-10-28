#!/usr/bin/env python3
"""
Safely enforce two invariants across a Python codebase:

1) Header comment = "# <relative/path/from/root.py>" as the first logical line
   (after shebang and optional encoding comment; before module docstring).
2) The first import is exactly: "from __future__ import annotations"
   (inserted or hoisted to the canonical "after shebang/encoding/docstring" slot).

Safety:
- Never writes if the edited text fails `ast.parse`.
- Avoids duplicate `__future__` lines: removes existing `annotations` future(s) then inserts one canonical line.
- Leaves files unchanged if we can't safely edit.

Usage:
  python tools/check_required_lines.py
  python tools/check_required_lines.py --root stephanie --fix
  python tools/check_required_lines.py --exclude tests --exclude .venv
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import List, Tuple, Optional

ENCODING_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="stephanie", help="Root directory to scan (default: stephanie)")
    p.add_argument("--exclude", action="append",
                   default=[".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".git"],
                   help="Directories to exclude (repeatable).")
    p.add_argument("--fix", action="store_true",
                   help="Apply safe edits. Without this flag, runs read-only and reports.")
    return p.parse_args()

def should_exclude(path: str, excludes: List[str]) -> bool:
    for ex in excludes:
        if f"{os.sep}{ex}{os.sep}" in path or path.endswith(os.sep + ex) or path == ex:
            return True
    return False

def read_text(path: str) -> Optional[str]:
    # Try utf-8 first
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        pass
    # Fallback: detect encoding from the first two lines per PEP 263
    try:
        with open(path, "rb") as fb:
            b = fb.read()
        enc = "utf-8"
        head = b.splitlines()[:2]
        for line in head:
            m = ENCODING_RE.search(line)
            if m:
                enc = m.group(1).decode("ascii", "ignore")
                break
        return b.decode(enc, errors="ignore")
    except Exception:
        return None

def write_text_atomic(path: str, text: str) -> None:
    # Atomic-ish: write to temp then replace
    d = os.path.dirname(path)
    base = os.path.basename(path)
    tmp = os.path.join(d, f".{base}.tmp_edit")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

def detect_docstring_span(src: str) -> Tuple[int, int]:
    """
    Return (start_lineno, end_lineno) for a top-level module docstring if present, else (0,0).
    """
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return (0, 0)
    if not getattr(mod, "body", None):
        return (0, 0)
    first = mod.body[0]
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        start = getattr(first, "lineno", 1)
        end = getattr(first, "end_lineno", start)
        return (start, end)
    return (0, 0)

def find_header_insert_index(lines: List[str]) -> int:
    """
    Insert header after shebang and optional encoding line, but BEFORE the module docstring.
    Comments before the docstring are allowed and don't affect docstring semantics.
    """
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    # encoding (must be in first two lines per PEP 263)
    enc_limit = min(2, len(lines))
    for j in range(i, enc_limit):
        if "coding:" in lines[j] or "coding=" in lines[j]:
            i = j + 1
    return i

def find_future_insert_index(lines: List[str]) -> int:
    """
    Insert the future import after shebang/encoding/docstring.
    """
    i = 0
    if lines and lines[0].startswith("#!"):
        i = 1
    # encoding (first two lines)
    enc_limit = min(2, len(lines))
    for j in range(i, enc_limit):
        if "coding:" in lines[j] or "coding=" in lines[j]:
            i = j + 1
    # after module docstring, if any
    src = "".join(lines)
    _start, end = detect_docstring_span(src)
    if end:
        i = max(i, end)  # docstring end is 1-based
    return i

def expected_header(root: str, path: str) -> str:
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    return f"# {rel}"

def has_header_at_top(lines: List[str], header: str) -> bool:
    # Check the first few non-empty comment/blank lines after shebang/encoding
    idx = find_header_insert_index(lines)
    window = lines[idx: idx + 5]
    for ln in window:
        s = ln.strip()
        if not s or s.startswith("#"):
            if s == header:
                return True
        else:
            break  # hit real code/text before finding header
    return False

def remove_existing_future_annotations(lines: List[str]) -> Tuple[List[str], bool]:
    """
    Remove any existing 'from __future__ import annotations' statements.
    Returns (new_lines, removed_any).
    """
    src = "".join(lines)
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return (lines, False)

    to_remove: List[Tuple[int, int]] = []
    for node in mod.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            names = {alias.name for alias in node.names}
            if "annotations" in names:
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                if start > 0 and end >= start:
                    to_remove.append((start, end))

    if not to_remove:
        return (lines, False)

    # Remove from bottom to top to keep indices stable
    new_lines = lines[:]
    for (start, end) in sorted(to_remove, key=lambda x: x[0], reverse=True):
        # Convert to 0-based slicing
        del new_lines[start - 1:end]

    return (new_lines, True)

def ensure_header_and_future(src: str, root: str, path: str) -> Tuple[str, dict]:
    """
    Returns (new_src, actions) where actions is a dict of booleans:
      {
        'added_header': bool,
        'inserted_future': bool,
        'removed_existing_future': bool,
        'changed': bool
      }
    """
    actions = {
        "added_header": False,
        "inserted_future": False,
        "removed_existing_future": False,
        "changed": False,
    }

    lines = src.splitlines(keepends=True)
    hdr = expected_header(root, path)

    # 1) Insert header if missing
    if not has_header_at_top(lines, hdr):
        idx = find_header_insert_index(lines)
        lines[idx:idx] = [hdr + "\n"]
        actions["added_header"] = True

    # 2) Remove existing __future__ annotations (avoid duplicates)
    lines, removed = remove_existing_future_annotations(lines)
    actions["removed_existing_future"] = removed

    # 3) Insert canonical future import
    #    Always (re)insert—either it was absent or we removed old ones to hoist it
    ins_idx = find_future_insert_index(lines)
    future_line = "from __future__ import annotations\n"
    # If the next non-empty, non-comment line already is the exact future import, skip insert
    ahead = "".join(lines[ins_idx: ins_idx + 3])
    if "from __future__ import annotations" not in ahead:
        lines[ins_idx:ins_idx] = [future_line]
        actions["inserted_future"] = True

    new_src = "".join(lines)
    actions["changed"] = (new_src != src)
    return new_src, actions

def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.root)

    scanned = 0
    updated = 0
    unsafe_skipped = 0
    unchanged = 0
    read_errors = 0

    changed_files: List[str] = []
    skipped_files: List[Tuple[str, str]] = []  # (path, reason)

    for dirpath, dirnames, filenames in os.walk(root):
        if should_exclude(dirpath + os.sep, args.exclude):
            dirnames[:] = [d for d in dirnames if not should_exclude(os.path.join(dirpath, d), args.exclude)]
            continue

        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            scanned += 1

            text = read_text(path)
            if text is None:
                read_errors += 1
                skipped_files.append((path, "read_error"))
                continue

            # Prepare edit
            new_text, actions = ensure_header_and_future(text, root, path)

            # If nothing would change, record and move on
            if not actions["changed"]:
                unchanged += 1
                continue

            # Safety: Post-edit must be parseable
            try:
                ast.parse(new_text)
            except SyntaxError as e:
                unsafe_skipped += 1
                skipped_files.append((path, f"post_edit_syntax_error: {e}"))
                continue

            if args.fix:
                try:
                    write_text_atomic(path, new_text)
                except Exception as e:
                    unsafe_skipped += 1
                    skipped_files.append((path, f"write_error: {e}"))
                    continue
                updated += 1
                changed_files.append(path)
            else:
                # Dry run: pretend we'd change it, but do not write
                updated += 1
                changed_files.append(path)

    # Reporting
    mode = "APPLY" if args.fix else "DRY-RUN"
    print(f"\n[{mode}] Scanned {scanned} Python file(s) under {root}.")
    print(f" - Would update: {updated}{' (applied)' if args.fix else ''}")
    print(f" - Unchanged:    {unchanged}")
    print(f" - Skipped:      {unsafe_skipped} (unsafe or errors)")
    if read_errors:
        print(f" - Read errors:  {read_errors}")

    if changed_files:
        print("\nChanged files:")
        for p in changed_files:
            print(f"  • {p}")

    if skipped_files:
        print("\nSkipped files (reason):")
        for p, reason in skipped_files:
            print(f"  • {p} — {reason}")

    # Exit non-zero if in dry-run and any changes would be needed
    if not args.fix and changed_files:
        return 1
    # Exit non-zero if apply mode had any unsafe skips or write errors
    if args.fix and skipped_files:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
