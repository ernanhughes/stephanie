#!/usr/bin/env python3
"""
Fix AppContext import in agents that already use the new constructor.

It searches for files under the given root (default: ./stephanie)
that contain a constructor like:
    def __init__(self, app: AppContext):

If found, it checks for the exact import line:
    from stephanie.core.app_context import AppContext

If missing, it inserts that import at the end of the current top-of-file
import section (after shebang/encoding/docstring and any multi-line imports).

Usage:
  python tools/fix_appcontext_imports.py                  # dry-run
  python tools/fix_appcontext_imports.py --apply          # write changes
  python tools/fix_appcontext_imports.py stephanie --apply
  python tools/fix_appcontext_imports.py stephanie --apply --import-line "from stephanie.app.context import AppContext"

Options:
  --apply            Actually write changes (default is dry-run)
  --import-line      Override the import line (default shown above)
  --pattern          Override the constructor regex to search for
  --backup-ext       If set (e.g. ".bak"), write a backup copy before modifying
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_IMPORT_LINE = "from stephanie.core.app_context import AppContext"
DEFAULT_PATTERN = r"def\s+__init__\s*\(\s*self\s*,\s*app\s*:\s*AppContext\s*\)\s*:"

def has_target_constructor(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None

def has_import_line(text: str, import_line: str) -> bool:
    # Exact substring match is what the user asked for.
    return import_line in text

def find_docstring_end_lineno(text: str) -> int:
    """
    Return the last line number of the module docstring (1-based), or 0 if none.
    Uses AST to handle triple-quoted strings reliably.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    if tree.body and isinstance(tree.body[0], ast.Expr):
        val = tree.body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
        # Py<3.8 compatibility: ast.Str
        if hasattr(ast, "Str") and isinstance(val, getattr(ast, "Str")):
            return getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
    return 0

def is_import_start(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("import ") or s.startswith("from ")

def is_comment_or_blank(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")

def skip_shebang_and_encoding(lines) -> int:
    """
    Return index after shebang and encoding declarations at file start.
    """
    i = 0
    n = len(lines)
    # Shebang
    if i < n and lines[i].startswith("#!"):
        i += 1
    # Encoding (PEP 263)
    # Allow one or two such lines right at the top
    while i < n and re.match(r'^\s*#.*coding[:=]\s*[-\w.]+', lines[i]):
        i += 1
    # Skip any leading blank lines or comments before docstring/imports
    while i < n and is_comment_or_blank(lines[i]):
        i += 1
    return i

def compute_insert_index(text: str) -> int:
    """
    Decide insertion line index (0-based) for a new import: after
    - shebang, encoding, module docstring
    - the complete top import section (incl. multi-line imports)
    If no imports, insert right after docstring (or start).
    """
    lines = text.splitlines(keepends=False)
    n = len(lines)

    # Start after shebang/encoding/comments-at-top
    start_i = skip_shebang_and_encoding(lines)

    # Move start_i after module docstring if present
    doc_end_line = find_docstring_end_lineno("\n".join(lines))  # 1-based
    if doc_end_line > 0:
        # There may be blank/comment lines after docstring; advance past them
        start_i = max(start_i, doc_end_line)
        while start_i < n and is_comment_or_blank(lines[start_i]):
            start_i += 1

    # Now scan import block (allow blank/comments between groups, handle paren continuations and backslashes)
    i = start_i
    last_import_line: Optional[int] = None
    paren_depth = 0
    seen_any_import = False

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if paren_depth > 0:
            paren_depth += line.count("(") - line.count(")")
            last_import_line = i
            i += 1
            continue

        if is_comment_or_blank(line):
            # Still can be within the import header region
            i += 1
            continue

        if is_import_start(line):
            seen_any_import = True
            last_import_line = i
            # crude but effective handling of multiline imports
            paren_depth = line.count("(") - line.count(")")
            if stripped.endswith("\\") and paren_depth == 0:
                # treat backslash continuation as an open group
                paren_depth = 1
            i += 1
            continue

        # first non-import, non-blank/comment line after header
        break

    if seen_any_import and last_import_line is not None:
        return last_import_line + 1
    else:
        # No imports — insert right after docstring/headers
        return start_i

def apply_fix(path: Path, import_line: str, pattern: str, backup_ext: str) -> Tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, "skip (encoding error)"
    except Exception as e:
        return False, f"skip (read error: {e})"

    if not has_target_constructor(text, pattern):
        return False, "skip (no constructor)"

    if has_import_line(text, import_line):
        return False, "ok (import already present)"

    insert_at = compute_insert_index(text)
    lines = text.splitlines(keepends=True)

    # Ensure import line ends with newline
    new_import = import_line
    if not new_import.endswith("\n"):
        new_import += "\n"

    # If inserting into a non-empty import block, keep one blank line after
    inserting_after_imports = insert_at > 0 and is_import_start(lines[insert_at-1] if insert_at-1 < len(lines) else "")
    payload = new_import
    if inserting_after_imports and insert_at < len(lines) and not is_comment_or_blank(lines[insert_at]):
        payload += "\n"

    new_text = "".join(lines[:insert_at]) + payload + "".join(lines[insert_at:])

    try:
        if backup_ext:
            bak = path.with_suffix(path.suffix + backup_ext)
            bak.write_text(text, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")
        return True, "fixed (import inserted)"
    except Exception as e:
        return False, f"error (write failed: {e})"

def main():
    ap = argparse.ArgumentParser(description="Insert AppContext import where __init__(self, app: AppContext) is used.")
    ap.add_argument("root", nargs="?", default="stephanie", help="Root directory to scan (default: stephanie)")
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    ap.add_argument("--import-line", default=DEFAULT_IMPORT_LINE, help="Import line to insert")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="Regex for constructor signature")
    ap.add_argument("--backup-ext", default="", help='Backup extension (e.g., ".bak")')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    changed = 0
    checked = 0

    for py in root.rglob("*.py"):
        # Skip common virtual env or cache dirs inside root if any
        parts = set(py.parts)
        if any(p in parts for p in (".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache")):
            continue

        checked += 1
        # Dry-run preview: we still compute but don't write unless --apply
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue

        if not has_target_constructor(text, args.pattern):
            continue

        if has_import_line(text, args.import_line):
            print(f"[OK]    {py}")
            continue

        if args.apply:
            ok, msg = apply_fix(py, args.import_line, args.pattern, args.backup_ext)
            status = "FIXED" if ok else "ERR"
            if ok: changed += 1
            print(f"[{status}] {py} - {msg}")
        else:
            insert_at = compute_insert_index(text)
            print(f"[ADD]   {py} -> insert after line {insert_at}: {args.import_line}")

    if args.apply:
        print(f"\nDone. Changed files: {changed}")
    else:
        print("\nDry-run. Use --apply to write changes.")

if __name__ == "__main__":
    main()
