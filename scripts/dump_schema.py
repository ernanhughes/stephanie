#!/usr/bin/env python3
"""
dump_schema.py
- Dump a PostgreSQL schema with pg_dump (schema-only) and pretty-format it
- Or pretty-format an existing SQL file

Usage examples
--------------
# 1) Dump directly from DB (password via prompt)
python tools/dump_schema.py --host localhost --user postgres --dbname co --output schema.sql --ask-password

# 2) Dump with PGPASSWORD env (PowerShell)
$env:PGPASSWORD="secret"; python tools/dump_schema.py --host localhost --user postgres --dbname co --output schema.sql

# 3) Format an existing file
python tools/dump_schema.py --input schema_raw.sql --output schema.sql
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

def find_pg_dump(explicit_path: Optional[str]) -> str:
    """Find pg_dump (works on Windows/macOS/Linux)."""
    if explicit_path:
        return explicit_path
    path = shutil.which("pg_dump")
    if path:
        return path
    # Windows: try common install dirs
    for envkey in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(envkey)
        if not base:
            continue
        for p in Path(base).glob("PostgreSQL/*/bin/pg_dump.exe"):
            return str(p)
    raise FileNotFoundError(
        "pg_dump not found. Add it to PATH or pass --pg-dump-path "
        "(e.g. 'C:\\Program Files\\PostgreSQL\\16\\bin\\pg_dump.exe')."
    )

def run_pg_dump(host: str, port: int, user: str, dbname: str,
                password: Optional[str], pg_dump_path: Optional[str]) -> str:
    pg_dump = find_pg_dump(pg_dump_path)
    cmd = [
        pg_dump, "-h", host, "-p", str(port), "-U", user, "-d", dbname,
        "--schema-only", "--no-owner", "--no-privileges"
    ]
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    proc = subprocess.run(cmd, env=env, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pg_dump failed ({proc.returncode}).\nSTDERR:\n{proc.stderr.decode(errors='ignore')}"
        )
    return proc.stdout.decode("utf-8", errors="ignore")

_BOILERPLATE_PATTERNS = [
    r"^\s*-- PostgreSQL database dump",
    r"^\s*-- Dumped from database version",
    r"^\s*-- Dumped by pg_dump version",
    r"^\s*SELECT pg_catalog\.set_config\(",
    r"^\s*SET\s+\w+",
    r"^\s*RESET\s+\w+",
    r"^\s*\\connect\b",
    r"^\s*\\(?:un)?restrict\b.*$",
]

def strip_boilerplate(text: str, aggressive: bool = False) -> str:
    keep: list[str] = []
    patterns = [re.compile(p) for p in _BOILERPLATE_PATTERNS]
    # Optionally drop CREATE/DROP/ALTER DATABASE blocks, which are often noisy in repos
    drop_db = re.compile(r"^\s*(DROP|CREATE|ALTER)\s+DATABASE\b", re.IGNORECASE)
    for line in text.splitlines():
        s = line.strip()
        if any(p.match(s) for p in patterns):
            continue
        if aggressive and drop_db.match(s):
            continue
        keep.append(line)
    cleaned = "\n".join(keep)
    # collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned

def try_sqlparse_format(sql: str, keyword_case: str = "upper") -> str:
    try:
        import sqlparse
    except Exception:
        # sqlparse not installed—return original
        return sql
    return sqlparse.format(sql, reindent=True, keyword_case=keyword_case)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")

def main():
    ap = argparse.ArgumentParser(description="Dump and pretty-format PostgreSQL schema.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", help="Existing SQL file to format instead of running pg_dump.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user")
    ap.add_argument("--dbname")
    ap.add_argument("--ask-password", action="store_true", help="Prompt for DB password.")
    ap.add_argument("--pg-dump-path", help="Path to pg_dump(.exe) if not in PATH.")
    ap.add_argument("--output", default="schema.sql", help="Output file path.")
    ap.add_argument("--no-format", action="store_true", help="Skip sqlparse reformatting.")
    ap.add_argument("--keep-boilerplate", action="store_true", help="Keep pg_dump boilerplate.")
    ap.add_argument("--keep-db-ddl", action="store_true", help="Keep CREATE/DROP/ALTER DATABASE lines.")
    ap.add_argument("--keyword-case", choices=["upper", "lower", "capitalize"], default="upper")

    args = ap.parse_args()
    out_path = Path(args.output)

    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    else:
        # Need minimal DB args to run pg_dump
        missing = [k for k in ("user", "dbname") if not getattr(args, k)]
        if missing:
            ap.error(f"Missing required arguments for pg_dump: {', '.join(missing)}")
        pw = getpass.getpass("Postgres password: ") if args.ask_password else os.environ.get("PGPASSWORD")
        raw = run_pg_dump(args.host, args.port, args.user, args.dbname, pw, args.pg_dump_path)

    cleaned = raw if args.keep_boilerplate else strip_boilerplate(raw, aggressive=not args.keep_db_ddl)
    formatted = cleaned if args.no_format else try_sqlparse_format(cleaned, keyword_case=args.keyword_case)
    write_text(out_path, formatted)
    print(f"Wrote {out_path} ({len(formatted)} bytes).")


# py scripts\dump_schema.py --host localhost --user co --dbname co --output schema_raw.sql --ask-password
if __name__ == "__main__":
    main()
