#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"Command failed:\n  {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bundle ONLY uncommitted changes (staged + unstaged) into a single markdown file."
    )
    ap.add_argument("--out", default="uncommitted_bundle.md", help="Output markdown path")
    ap.add_argument("--repo", default=".", help="Repo root (default: current dir)")
    ap.add_argument("--include-diff", action="store_true", help="Include unified diff of uncommitted changes")
    ap.add_argument("--diff-context", type=int, default=3, help="Lines of diff context (default: 3)")
    ap.add_argument("--max-bytes", type=int, default=500_000, help="Skip files larger than this (default: 500k)")
    ap.add_argument("--extensions", default=".py,.yaml", help="Comma-separated whitelist (e.g. .py,.md,.yaml). Empty=all")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_path = Path(args.out).resolve()

    allowed_ext = None
    if args.extensions.strip():
        allowed_ext = {e.strip() for e in args.extensions.split(",") if e.strip()}

    # name-status for uncommitted changes (includes staged + unstaged)
    status = run(["git", "status", "--porcelain=v1"], repo).splitlines()

    changed: list[tuple[str, str]] = []  # (code, path)
    for line in status:
        if not line.strip():
            continue
        # Format: XY <path>  (or XY <old> -> <new> for renames)
        code = line[:2]
        rest = line[3:]

        # Handle rename "R  old -> new"
        if " -> " in rest:
            path = rest.split(" -> ", 1)[1].strip()
        else:
            path = rest.strip()

        changed.append((code, path))

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("# Uncommitted Git Change Bundle\n")
    lines.append(f"- Generated: `{now}`\n")
    lines.append("## Files (staged + unstaged)\n")
    if not changed:
        lines.append("*No uncommitted changes found.*\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote: {out_path}")
        return

    for code, path in changed:
        lines.append(f"- `{code}` `{path}`")
    lines.append("")

    if args.include_diff:
        diff_text = run(["git", "diff", f"-U{args.diff_context}"], repo)
        cached_text = run(["git", "diff", "--cached", f"-U{args.diff_context}"], repo)

        lines.append("## Unified Diff (unstaged)\n")
        lines.append("```diff")
        lines.append(diff_text.rstrip("\n"))
        lines.append("```\n")

        lines.append("## Unified Diff (staged)\n")
        lines.append("```diff")
        lines.append(cached_text.rstrip("\n"))
        lines.append("```\n")

    lines.append("## File Contents (current working tree)\n")

    for code, rel in changed:
        p = (repo / rel)
        if code.strip().startswith("D") or (code[1] == "D"):
            lines.append(f"### {rel}\n\n*(deleted)*\n")
            continue

        if not p.exists():
            lines.append(f"### {rel}\n\n*(missing on disk)*\n")
            continue

        if allowed_ext is not None and p.suffix not in allowed_ext:
            lines.append(f"### {rel}\n\n*(skipped: extension not in whitelist)*\n")
            continue

        size = p.stat().st_size
        if size > args.max_bytes:
            lines.append(f"### {rel}\n\n*(skipped: {size} bytes > {args.max_bytes})*\n")
            continue

        content = p.read_text(encoding="utf-8", errors="replace")
        lang = p.suffix.lstrip(".")
        if lang == "yml":
            lang = "yaml"
        if lang == "txt":
            lang = ""

        lines.append(f"### {rel}\n")
        lines.append("```" + lang)
        lines.append(content.rstrip("\n"))
        lines.append("```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
