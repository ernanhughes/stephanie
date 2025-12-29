#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"Command failed:\n  {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bundle changed git files into a single markdown file for sharing."
    )
    ap.add_argument("--base", default="origin/main", help="Base ref to diff against (default: origin/main)")
    ap.add_argument("--out", default="changed_files_bundle.md", help="Output markdown path")
    ap.add_argument("--repo", default=".", help="Repo root (default: current dir)")
    ap.add_argument("--include-diff", action="store_true", help="Include unified diffs (in addition to full file contents)")
    ap.add_argument("--diff-context", type=int, default=3, help="Lines of diff context (default: 3)")
    ap.add_argument("--max-bytes", type=int, default=500_000, help="Skip files larger than this (default: 500k)")
    ap.add_argument("--extensions", default="", help="Comma-separated whitelist (e.g. .py,.md,.yaml). Empty=all")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_path = Path(args.out).resolve()

    # Ensure base exists locally
    # (won't fetch automatically; run `git fetch` if needed)
    _ = run(["git", "rev-parse", "--verify", args.base], repo).strip()

    # Get changed file list (tracked)
    # Includes modified, added, renamed. Excludes deleted by default (we’ll note them).
    status = run(["git", "diff", "--name-status", f"{args.base}...HEAD"], repo).splitlines()

    allowed_ext = None
    if args.extensions.strip():
        allowed_ext = {e.strip() for e in args.extensions.split(",") if e.strip()}

    changed: list[tuple[str, str]] = []  # (status, path)
    for line in status:
        if not line.strip():
            continue
        parts = line.split("\t")
        st = parts[0]
        if st.startswith("R"):  # rename: "R100 old new"
            # keep new name
            path = parts[-1]
        else:
            path = parts[1] if len(parts) > 1 else parts[0].split(maxsplit=1)[-1]
            # some git outputs: "M\tpath"
            if "\t" not in line and len(parts) == 1:
                # fallback
                st, path = line.split(maxsplit=1)

        changed.append((st, path))

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = []
    lines.append(f"# Git Change Bundle\n")
    lines.append(f"- Base: `{args.base}`\n- Head: `HEAD`\n- Generated: `{now}`\n")
    lines.append("## Files\n")
    for st, path in changed:
        lines.append(f"- `{st}` `{path}`")
    lines.append("")

    # Optionally add diffs
    if args.include_diff:
        diff_text = run(
            ["git", "diff", f"{args.base}...HEAD", f"-U{args.diff_context}"],
            repo
        )
        lines.append("## Unified Diff\n")
        lines.append("```diff")
        lines.append(diff_text.rstrip("\n"))
        lines.append("```\n")

    # Add full contents
    lines.append("## File Contents\n")
    for st, rel in changed:
        # Skip deleted files
        if st.startswith("D"):
            lines.append(f"### {rel}\n\n*(deleted)*\n")
            continue

        p = (repo / rel)
        if not p.exists():
            # might be rename edge or generated file not present
            lines.append(f"### {rel}\n\n*(missing on disk)*\n")
            continue

        if allowed_ext is not None and p.suffix not in allowed_ext:
            lines.append(f"### {rel}\n\n*(skipped: extension not in whitelist)*\n")
            continue

        try:
            size = p.stat().st_size
            if size > args.max_bytes:
                lines.append(f"### {rel}\n\n*(skipped: {size} bytes > {args.max_bytes})*\n")
                continue

            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            lines.append(f"### {rel}\n\n*(error reading file: {e})*\n")
            continue

        lang = p.suffix.lstrip(".")
        # reasonable mapping for common stuff
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
