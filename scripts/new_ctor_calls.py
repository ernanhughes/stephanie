#!/usr/bin/env python3
import sys
import re
import pathlib

# Heuristic: replace ClassName(cfg, memory, container, logger) -> ClassName(app)
CALL_RE = re.compile(
    r"(\b[A-Za-z_][A-Za-z0-9_]*\s*\()\s*cfg\s*,\s*memory\s*,\s*container\s*,\s*logger\s*(\))"
)

def rewrite(path: pathlib.Path):
    s = path.read_text(encoding="utf-8")
    n = CALL_RE.sub(r"\1app\2", s)
    if n != s:
        path.write_text(n, encoding="utf-8")
        return True
    return False

def main(root):
    changed = 0
    for py in pathlib.Path(root).rglob("*.py"):
        if rewrite(py):
            changed += 1
            print("updated", py)
    print("changed files:", changed)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit("usage: new_ctor_calls.py <repo-root>")
    main(sys.argv[1])
