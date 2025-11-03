#!/usr/bin/env python3
"""
dir_tree.py — portable repo tree dumper (text or JSON)

Features
- Cross-platform (Windows/macOS/Linux), no third-party deps
- Exclude patterns (glob): ".git,node_modules,__pycache__,*.egg-info,.venv,venv,dist,build"
- Depth limit (default 4)
- Show aggregate sizes and file/dir counts
- Output formats: text (tree) or JSON
- Optionally only list directories
- Optional anonymization by hashing names for public sharing

Examples
  python dir_tree.py --root . --out repo-tree.txt
  python dir_tree.py --root . --format json --out repo-tree.json
  python dir_tree.py --root . --max-depth 5 --only-dirs --show-sizes
  python dir_tree.py --root . --exclude ".git,node_modules,dist,build,__pycache__,.venv,*.egg-info" --out repo-tree.txt
  python dir_tree.py --root . --anonymize --format json --out repo-anon.json
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_EXCLUDES = [
    ".git", ".hg", ".svn",
    ".venv", "venv", ".tox",
    "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
    ".ipynb_checkpoints", ".idea", ".DS_Store",
]

def human_size(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"

def sha8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

@dataclass
class Node:
    name: str
    path: str
    type: str  # "dir" | "file"
    size: int = 0
    files: int = 0
    dirs: int = 0
    children: Optional[List["Node"]] = None
    pruned: int = 0  # children not shown due to depth limit

    def to_dict(self, anonymize: bool = False) -> Dict[str, Any]:
        nm = self.name
        pth = self.path
        if anonymize:
            nm = f"{'D' if self.type=='dir' else 'F'}-{sha8(self.name)}"
            pth = "/".join("H-"+sha8(seg) for seg in self.path.replace("\\","/").split("/"))
        d = {
            "name": nm,
            "path": pth,
            "type": self.type,
            "size": self.size,
            "files": self.files,
            "dirs": self.dirs,
            "pruned": self.pruned,
        }
        if self.children is not None:
            d["children"] = [c.to_dict(anonymize=anonymize) for c in self.children]
        return d

def compile_patterns(patterns_csv: str) -> List[str]:
    pats: List[str] = []
    for raw in patterns_csv.split(","):
        s = raw.strip()
        if s:
            pats.append(s)
    return pats

def is_excluded(rel_path: str, name: str, patterns: List[str], include_hidden: bool) -> bool:
    # Hidden file/dir (starts with .) handling
    if not include_hidden and name.startswith("."):
        return True
    # Glob match by name and by relative path
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
            return True
        # Also check any path segment match (common for folder globs)
        parts = rel_path.replace("\\", "/").split("/")
        if any(fnmatch.fnmatch(seg, pat) for seg in parts):
            return True
    return False

def safe_getsize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def scan(root: str,
         rel_root: str,
         max_depth: int,
         depth: int,
         excludes: List[str],
         include_hidden: bool,
         only_dirs: bool,
         follow_symlinks: bool) -> Node:
    name = os.path.basename(root) or root
    node = Node(name=name, path=rel_root or ".", type="dir", size=0, files=0, dirs=1, children=[])
    try:
        entries = list(os.scandir(root))
    except OSError:
        # unreadable directory
        node.pruned = 0
        return node

    # Sort: directories first, then files, alphabetically
    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=follow_symlinks), e.name.lower()))

    if depth >= max_depth:
        # We won't descend further; just count how many we’re not showing
        node.pruned = len(entries)
        # Still accumulate summary sizes/counts shallowly
        for e in entries:
            rel = os.path.join(rel_root, e.name) if rel_root else e.name
            if is_excluded(rel, e.name, excludes, include_hidden):
                continue
            if e.is_file(follow_symlinks=False):
                node.files += 1
                node.size += safe_getsize(e.path)
            elif e.is_dir(follow_symlinks=follow_symlinks):
                node.dirs += 1
        return node

    for e in entries:
        rel = os.path.join(rel_root, e.name) if rel_root else e.name
        if is_excluded(rel, e.name, excludes, include_hidden):
            continue
        try:
            if e.is_dir(follow_symlinks=follow_symlinks):
                child = scan(e.path, rel, max_depth, depth+1, excludes, include_hidden, only_dirs, follow_symlinks)
                node.size += child.size
                node.files += child.files
                node.dirs += child.dirs
                if node.children is not None:
                    node.children.append(child)
            elif not only_dirs and e.is_file(follow_symlinks=False):
                fsize = safe_getsize(e.path)
                node.size += fsize
                node.files += 1
                if node.children is not None:
                    node.children.append(Node(name=e.name, path=rel, type="file", size=fsize, files=1, dirs=0))
        except OSError:
            # Skip problematic entries
            continue
    return node

def print_tree(node: Node, show_sizes: bool, out, prefix: str = "", is_last: bool = True):
    connector = "└── " if is_last else "├── "
    label = node.name
    if show_sizes:
        if node.type == "dir":
            label += f"  [dirs:{node.dirs-1 if node.dirs>0 else 0} files:{node.files} size:{human_size(node.size)}]"
        else:
            label += f"  [{human_size(node.size)}]"
    if node.pruned:
        label += f"  …(+{node.pruned} more)"
    out.write(prefix + connector + label + "\n")

    if node.children:
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, ch in enumerate(node.children):
            print_tree(ch, show_sizes, out, new_prefix, i == len(node.children)-1)

def main():
    ap = argparse.ArgumentParser(description="Dump a directory tree (portable, no deps).")
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument("--max-depth", type=int, default=4, help="Max depth to display (default: 4)")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDES), help="Comma-separated glob patterns to exclude")
    ap.add_argument("--include-hidden", action="store_true", help="Include dotfiles/directories")
    ap.add_argument("--only-dirs", action="store_true", help="List directories only (no files)")
    ap.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks for directories")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ap.add_argument("--show-sizes", action="store_true", help="Show human-readable sizes in text mode")
    ap.add_argument("--anonymize", action="store_true", help="Hash names/paths for sharing publicly")
    ap.add_argument("--out", default="-", help="Output file path or '-' for stdout")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    rel_root = os.path.basename(root.rstrip(os.sep))
    excludes = compile_patterns(args.exclude) if args.exclude else []

    node = scan(root, rel_root, args.max_depth, 0, excludes, args.include_hidden, args.only_dirs, args.follow_symlinks)

    if args.anonymize:
        # Convert to dict with anonymized names/paths
        data = node.to_dict(anonymize=True)
    else:
        data = node.to_dict(anonymize=False)

    # Output
    if args.out == "-":
        out = sys.stdout
        must_close = False
    else:
        out = open(args.out, "w", encoding="utf-8")
        must_close = True

    try:
        if args.format == "json":
            json.dump(data, out, indent=2)
            out.write("\n")
        else:
            # Print header line with totals
            header = f"{data['name']}  [dirs:{data['dirs']-1 if data['dirs']>0 else 0} files:{data['files']} size:{human_size(data['size'])}]"
            out.write(header + ("\n" if not header.endswith("\n") else ""))
            # Render tree
            # Rebuild Node from dict (for simplicity reuse print logic on the original node)
            print_tree(node if not args.anonymize else Node(**{
                "name": data["name"], "path": data["path"], "type": data["type"], "size": data["size"],
                "files": data["files"], "dirs": data["dirs"], "children": [],
                "pruned": data.get("pruned", 0)
            }), args.show_sizes, out, prefix="", is_last=True)
            # The header is shown; tree root printed as last line too for consistency.
    finally:
        if must_close:
            out.close()

if __name__ == "__main__":
    main()
