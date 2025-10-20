#!/usr/bin/env python3
# analyze_json_heavy.py
"""
Stream-inspect a huge JSON file and report what's taking up space.

- Memory-safe via ijson (streaming parser)
- Prints top-level key sizes, heaviest paths, largest strings, and suspects
- Works for JSON objects or arrays at the root

Usage:
  pip install ijson
  python analyze_json_heavy.py /path/to/file.json --top 25
"""
from __future__ import annotations
import argparse
import os
import sys
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, List

# ---- Optional dependency check ----
try:
    import ijson  # type: ignore
except Exception:
    print("This script requires 'ijson' (streaming JSON parser). Install with: pip install ijson")
    sys.exit(1)

# ---------- helpers ----------
SUSPECT_KEYS = {
    "trace","REPORTS","STAGE_DETAILS",
    "documents","raw_documents","pages","images",
    "embeddings","vectors","tensors",
    "html","raw_html","pdf_bytes","screenshots",
    "llm_messages","conversation","prompt_history",
    "few_shot","icl_examples","cartridges","chunks","tables",
    "mars_corpus","score_bundles","score_corpus",
}

def fmt_bytes(n: float) -> str:
    if n < 1024: return f"{int(n)} B"
    for unit in ["KB","MB","GB","TB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def topn(d: Dict[str, float], n: int) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]

def normalize_prefix(prefix: str) -> str:
    # collapse ".item" segments to "[]"
    return re.sub(r'(\.item)+', lambda m: "[" + "][".join([""] * m.group(0).count(".item")) + "]", prefix).replace("[]", "[*]")

def top_level_of(prefix: str) -> str:
    if not prefix:
        return "<root>"
    return prefix.split('.', 1)[0]

def scalar_estimated_bytes(event: str, value) -> int:
    if event == "string":
        return len(value.encode("utf-8", errors="ignore"))
    if event == "number":
        # rough JSON number size; fine for comparing
        return 16
    if event in ("boolean", "null"):
        return 4
    # map_key is just the key name; count it lightly
    if event == "map_key":
        return len(str(value))
    return 0

# ---------- core ----------
def analyze_json(path: str, top: int = 20) -> dict:
    file_size = os.path.getsize(path)
    size_by_top: Dict[str, int] = defaultdict(int)
    size_by_prefix: Dict[str, int] = defaultdict(int)
    large_strings: List[Tuple[int, str, str]] = []  # (bytes, prefix, preview)
    suspect_hits: Counter[str] = Counter()
    array_item_counts: Counter[str] = Counter()  # approximate: count items seen
    last_seen_item_prefix: Dict[str, int] = {}   # to avoid double-count nested counts

    # Parse
    with open(path, "rb", buffering=1024*1024) as fp:
        for prefix, event, value in ijson.parse(fp):
            # track scalars
            if event in ("string","number","boolean","null","map_key"):
                b = scalar_estimated_bytes(event, value)
                size_by_prefix[prefix] += b
                size_by_top[top_level_of(prefix)] += b

                # large strings
                if event == "string" and b >= 200_000:  # ~200 KB
                    preview = value[:200].replace("\n"," ") + ("..." if len(value) > 200 else "")
                    large_strings.append((b, prefix, preview))

            # approximate array item counts
            # if we see a scalar at X.item or the start of a map at X.item, count it as an item of array X
            if prefix.endswith(".item") and event in ("string","number","boolean","null","start_map","start_array"):
                base = prefix[:-5]  # drop ".item"
                # prevent repeated counting when nested events arrive for same item
                last = last_seen_item_prefix.get(base, -1)
                # naive running counter
                last_seen_item_prefix[base] = last + 1
                array_item_counts[base] = last_seen_item_prefix[base] + 1

            # suspect key hit
            tl = top_level_of(prefix)
            if tl in SUSPECT_KEYS:
                suspect_hits[tl] += 1

    # Summaries
    large_strings.sort(key=lambda t: t[0], reverse=True)
    heavy_paths = topn({normalize_prefix(k): v for k, v in size_by_prefix.items()}, top * 2)
    heavy_top = topn(size_by_top, top)

    heavy_arrays = topn({normalize_prefix(k): v for k, v in array_item_counts.items()}, top)

    return {
        "file_size": file_size,
        "top_level_sizes": heavy_top,
        "heaviest_paths": heavy_paths[:top],
        "largest_strings": [(fmt_bytes(b), normalize_prefix(p), prev) for b,p,prev in large_strings[:top]],
        "heavy_arrays_guess": [(p, int(c)) for p,c in heavy_arrays],
        "suspect_hits": suspect_hits.most_common(),
    }

def print_report(report: dict, top: int):
    print("\n=== JSON Size Report ===")
    print(f"File size: {fmt_bytes(report['file_size'])}")

    print("\n-- Top-level keys by estimated size --")
    if not report["top_level_sizes"]:
        print("  (no data)")
    for k, sz in report["top_level_sizes"]:
        print(f"  {k:30s} {fmt_bytes(sz)}")

    print("\n-- Heaviest paths (prefixes) --")
    for p, sz in report["heaviest_paths"]:
        print(f"  {fmt_bytes(sz):>10}  {p}")

    print("\n-- Largest string values --")
    if not report["largest_strings"]:
        print("  (no large strings found)")
    for b, p, prev in report["largest_strings"]:
        print(f"  {b:>10}  {p}")
        print(f"      preview: {prev}")

    print("\n-- Arrays (approx item counts) --")
    if not report["heavy_arrays_guess"]:
        print("  (no arrays detected or count unknown)")
    for p, c in report["heavy_arrays_guess"]:
        print(f"  {c:>8}  items  in  {p}")

    print("\n-- Suspect keys (hit counts) --")
    if not report["suspect_hits"]:
        print("  (none)")
    else:
        for k, n in report["suspect_hits"]:
            print(f"  {k:22s} {n}")

    # quick suggestions
    suspects_present = [k for k,_ in report["suspect_hits"]]
    if suspects_present:
        print("\nSuggestions:")
        print("  • Consider pruning these from context snapshots or storing by ID:")
        print("    ", ", ".join(sorted(suspects_present)))
        print("  • Very heavy prefixes above likely correspond to nested traces or ICL examples.")
        print("  • Store blobs (HTML/PDF/embeddings/examples) out-of-band and log only identifiers.")

def main():
    print("Starting analysis...")
    
    ap = argparse.ArgumentParser(description="Analyze large JSON for heavy keys/paths/strings (streaming).")
    ap.add_argument("json_path", help="Path to the JSON file")
    ap.add_argument("--top", type=int, default=20, help="How many items to show per section")
    args = ap.parse_args()

    if not os.path.exists(args.json_path):
        print(f"File not found: {args.json_path}")
        sys.exit(1)

    print("Starting analysis...")
    report = analyze_json(args.json_path, top=args.top)
    print_report(report, args.top)

if __name__ == "__main__":
    main()
