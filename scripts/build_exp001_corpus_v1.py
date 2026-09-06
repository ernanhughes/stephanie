"""Build Experiment 001 Corpus v1: 12 paired injections + 2 FP traps.

Takes current new-books chapters as clean sources, injects exactly one
historical defect per variant, writes manifests with triple hashes, and
runs the injection integrity preflight (predicate false on clean, true
on variant) before writing anything.

Usage:
    python scripts/build_exp001_corpus_v1.py --out outputs/portfolio_exp001_corpus_v1
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEW_BOOKS = Path(r"C:\Projects\new-books\content\books\dspy-from-first-principles")
OUT_DEFAULT = "outputs/portfolio_exp001_corpus_v1"
MAX_CHARS = 3500


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def excerpt_of(chapter: str) -> str:
    return (NEW_BOOKS / f"{chapter}-chapter.md").read_text(encoding="utf-8")[:MAX_CHARS]


def insert_after_paragraph(text: str, block: str, paragraph_index: int = 2) -> tuple[str, str]:
    """Insert block after the Nth blank-line paragraph. Returns (variant, anchor_context)."""
    parts = re.split(r"(\n\s*\n)", text)
    # parts alternate [para, sep, para, sep, ...]; insert after paragraph_index-th para.
    para_positions = [i for i in range(0, len(parts), 2)]
    idx = para_positions[min(paragraph_index, len(para_positions) - 1)]
    anchor = parts[idx][-200:]
    variant = "".join(parts[: idx + 1]) + "\n\n" + block + "\n" + "".join(parts[idx + 1:])
    return variant, anchor


def duplicate_first_h2(text: str) -> tuple[str, str, str]:
    """Duplicate the first ## header title (DUPLICATE_ID proxy)."""
    match = re.search(r"(?m)^(##\s+.+)$", text)
    if not match:
        raise ValueError("no ## header found for DUPLICATE_ID injection")
    title = match.group(1)
    marker = f"\n\n{title}\n\nThe following restates the section above for emphasis.\n"
    pos = len(text) // 2
    return text[:pos] + marker + text[pos:], title, title


# ---------------------------------------------------------------- specs

def spec(defect_id, chapter, category, description, required_evidence,
         acceptable_labels, block=None, operation="insert_after_paragraph",
         paragraph_index=2, validator_markers=(), validator_absent_ok=True,
         canonical_extra=""):
    return {
        "defect_id": defect_id, "chapter": chapter, "category": category,
        "description": description, "required_evidence": required_evidence,
        "acceptable_labels": acceptable_labels, "block": block,
        "operation": operation, "paragraph_index": paragraph_index,
        "validator_markers": validator_markers,
        "canonical_extra": canonical_extra,
    }


SPECS = [
    spec("D07-DUP", "07", "DUPLICATE_ID",
         "Two sections share one header title, so the section id collides.",
         ["duplicate", "header", "section", "collide"],
         ["DUPLICATE_ID"], operation="duplicate_h2"),
    spec("D07-ORPHAN", "07", "ORPHAN_CONCEPT",
         "CONCEPT DECLARATION example-roles is never used by any argument.",
         ["example-roles", "unused", "argument"],
         ["ORPHAN_CONCEPT", "MISSING_PREREQUISITE"],
         block=("> CONCEPT DECLARATION: `example-roles` — roles that worked examples "
                "may play in a DSPy program (demonstration, training case, evaluation case)."),
         validator_markers=("example-roles",)),
    spec("D07-UNUSED", "07", "UNUSED_EVIDENCE",
         "EVIDENCE TABLE four-words-table is present but linked by no argument.",
         ["four-words-table", "unlinked", "evidence"],
         ["UNUSED_EVIDENCE"],
         block=("> EVIDENCE TABLE: `four-words-table` — four canonical phrasings of the "
                "optimizer claim with per-phrasing scores."),
         validator_markers=("four-words-table",)),
    spec("D08-UNKNOWN", "08", "UNKNOWN_CONCEPT",
         "measurement-process is relied upon as established but introduced nowhere.",
         ["measurement-process", "introduced", "nowhere"],
         ["UNKNOWN_CONCEPT", "CONCEPT_BEFORE_DEFINITION"],
         block=("The measurement-process therefore determines the ceiling before any "
                "optimizer runs, so we fix the process first and tune second."),
         validator_markers=("measurement-process",)),
    spec("D08-CBD", "08", "CONCEPT_BEFORE_DEFINITION",
         "editorial-metric-v1 and per-case-evidence are used before any definition.",
         ["editorial-metric-v1", "per-case-evidence", "definition"],
         ["CONCEPT_BEFORE_DEFINITION", "UNKNOWN_CONCEPT"],
         block=("Scored against editorial-metric-v1, every section already reports "
                "per-case-evidence, so the remaining work is purely editorial."),
         validator_markers=("editorial-metric-v1", "per-case-evidence")),
    spec("D08-UNUSED", "08", "UNUSED_EVIDENCE",
         "sanity-baselines and v2-tripwire-pattern evidence is linked by no argument.",
         ["sanity-baselines", "v2-tripwire-pattern", "unlinked"],
         ["UNUSED_EVIDENCE"],
         block=("> EVIDENCE: `sanity-baselines` and the `v2-tripwire-pattern` run log, "
                "recorded for completeness."),
         validator_markers=("sanity-baselines", "v2-tripwire-pattern")),
    spec("D09-CBD", "09", "CONCEPT_BEFORE_DEFINITION",
         "judge-validation is invoked without prior definition.",
         ["judge-validation", "definition", "without"],
         ["CONCEPT_BEFORE_DEFINITION", "UNKNOWN_CONCEPT"],
         block=("Because judge-validation already passed on the draft, we proceed "
                "directly to the rewrite."),
         validator_markers=("judge-validation",)),
    spec("D09-UNUSED", "09", "UNUSED_EVIDENCE",
         "attack-suite-implementation and judge-contract evidence is unlinked.",
         ["attack-suite-implementation", "judge-contract", "unlinked"],
         ["UNUSED_EVIDENCE"],
         block=("> EVIDENCE: the `attack-suite-implementation` listing and the "
                "`judge-contract` excerpt, attached for reference."),
         validator_markers=("attack-suite-implementation", "judge-contract")),
    spec("D10-CBD", "10", "CONCEPT_BEFORE_DEFINITION",
         "baseline-after-invariant and evidence-consumption used without definitions.",
         ["baseline-after-invariant", "evidence-consumption", "definition"],
         ["CONCEPT_BEFORE_DEFINITION", "UNKNOWN_CONCEPT"],
         block=("Once baseline-after-invariant holds, evidence-consumption becomes the "
                "binding constraint on corpus size."),
         validator_markers=("baseline-after-invariant", "evidence-consumption")),
    spec("D10-UNUSED", "10", "UNUSED_EVIDENCE",
         "allowed-change-table evidence is linked by no argument.",
         ["allowed-change-table", "unlinked", "evidence"],
         ["UNUSED_EVIDENCE"],
         block=("> EVIDENCE TABLE: `allowed-change-table` — permitted edit operations "
                "per review stage."),
         validator_markers=("allowed-change-table",)),
    spec("D10-ORPHAN", "10", "ORPHAN_CONCEPT",
         "CONCEPT DECLARATION proposal-vs-promotion never used by any argument.",
         ["proposal-vs-promotion", "unused", "argument"],
         ["ORPHAN_CONCEPT", "MISSING_PREREQUISITE"],
         block=("> CONCEPT DECLARATION: `proposal-vs-promotion` — the distinction between "
                "proposing a candidate change and promoting it to the manuscript."),
         validator_markers=("proposal-vs-promotion",)),
    spec("D11-UNUSED", "11", "UNUSED_EVIDENCE",
         "single-run-table evidence is linked by no argument.",
         ["single-run-table", "unlinked", "evidence"],
         ["UNUSED_EVIDENCE"],
         block=("> EVIDENCE TABLE: `single-run-table` — single-run scores for the "
                " worked chapters."),
         validator_markers=("single-run-table",)),
]

TRAPS = [
    {
        "defect_id": "T-ORPHAN-SEC", "chapter": "09", "category": "ORPHAN_SECTION",
        "description": "Specificity trap: section contributes via synonyms only.",
        "block": ("## How scoring guides revision\n\nThe metric guiding each pass, "
                  "the judge reviewing borderline calls, the optimizer choosing among "
                  "candidates, and the candidate pool itself all shape the final text. "
                  "Each contributes through its role rather than through a declared id."),
        "known_fp": ["ORPHAN_SECTION"],
    },
    {
        "defect_id": "T-WEAK-TRANS", "chapter": "09", "category": "WEAK_TRANSITION",
        "description": "Specificity trap: legitimate operationalizing progression 9->10.",
        "block": ("TRANSITION BRIDGE: the abstract open question above becomes a concrete "
                  "compile-boundary question below — SAME_QUESTION, OPERATIONALIZES. The "
                  "next chapter builds the scoring function this chapter motivated."),
        "known_fp": ["WEAK_TRANSITION"],
    },
]


# ---------------------------------------------------------------- build

def apply_injection(clear: str, s: dict) -> tuple[str, str, str]:
    """Returns (variant_text, before_context, after_text)."""
    if s["operation"] == "duplicate_h2":
        variant, before, after = duplicate_first_h2(clear)
        return variant, before, after
    variant, anchor = insert_after_paragraph(clear, s["block"], s["paragraph_index"])
    return variant, anchor, s["block"]


def validate_markers(clear: str, variant: str, markers: tuple) -> tuple[bool, bool]:
    clean_ok = not any(m in clear for m in markers)
    variant_ok = all(m in variant for m in markers)
    return clean_ok, variant_ok


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    pair_index = 0
    for s in SPECS:
        pair_index += 1
        pair_id = f"exp001-p{pair_index:02d}"
        clean = excerpt_of(s["chapter"])
        variant, before, after = apply_injection(clean, s)
        clean_ok, variant_ok = validate_markers(clean, variant, s["validator_markers"] or ())
        # DUPLICATE_ID structural predicate: title count 1 -> 2.
        if s["operation"] == "duplicate_h2":
            title = after if isinstance(after, str) and after.startswith("##") else before
            clean_ok = clean.count(title) == 1
            variant_ok = variant.count(title) == 2
        status = "ok" if (clean_ok and variant_ok) else "PREFLIGHT_FAIL"
        print(f"{s['defect_id']} clean_absent={clean_ok} variant_present={variant_ok} -> {status}")
        if status != "ok":
            print("PREFLIGHT FAILED — aborting, nothing written")
            return 1

        injection_spec = {
            "defect_id": s["defect_id"], "category": s["category"],
            "operation": s["operation"], "paragraph_index": s.get("paragraph_index"),
            "before_context": before[-300:], "after_text": after[:2000],
        }
        spec_hash = sha(json.dumps(injection_spec, sort_keys=True))
        clean_hash, variant_hash = sha(clean), sha(variant)

        base_manifest = {
            "pair_id": pair_id,
            "mutation_class": "historical_reintroduction",
            "source": {"chapter": s["chapter"], "clean_hash": clean_hash,
                       "variant_hash": variant_hash, "excerpt_chars": len(clean)},
            "injection": injection_spec,
            "injection_spec_hash": spec_hash,
        }
        clean_manifest = {
            **base_manifest,
            "case_id": f"{pair_id}-clean",
            "variant_text": clean,
            "twin_of": f"{pair_id}-defect",
            "manifest_hash": sha(clean_hash + spec_hash + "clean"),
            "ground_truth": {
                "defect_present": False,
                # Defect signature retained so clean-twin false alarms are
                # adjudicable (matched absent defect -> FALSE_POSITIVE).
                "defect_id": s["defect_id"],
                "canonical_description": f"No {s['category']} defect: {s['description']}",
                "required_evidence": list(s["required_evidence"]),
                "acceptable_labels": list(s["acceptable_labels"]),
                "known_fp": [],
            },
        }
        defect_manifest = {
            **base_manifest,
            "case_id": f"{pair_id}-defect",
            "variant_text": variant,
            "twin_of": f"{pair_id}-clean",
            "manifest_hash": sha(clean_hash + spec_hash + "defect"),
            "ground_truth": {
                "defect_present": True,
                "canonical_description": s["description"],
                "required_evidence": list(s["required_evidence"]),
                "acceptable_labels": list(s["acceptable_labels"]),
                "known_fp": [],
            },
        }
        cases.extend([clean_manifest, defect_manifest])

    for trap in TRAPS:
        clean = excerpt_of(trap["chapter"])
        variant, before, after = apply_injection(
            clean, {"operation": "insert", "block": trap["block"], "paragraph_index": 2})
        clean_ok, variant_ok = validate_markers(clean, variant, (trap["block"][:40],))
        status = "ok" if (clean_ok and variant_ok) else "PREFLIGHT_FAIL"
        print(f"{trap['defect_id']} clean_absent={clean_ok} variant_present={variant_ok} -> {status}")
        if status != "ok":
            print("PREFLIGHT FAILED — aborting, nothing written")
            return 1
        injection_spec = {
            "defect_id": trap["defect_id"], "category": trap["category"],
            "operation": "insert_after_paragraph",
            "before_context": before[-300:], "after_text": after[:2000],
        }
        spec_hash = sha(json.dumps(injection_spec, sort_keys=True))
        clean_hash, variant_hash = sha(clean), sha(variant)
        cases.append({
            "case_id": trap["defect_id"], "pair_id": trap["defect_id"],
            "mutation_class": "specificity_trap",
            "source": {"chapter": trap["chapter"], "clean_hash": clean_hash,
                       "variant_hash": variant_hash, "excerpt_chars": len(clean)},
            "injection": injection_spec,
            "injection_spec_hash": spec_hash,
            "variant_text": variant,
            "twin_of": None,
            "manifest_hash": sha(clean_hash + spec_hash + "trap"),
            "ground_truth": {
                "defect_present": False,
                "canonical_description": trap["description"],
                "required_evidence": [],
                "acceptable_labels": [],
                "known_fp": list(trap["known_fp"]),
            },
        })

    payload = {
        "corpus": {
            "version": "portfolio-exp001-corpus/v1",
            "cases": len(cases),
            "pairs": len(SPECS),
            "traps": len(TRAPS),
            "preflight": "pass",
        },
        "cases": cases,
    }
    (out_dir / "manifests.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} case manifests -> {out_dir / 'manifests.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
