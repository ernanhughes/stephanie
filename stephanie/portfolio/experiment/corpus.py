# stephanie/portfolio/experiment/corpus.py
"""Corpus v1 loader: manifests -> frozen benchmark cases (paired + traps).

Case metadata contract:
    pair_id, defect_present (bool), defect_id (str | None),
    acceptable_labels (list[str]), known_fp (list[str]),
    twin_of (case_id | None), mutation_class, manifest_hash
"""
from __future__ import annotations

import json
from pathlib import Path

from stephanie.portfolio.experiment.case import ExpectedFinding, PortfolioBenchmarkCase

REVIEW_PROMPT = """Review the following book chapter excerpt for ARGUMENT-STRUCTURE defects only.
Report each finding as one JSON object with keys: category, claim, location.
Allowed categories: DUPLICATE_ID, UNKNOWN_CONCEPT, CONCEPT_BEFORE_DEFINITION,
ORPHAN_CONCEPT, UNUSED_EVIDENCE, ORPHAN_SECTION, WEAK_TRANSITION, OTHER.
Output ONLY the JSON array, no preamble and no explanation.

Excerpt:
""".strip()


def load_corpus(corpus_dir: str | Path) -> tuple[list[PortfolioBenchmarkCase], dict]:
    corpus_dir = Path(corpus_dir)
    manifests = json.loads((corpus_dir / "manifests.json").read_text(encoding="utf-8"))
    meta = manifests["corpus"]
    cases: list[PortfolioBenchmarkCase] = []
    for manifest in manifests["cases"]:
        source_text = manifest["variant_text"]
        prompt = REVIEW_PROMPT + "\n" + source_text
        expected: list[ExpectedFinding] = []
        gt = manifest["ground_truth"]
        if gt.get("required_evidence"):
            expected.append(ExpectedFinding(
                code=manifest["injection"]["category"],
                keywords=tuple(gt["required_evidence"]),
                severity="major",
                note=gt["canonical_description"][:300],
            ))
        case = PortfolioBenchmarkCase.freeze(
            case_id=manifest["case_id"],
            task_type="book.argument.review",
            prompt=prompt,
            source_text=source_text,
            expected=tuple(expected),
            metadata={
                "pair_id": manifest["pair_id"],
                "defect_present": manifest["ground_truth"]["defect_present"],
                "defect_id": manifest["injection"]["defect_id"],
                "acceptable_labels": manifest["ground_truth"]["acceptable_labels"],
                "known_fp": manifest["ground_truth"].get("known_fp", []),
                "twin_of": manifest.get("twin_of"),
                "mutation_class": manifest.get("mutation_class", "historical_reintroduction"),
                "manifest_hash": manifest["manifest_hash"],
                "corpus_version": meta["version"],
            },
        )
        cases.append(case)
    return cases, meta
