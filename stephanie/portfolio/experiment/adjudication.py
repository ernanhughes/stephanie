# stephanie/portfolio/experiment/adjudication.py
"""Adjudication hierarchy (§Experiment 001: ground truth).

1. deterministic code/keyword match against frozen expectations;
2. duplicate suppression within (case, arm);
3. everything else -> UNVERIFIABLE (canary: no LLM judge, no circularity).

Provenance retained on every classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from stephanie.portfolio.experiment.case import PortfolioBenchmarkCase
from stephanie.portfolio.experiment.finding import Finding, FindingClass


@dataclass(frozen=True)
class Adjudication:
    finding_id: str
    classification: FindingClass
    matched_code: str | None = None
    method: str = "deterministic_match"
    provenance: Mapping[str, str] = field(default_factory=dict)


# Frozen adjudication contract version. 3.8B eligibility requires this
# exact version (plus corpus version) to match the validated experiment.
ADJUDICATION_VERSION = "v2-content-evidence+labels+twin-fp"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


class DeterministicAdjudicator:
    def adjudicate(
        self, findings: Sequence[Finding], case: PortfolioBenchmarkCase
    ) -> list[tuple[Finding, Adjudication]]:
        from dataclasses import replace

        known_fp = set((case.metadata or {}).get("known_fp", []))
        acceptable = set((case.metadata or {}).get("acceptable_labels", []))
        defect_present = (case.metadata or {}).get("defect_present", True)
        results: list[tuple[Finding, Adjudication]] = []
        seen_codes: set[str] = set()
        for finding in findings:
            haystack = _normalize(" ".join([
                finding.category or "",
                finding.claim or "",
                finding.location or "",
                " ".join(finding.evidence),
            ]))
            matched = None
            haystack_flat = _compact(haystack)
            for expected in case.expected:
                code_hit = _compact(expected.code) in haystack_flat
                keyword_hits = sum(1 for kw in expected.keywords if _normalize(kw) in haystack)
                label_ok = (not acceptable) or (finding.category in acceptable)
                # A bare category label is not a detection (models spray allowed
                # categories as generic buckets). Require content evidence:
                # code + defect keyword (+ acceptable label), or two keywords
                # alone (semantic identity over exact taxonomy).
                if (code_hit and label_ok and (not expected.keywords or keyword_hits >= 1)) or (
                    expected.keywords and keyword_hits >= min(2, len(expected.keywords))
                ):
                    matched = expected.code
                    break
            # Known FP traps (corpus-attested checker artifacts) outrank TP matching.
            fp_hit = matched in known_fp if matched else any(
                _compact(code) in haystack_flat for code in known_fp
            )
            if fp_hit:
                fp_code = matched if matched in known_fp else next(
                    code for code in known_fp
                    if _compact(code) in haystack_flat
                )
                results.append((
                    replace(finding, classification=FindingClass.FALSE_POSITIVE,
                            matched_code=fp_code, unique=False),
                    Adjudication(finding_id=finding.finding_id,
                                 classification=FindingClass.FALSE_POSITIVE,
                                 matched_code=fp_code, method="deterministic_fp_trap",
                                 provenance={"case_id": case.case_id, "source": "frozen_corpus"}),
                ))
                continue
            if matched is None:
                classification, method = FindingClass.UNVERIFIABLE, "no_deterministic_match"
            elif not defect_present:
                # Clean twin: the defect is absent by construction, so a match
                # is a false alarm, never a detection.
                classification, method = FindingClass.FALSE_POSITIVE, "clean_twin_false_alarm"
            elif matched in seen_codes:
                classification, method = FindingClass.DUPLICATE, "code_already_counted"
            else:
                classification, method = FindingClass.TRUE_POSITIVE, "deterministic_match"
                seen_codes.add(matched)
            adjudication = Adjudication(
                finding_id=finding.finding_id,
                classification=classification,
                matched_code=matched,
                method=method,
                provenance={"case_id": case.case_id, "source": "frozen_corpus"},
            )
            results.append((
                replace(finding, classification=classification, matched_code=matched,
                        unique=(classification == FindingClass.TRUE_POSITIVE)),
                adjudication,
            ))
        return results
