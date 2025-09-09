# stephanie/tools/plan_from_document.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _abbr_from_text(text: str) -> Dict[str, str]:
    # find patterns like "Long Phrase (LP)" and "LP = Long Phrase"
    abbr = {}
    for m in re.finditer(r"\b([A-Z][A-Za-z][\w\s\-]{3,})\s+\(([A-Z]{2,7})\)", text):
        full, short = m.group(1).strip(), m.group(2).strip()
        abbr[full] = short
    for m in re.finditer(r"\b([A-Z]{2,7})\s*=\s*([A-Z][A-Za-z][\w\s\-]{3,})", text):
        short, full = m.group(1).strip(), m.group(2).strip()
        abbr[full] = short
    return abbr

def _claims_from_section(section_text: str, max_units=8) -> List[Dict[str, Any]]:
    # select strong sentences that sound like claims/results
    sents = re.split(r'(?<=[.!?])\s+', section_text.strip())
    picks = []
    keys = ("we propose", "our method", "improv", "outperform", "reduce", "achiev", "state-of-the-art",
            "ablation", "significant", "accuracy", "error", "throughput", "latency")
    for s in sents:
        if any(k in s.lower() for k in keys) and len(s) > 40:
            picks.append(s.strip())
        if len(picks) >= max_units:
            break
    return [{"claim_id": f"C{i+1}", "claim": c, "evidence": "See paper"} for i, c in enumerate(picks)]

def build_plan_from_memory(memory, document_id: int, section_preference=("methods","results","approach","experiments")) -> Dict[str, Any]:
    doc = memory.documents.get(document_id)
    if not doc:
        raise ValueError(f"Document id {document_id} not found")
    # fetch sections (if profiled) else fall back to full text
    sections = {s.section_name.lower(): s.section_text for s in memory.document_sections.get_by_document(document_id) or []}
    paper_text = doc.text or doc.content or ""
    section_title = None
    chosen_text = ""
    for name in section_preference:
        if name in sections and len(sections[name]) > 200:
            section_title, chosen_text = name.title(), sections[name]
            break
    if not section_title:
        section_title, chosen_text = "Body", (sections.get("body") or paper_text)

    units = _claims_from_section(chosen_text, max_units=8)
    abbr = _abbr_from_text(paper_text[:20000])  # cap to keep fast

    return {
        "section_title": section_title,
        "units": units,
        "entities": {"ABBR": abbr, "REQUIRED": list({w for w in ("accuracy","latency","throughput") if w in paper_text.lower()})},
        "paper_text": paper_text
    }

def save_plan(plan: Dict[str, Any], out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(plan, indent=2))
    return out_path
