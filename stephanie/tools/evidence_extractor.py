# stephanie/tools/evidence_extractor.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import time

@dataclass
class TransferEdge:
    from_paper: Optional[str]
    to_paper: Optional[str]
    section: Optional[str]
    ts: float
    run_id: Optional[str]
    agent: Optional[str]
    event: Optional[str]
    # Learning-specific additions
    transfer_type: str = "procedural"  # conceptual, procedural, metacognitive
    confidence: float = 0.7
    learning_score: float = 0.0

def _safe_str(x) -> Optional[str]:
    if x is None: return None
    s = str(x).strip()
    return s or None

def _ts(row) -> float:
    try:
        t = float(row.get("ts") or 0.0)
    except Exception:
        t = 0.0
    if t <= 0:
        try:
            t = float(row.get("extras", {}).get("publisher_ts") or 0.0)
        except Exception:
            t = 0.0
    return t

def _rowkey(r: dict) -> Tuple:
    return (
        _safe_str(r.get("run_id")),
        _safe_str(r.get("paper_id")),
        _safe_str(r.get("section_name")),
        _safe_str(r.get("event")),
        _ts(r),
        _safe_str(r.get("agent")),
    )

def _normalize_payloads(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        out.append({
            "id": r.get("id"),
            "ts": _ts(r),
            "event": _safe_str(r.get("event")),
            "run_id": _safe_str(r.get("run_id")),
            "paper_id": _safe_str(r.get("paper_id")),
            "section_name": _safe_str(r.get("section_name")),
            "agent": _safe_str(r.get("agent")),
            "payload": r.get("payload") or {},
            "subject": _safe_str(r.get("subject")),
            "extras": r.get("extras", r.get("extras_json") or {}) or {},
            "goal": _safe_str(r.get("goal")),
            "title": _safe_str(r.get("title")),
            "goal_text": _safe_str(r.get("goal_text")),
        })
    
    # Dedupe exact duplicates
    seen = set()
    uniq = []
    for r in out:
        k = _rowkey(r)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    
    # Sort by timestamp (critical for chronological learning evidence)
    return sorted(uniq, key=lambda x: (x["ts"], x["id"] or 0))

def _determine_transfer_type(prev: dict, current: dict) -> str:
    """Determine the type of knowledge transfer based on context"""
    # Conceptual: theory foundations being applied
    if "theory" in str(prev.get("payload", "")).lower() and "theory" in str(current.get("payload", "")).lower():
        return "conceptual"
    
    # Procedural: methods being reused (most common)
    if "method" in str(prev.get("section_name", "")).lower() and "method" in str(current.get("section_name", "")).lower():
        return "procedural"
    
    # Metacognitive: strategy adaptation across contexts
    if "strategy" in str(prev.get("payload", "")).lower() or "approach" in str(prev.get("payload", "")).lower():
        return "metacognitive"
    
    return "procedural"  # default

def _calculate_confidence(prev: dict, current: dict, time_diff: float) -> float:
    """Calculate confidence in knowledge transfer based on context"""
    base = 0.7
    
    # Higher confidence if same section type
    if prev.get("section_name") == current.get("section_name"):
        base += 0.15
    
    # Higher confidence if short time between papers
    if time_diff < 3600:  # within 1 hour
        base += 0.1
    
    # Higher confidence if same agent involved
    if prev.get("agent") == current.get("agent"):
        base += 0.1
    
    return min(0.95, max(0.5, base))  # cap between 0.5 and 0.95

def _calculate_learning_score(confidence: float, transfer_type: str) -> float:
    """Calculate weighted learning score (0-10)"""
    type_weights = {
        "conceptual": 1.2,
        "procedural": 1.0,
        "metacognitive": 1.5
    }
    return min(10, confidence * 10 * type_weights.get(transfer_type, 1.0))

def build_transfer_matrix_from_rows(rows: List[dict]) -> Dict[str, Any]:
    if not rows:
        return {
            "kpi": {
                "total_events": 0,
                "runs": 0,
                "papers": 0,
                "sections": 0,
                "agents": 0,
                "cross_paper_transfers": 0,
                "learning_score": 0.0,
                "diagnostics": {
                    "section_coverage": 0.0,
                    "first_transfer_latency": 0.0,
                    "transfers_per_hour": 0.0,
                    "events_per_transfer": float("inf"),
                    "longest_chain": 0
                }
            },
            "sections_covered": [],
            "agents": [],
            "top_pairs": [],
            "knowledge_transfer_evidence": [],
            "learning_insights": []
        }
    
    rows = _normalize_payloads(rows)
    papers = set()
    sections = set()
    agents = set()
    runs = set()
    total_learning_score = 0.0

    for r in rows:
        if r["paper_id"]: papers.add(r["paper_id"])
        if r["section_name"]: sections.add(r["section_name"])
        if r["agent"]: agents.add(r["agent"])
        if r["run_id"]: runs.add(r["run_id"])

    # Cross-paper edges: whenever consecutive *global* events switch paper_id
    edges: List[TransferEdge] = []
    for i in range(1, len(rows)):
        prev, cur = rows[i-1], rows[i]
        if prev["paper_id"] != cur["paper_id"] and prev["paper_id"] and cur["paper_id"]:
            time_diff = cur["ts"] - prev["ts"]
            transfer_type = _determine_transfer_type(prev, cur)
            confidence = _calculate_confidence(prev, cur, time_diff)
            learning_score = _calculate_learning_score(confidence, transfer_type)
            
            total_learning_score += learning_score
            
            edges.append(TransferEdge(
                from_paper=prev["paper_id"],
                to_paper=cur["paper_id"],
                section=cur["section_name"],
                ts=cur["ts"],
                run_id=cur["run_id"],
                agent=cur["agent"],
                event=cur["event"],
                transfer_type=transfer_type,
                confidence=confidence,
                learning_score=learning_score
            ))

    # Calculate overall learning score (average of all transfer scores)
    avg_learning_score = total_learning_score / len(edges) if edges else 0.0

    # Calculate diagnostic metrics
    diagnostics = _calculate_diagnostics(rows, edges)
    
    # Summarize
    by_pair = Counter((e.from_paper, e.to_paper) for e in edges if e.from_paper and e.to_paper)
    top_pairs = sorted(by_pair.items(), key=lambda kv: kv[1], reverse=True)[:25]

    kpi = {
        "total_events": len(rows),
        "runs": len(runs),
        "papers": len(papers),
        "sections": len(sections),
        "agents": len(agents),
        "cross_paper_transfers": len(edges),
        "learning_score": round(avg_learning_score, 1),
        "diagnostics": diagnostics
    }

    evidence = [{
        "from_paper": e.from_paper,
        "to_paper": e.to_paper,
        "section": e.section,
        "timestamp": e.ts,
        "run_id": e.run_id,
        "agent": e.agent,
        "event": e.event,
        "transfer_type": e.transfer_type,
        "confidence": e.confidence,
        "learning_score": e.learning_score,
        "evidence": f"Knowledge transfer from {e.from_paper} → {e.to_paper} in {e.section or 'N/A'} section",
        "type_label": e.transfer_type.replace("_", " ").title()
    } for e in edges]

    # Generate learning insights
    learning_insights = _generate_learning_insights(rows, edges, kpi)

    return {
        "kpi": kpi,
        "sections_covered": sorted([s for s in sections if s]),
        "agents": sorted([a for a in agents if a]),
        "top_pairs": [{"from": fp, "to": tp, "count": c} for ((fp, tp), c) in top_pairs],
        "knowledge_transfer_evidence": evidence,
        "learning_insights": learning_insights
    }

def _calculate_diagnostics(rows: List[dict], edges: List[TransferEdge]) -> Dict[str, Any]:
    """Calculate diagnostic metrics from the event stream and transfer edges"""
    if not rows:
        return {
            "section_coverage": 0.0,
            "first_transfer_latency": 0.0,
            "transfers_per_hour": 0.0,
            "events_per_transfer": float("inf"),
            "longest_chain": 0
        }
    
    # Sort rows by timestamp
    rows_sorted = sorted(rows, key=lambda r: r.get("ts") or 0)
    first_ts = rows_sorted[0].get("ts") or 0
    last_ts = rows_sorted[-1].get("ts") or first_ts
    duration_sec = max(0, (last_ts - first_ts))
    
    # Get sections from the run
    run_sections = {r.get("section_name") for r in rows if r.get("section_name")}
    
    # Calculate coverage
    covered_sections = {e.section for e in edges if e.section}
    section_coverage = (len(covered_sections) / len(run_sections)) if run_sections else 0.0
    
    # Calculate latency to first transfer
    first_edge_ts = min((e.ts for e in edges), default=None)
    first_transfer_latency = (first_edge_ts - first_ts) if (first_edge_ts and first_ts) else 0.0
    
    # Calculate transfer rate
    hours = max(1e-6, duration_sec / 3600.0)
    transfers_per_hour = len(edges) / hours if duration_sec > 0 else 0.0
    
    # Calculate event density
    events_per_transfer = (len(rows) / len(edges)) if edges else float("inf")
    
    # Calculate longest chain
    longest_chain = 0
    if edges:
        chain_len = 1
        for i in range(1, len(edges)):
            prev, cur = edges[i-1], edges[i]
            if prev.to_paper == cur.from_paper:
                chain_len += 1
            else:
                longest_chain = max(longest_chain, chain_len)
                chain_len = 1
        longest_chain = max(longest_chain, chain_len)
    
    return {
        "section_coverage": round(section_coverage, 2),
        "first_transfer_latency": round(first_transfer_latency, 2),
        "transfers_per_hour": round(transfers_per_hour, 2),
        "events_per_transfer": round(events_per_transfer, 2) if events_per_transfer != float("inf") else float("inf"),
        "longest_chain": longest_chain
    }

def _generate_learning_insights(rows: List[dict], edges: List[TransferEdge], kpi: Dict) -> List[Dict]:
    """
    Generate meaningful insights about the learning patterns observed.
    Uses BOTH rows (full event stream) and edges (detected transfers).
    Returns up to ~5 concise insights with titles, content, type, recommendation.
    """
    insights: List[Dict[str, Any]] = []

    # Guard rails
    total_events = len(rows)
    total_edges = len(edges)
    if total_events == 0:
        return [{
            "title": "No Activity",
            "content": "No events recorded; cannot assess learning or transfer.",
            "type": "metacognitive",
            "recommendation": "Verify the pipeline run emitted bus events and retry."
        }]

    # ---------- Basic aggregates from rows ----------
    rows_sorted = sorted(rows, key=lambda r: r.get("ts") or 0)
    first_ts = rows_sorted[0].get("ts") or 0
    last_ts  = rows_sorted[-1].get("ts") or first_ts
    duration_sec = max(0, (last_ts - first_ts))

    run_sections = { (r.get("section_name") or "").strip() for r in rows_sorted if r.get("section_name") }
    run_agents   = { (r.get("agent") or "").strip() for r in rows_sorted if r.get("agent") }

    # ---------- Aggregates from edges ----------
    # Section/agent counts over transfers
    section_counts = Counter(e.section for e in edges if e.section)
    agent_counts   = Counter(e.agent for e in edges if e.agent)
    type_counts    = Counter(e.transfer_type for e in edges)

    # Coverage: how many run sections actually saw a transfer?
    covered_sections = set(section_counts.keys())
    section_coverage = (len(covered_sections) / max(1, len(run_sections))) if run_sections else 0.0
    uncovered_sections = sorted(run_sections - covered_sections)

    # Latency: first transfer after the run's first event
    first_edge_ts = min((e.ts for e in edges), default=None)
    latency_sec = (first_edge_ts - first_ts) if (first_edge_ts is not None and first_ts) else None

    # Transfer rate & density
    hours = max(1e-6, duration_sec / 3600.0)          # avoid div by zero
    transfers_per_hour = total_edges / hours if duration_sec > 0 else 0.0
    events_per_transfer = (total_events / total_edges) if total_edges > 0 else float("inf")

    # Chain depth (longest contiguous evolution path in chronological order)
    # Edges are chronological because you created them in order of rows.
    longest_chain_len = 0
    if total_edges > 0:
        chain_len = 1
        for i in range(1, total_edges):
            prev, cur = edges[i-1], edges[i]
            if prev.to_paper == cur.from_paper:
                chain_len += 1
            else:
                longest_chain_len = max(longest_chain_len, chain_len)
                chain_len = 1
        longest_chain_len = max(longest_chain_len, chain_len)

    # Learning score (already computed upstream)
    learning_score = kpi.get("learning_score", 0.0)

    # ---------- Insight 1: Learning Score level ----------
    if total_edges == 0:
        insights.append({
            "title": "Initial Learning Phase",
            "content": "No explicit cross-paper transfers were detected. The run may be establishing foundational knowledge.",
            "type": "metacognitive",
            "recommendation": "Repeat with similar content or add explicit knowledge mapping to encourage transfer."
        })
    else:
        if learning_score >= 7.0:
            insights.append({
                "title": "Strong Knowledge Integration",
                "content": f"Learning score {learning_score}/10 suggests effective transfer across contexts.",
                "type": "metacognitive",
                "recommendation": "Document the configuration and re-use for similar problems."
            })
        elif learning_score >= 4.0:
            insights.append({
                "title": "Developing Learning Patterns",
                "content": f"Moderate transfer observed (score {learning_score}/10). Patterns are emerging.",
                "type": "procedural",
                "recommendation": "Increase conceptual linking stages to improve transfer strength."
            })
        else:
            insights.append({
                "title": "Learning Opportunity Identified",
                "content": f"Limited transfer (score {learning_score}/10) indicates learning gaps.",
                "type": "conceptual",
                "recommendation": "Introduce explicit cross-paper context alignment before application stages."
            })

    # ---------- Insight 2: Coverage (uses rows) ----------
    if run_sections:
        pct = round(section_coverage * 100)
        if section_coverage >= 0.6:
            insights.append({
                "title": "Broad Section Coverage",
                "content": f"{pct}% of sections involved at least one transfer.",
                "type": "procedural",
                "recommendation": "Preserve section-general strategies; they appear widely applicable."
            })
        else:
            tip = ""
            if uncovered_sections:
                tip = f" Missing sections include: {', '.join(uncovered_sections[:3])}" + ("…" if len(uncovered_sections) > 3 else "")
            insights.append({
                "title": "Narrow Section Coverage",
                "content": f"Only {pct}% of sections show transfer activity.{tip}",
                "type": "conceptual",
                "recommendation": "Add targeted prompts or bridging steps for under-represented sections."
            })

    # ---------- Insight 3: Latency & Rate (uses rows) ----------
    if latency_sec is not None:
        mins = int(round(latency_sec / 60))
        if mins < 5:
            insights.append({
                "title": "Rapid Learning Kickoff",
                "content": f"First transfer occurred after just {mins} minute(s) from run start.",
                "type": "metacognitive",
                "recommendation": "This quick knowledge application suggests effective initial setup."
            })
        else:
            insights.append({
                "title": "Time-to-First-Transfer",
                "content": f"First transfer occurred after {mins} minute(s) from run start.",
                "type": "metacognitive",
                "recommendation": "Reduce early-stage overhead or add pre-warmed context to speed up first transfer."
            })
    
    if duration_sec > 0:
        if transfers_per_hour > 5:
            insights.append({
                "title": "High Transfer Throughput",
                "content": f"Observed {transfers_per_hour:.2f} transfer(s) per hour; event density is {events_per_transfer:.1f} events/transfer.",
                "type": "procedural",
                "recommendation": "This high throughput indicates efficient knowledge transfer processing."
            })
        else:
            insights.append({
                "title": "Transfer Throughput",
                "content": f"Observed {transfers_per_hour:.2f} transfer(s) per hour; event density is {events_per_transfer:.1f} events/transfer.",
                "type": "procedural",
                "recommendation": "Tune batching or cache strategies to increase transfer throughput."
            })

    # ---------- Insight 4: Type distribution (uses edges) ----------
    total = max(1, total_edges)
    conceptual_share = type_counts.get("conceptual", 0) / total
    metacog_share   = type_counts.get("metacognitive", 0) / total
    if conceptual_share > 0.40:
        insights.append({
            "title": "Strong Conceptual Foundation",
            "content": f"Conceptual transfers account for {conceptual_share:.0%} of all transfers.",
            "type": "conceptual",
            "recommendation": "Leverage conceptual strength to tackle harder theory-heavy tasks."
        })
    if metacog_share > 0.25:
        insights.append({
            "title": "Advanced Strategy Adaptation",
            "content": f"Metacognitive transfers represent {metacog_share:.0%} — signaling strategic adaptation.",
            "type": "metacognitive",
            "recommendation": "Capture these strategies as reusable policies in the pipeline."
        })

    # ---------- Insight 5: Hotspots & Contributors (uses edges) ----------
    if section_counts:
        top_section, cnt = section_counts.most_common(1)[0]
        insights.append({
            "title": f"Knowledge Application Hotspot: {top_section}",
            "content": f"{cnt} transfer(s) concentrated in {top_section}.",
            "type": "procedural",
            "recommendation": f"Optimize prompts/config for '{top_section}' to amplify this advantage."
        })
    if agent_counts:
        top_agent, cnt = agent_counts.most_common(1)[0]
        share = round((cnt / total) * 100)
        insights.append({
            "title": f"Top Learning Contributor: {top_agent}",
            "content": f"{share}% of transfers were executed by {top_agent}.",
            "type": "procedural",
            "recommendation": f"Study {top_agent}'s traces to standardize effective transfer patterns."
        })

    # ---------- Insight 6 (optional): Chain depth (uses edges) ----------
    if longest_chain_len >= 3:
        insights.append({
            "title": "Knowledge Evolution Chain",
            "content": f"Longest observed chain spans {longest_chain_len} transfers (cumulative learning across papers).",
            "type": "metacognitive",
            "recommendation": "Persist this chain as a pattern; it indicates stable, compounding transfer."
        })

    # Keep it tight
    return insights[:5]
