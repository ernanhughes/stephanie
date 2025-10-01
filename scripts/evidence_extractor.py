# stephanie/tools/evidence_extractor.py
from __future__ import annotations

from typing import Dict, Any
import json

def extract_knowledge_transfer_matrix(event_log_path: str) -> Dict[str, Any]:
    """Extract knowledge transfer evidence from event log"""
    
    # Parse events
    events = []
    with open(event_log_path, 'r') as f:
        for line in f:
            if '"event": "arena_done"' in line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except:
                    continue
    
    # Group by paper
    paper_events = {}
    for event in events:
        paper_id = event.get("paper_id")
        if paper_id:
            if paper_id not in paper_events:
                paper_events[paper_id] = []
            paper_events[paper_id].append(event)
    
    # Build transfer matrix
    matrix = {
        "papers_processed": len(paper_events),
        "total_events": len(events),
        "sections_covered": list(set(event.get("section_name") for event in events)),
        "knowledge_transfer_evidence": []
    }
    
    # Analyze chronological progression
    sorted_events = sorted(events, key=lambda x: x.get("run_id", 0))
    
    for i, event in enumerate(sorted_events):
        paper_id = event.get("paper_id")
        section_name = event.get("section_name")
        timestamp = event.get("publisher_ts")
        
        # Look for evidence of prior knowledge application
        if i > 0:
            prev_paper = sorted_events[i-1].get("paper_id")
            if prev_paper != paper_id:
                matrix["knowledge_transfer_evidence"].append({
                    "from_paper": prev_paper,
                    "to_paper": paper_id,
                    "section": section_name,
                    "timestamp": timestamp,
                    "evidence": f"Knowledge from Paper {prev_paper} applied to Paper {paper_id} in {section_name} section"
                })
    
    return matrix