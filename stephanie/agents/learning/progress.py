# stephanie/agents/learning/progress.py
from __future__ import annotations

class ProgressAdapter:
    def __init__(self, agent_progress, cfg):
        self.core = agent_progress  # reuse AgentProgress you already create
        self.cfg = cfg or {}

    def start_paper(self, paper, sections):
        self.core.start_paper(
            paper.get("id") or paper.get("doc_id"),
            paper.get("title", ""),
            len(sections),
        )

    def start_section(self, section, idx):
        self.core.start_section(section.get("section_name", "unknown"), idx)

    def stage(self, section, stage, **kv):
        self.core.stage(stage, section.get("section_name", "unknown"), **kv)

    def end_section(self, case, section, metrics):
        self.core.end_section(section.get("section_name", "unknown"), metrics)
