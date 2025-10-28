# stephanie/reports/formatter.py
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


class ReportFormatter:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_report(self, context: dict):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Goal
        item = context.get("goal")
        if isinstance(item, str):
            goal = item
        else:
            goal = item.get("goal_text", "Error No Goal")

        safe_goal = sanitize_goal_for_filename(goal)
        file_name = f"{safe_goal}_{timestamp}_report.md"
        file_path = self.output_dir / file_name

        # Pipeline header (printed once)
        header = f"""# 📝 Pipeline Report

**🗂️ Run ID:** `{context.get("run_id", "Error No Run_id")}`  
**🎯 Goal:** *{goal}*  
**📅 Timestamp:** {timestamp}  
**📁 Saved Report:** `{file_path}`

---
"""

        # Render each stage
        stage_reports = []
        for report in context.get("REPORTS", []):
            stage_reports.append(self._format_stage(report))

        content = header + "\n".join(stage_reports)
        file_path.write_text(content, encoding="utf-8")

        return content, header, str(file_path)

    def _format_stage(self, report: dict) -> str:
        stage_header = f"## 🔧 Stage: {report.get('stage')} ({report.get('agent')})"
        status_line = f"- **Status:** {report.get('status', '-')}"
        summary_line = f"- **Summary:** {report.get('summary','')}"
        timing_line = f"- **Start:** {report.get('start_time','?')} | **End:** {report.get('end_time','?')}"

        # Entries
        entries = report.get("entries", [])
        if entries:
            formatted_entries = "\n".join(
                f"  - **{e.get('event','unknown')}**: {self._format_entry(e)}"
                for e in entries
            )
            entries_block = f"\n<details>\n<summary>📜 Events ({len(entries)})</summary>\n\n{formatted_entries}\n</details>\n"
        else:
            entries_block = "_No events recorded._"

        return f"""{stage_header}
{status_line}  
{summary_line}  
{timing_line}

{entries_block}

---
"""

    def _format_entry(self, e: dict) -> str:
        """Render a single event dict into a compact string"""
        # Prioritize common keys
        if "message" in e:
            return e["message"]
        if "hypothesis_id" in e:
            return f"Hypothesis {e['hypothesis_id']} (confidence={e.get('confidence','?')})"
        if "scores" in e:
            return f"Scores: {e['scores']}"
        if "examples" in e:
            return f"Examples: {e['examples'][:2]}..."
        return ", ".join(f"{k}={v}" for k, v in e.items() if k != "event")


def sanitize_goal_for_filename(goal: str, length: int = 40) -> str:
    """Converts a goal string into a safe filename"""
    safe = re.sub(r"[^a-zA-Z0-9]", "_", goal)
    safe = safe[:length]
    return safe
