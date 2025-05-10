import os
from datetime import datetime, timezone
from pathlib import Path
import re
from datetime import datetime

class ReportFormatter:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_report(self, context):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        goal = context.get("goal", "Error No Goal")
        safe_goal = sanitize_goal_for_filename(goal)
        file_name = f'{safe_goal}_{timestamp}_report.md'
        file_path = self.output_dir / file_name

        content = f"""# 🧪 AI Co-Research Summary Report

**🗂️ Run ID:** `{context.get("run_id", "Error No Run_id")}`  
**🎯 Goal:** *{goal}*  
**📅 Timestamp:** {timestamp}

---

### 🔬 Hypotheses Generated:
{self._format_list(context.get("generated", []))}

---

### 🧠 Persona Reviews:
{self._format_reviews(context.get("reviews", []))}

---

### 🧬 Evolution Outcome:
- {len(context.get("evolved", []))} hypotheses evolved.

---

### 📘 Meta-Review Summary:
> {context.get("meta_review", "")}


### 📘 Feedback:
{context.get("feedback", "")}


---
"""

        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    def _format_list(self, items):
        return "\n".join(f"1. **{item.strip()}**" for item in items)

    def _format_reviews(self, reviews):
        if not reviews:
            return "No reviews recorded."
        formatted = []
        for r in reviews:
            persona = r.get("persona", "Unknown")
            review = r.get("review", "")
            formatted.append(f"**{persona}:**\n> {review}")
        return "\n\n".join(formatted)


def sanitize_goal_for_filename(goal: str) -> str:
    """
    Converts a goal string into a safe filename:
    - Replaces non-alphanumeric characters with underscores
    - Truncates to 100 characters
    - Appends a UTC timestamp
    """
    safe = re.sub(r'[^a-zA-Z0-9]', '_', goal)  # Replace non-alphanumeric
    safe = safe[:100]                          # Limit to 100 characters
    timestamp = datetime.utcnow().isoformat().replace(":", "-")
    return f"{safe}_{timestamp}_report.md"
