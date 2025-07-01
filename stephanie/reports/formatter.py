import re
from datetime import datetime, timezone
from pathlib import Path


class ReportFormatter:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_report(self, context):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        item = context.get("goal")
        if isinstance(item, str):
            goal = item
        else:
            goal = item.get("goal_text", "Error No Goal")

        safe_goal = sanitize_goal_for_filename(goal)
        file_name = f'{safe_goal}_{timestamp}_report.md'
        file_path = self.output_dir / file_name

        content = f"""# 🧪 AI Co-Research Summary Report

**🗂️ Run ID:** `{context.get("run_id", "Error No Run_id")}`  
**🎯 Goal:** *{goal}*  
**📅 Timestamp:** {timestamp}

---

### 🔬 Literature:
{self._format_list(context.get("literature", []))}



### 🔬 Hypotheses Generated:
{self._format_list([h if isinstance(h, str) else h.get("text", "") for h in context.get("hypotheses", [])])}
OK so 


---

### 🪞 Reflections:
{self._format_reflections(context.get("reflections", []))}


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

### 📘 DB Matches:
{context.get("proximity", {}).get("database_matches", [])}


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

    def _format_reflections(self, reflections):
        if not reflections:
            return "No reflections recorded."
        formatted = []
        for r in reflections:
            reflection = r.get("reflection", "")
            formatted.append(f"**Reflection:**\n> {reflection}")
        return "\n\n".join(formatted)



def sanitize_goal_for_filename(goal: str, length:int=40) -> str:
    """
    Converts a goal string into a safe filename:
    - Replaces non-alphanumeric characters with underscores
    - Truncates to 100 characters
    - Appends a UTC timestamp
    """
    safe = re.sub(r'[^a-zA-Z0-9]', '_', goal)  # Replace non-alphanumeric
    safe = safe[:length]                                       # Limit to (len) characters
    return safe
