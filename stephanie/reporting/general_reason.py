# stephanie/reports/general_reason.py
from datetime import datetime, timezone
from pathlib import Path


def save_markdown_report(result: dict, out_dir: str = "./reports"):
    goal = result["goal"]["goal_text"]
    run_id = result.get("run_id", datetime.now(timezone.utc).isoformat())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

    report_md = f"""# General Reasoner Run Report

**Run ID:** {run_id}  
**Timestamp:** {timestamp}  
**Goal:** {goal}

---

## 🧠 Hypotheses

"""
    for hyp in result["hypotheses"]:
        strategy = hyp.get("strategy", hyp["features"].get("strategy", "unknown"))
        report_md += f"""### Strategy: `{strategy}`  
{hyp["text"]}

---
"""

    report_md += "\n## 🧪 Judgments\n\n"
    for score in result["scoring"]:
        strategy_a = score["hypothesis_a"][:50].replace("\n", " ") + "..."
        strategy_b = score["hypothesis_b"][:50].replace("\n", " ") + "..."
        winner = score["winner"]
        reason = score["reason"]
        report_md += f"""- **Winner:** `{winner}`  
  - A: {strategy_a}  
  - B: {strategy_b}  
  - Reason: {reason}\n\n"""

    # Optional: Add per-strategy summary
    from collections import defaultdict

    score_stats = defaultdict(list)
    for score in result["scoring"]:
        score_stats[score["winner"]].append(
            score.get("score_b", 0)
        )  # use score_b as winner

    report_md += "\n## 📊 Strategy Performance\n\n"
    report_md += "| Strategy | Judged Wins | Avg Score |\n|----------|--------------|------------|\n"
    for strategy, scores in score_stats.items():
        avg_score = round(sum(scores) / len(scores), 2)
        report_md += f"| {strategy} | {len(scores)} | {avg_score} |\n"

    # Save the markdown
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"general_reasoner_report_{timestamp}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"✅ Markdown report saved to {out_path}")
