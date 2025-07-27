# sis/dashboard.py

import gradio as gr
import sqlite3
import json

DB_PATH = "insight_data.db"

# Query recent insights
def fetch_insights():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, goal, agent, stage, notes, context, scores FROM insights ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return rows

# Render insights as a table
def view_insights():
    rows = fetch_insights()
    formatted = []
    for row in rows:
        timestamp, goal, agent, stage, notes, context_json, scores_json = row
        context = json.dumps(json.loads(context_json), indent=2)
        scores = json.dumps(json.loads(scores_json), indent=2)
        formatted.append([
            timestamp, goal, agent, stage, notes, context, scores
        ])
    return formatted

headers = ["Timestamp", "Goal", "Agent", "Stage", "Notes", "Context", "Scores"]

with gr.Blocks() as demo:
    gr.Markdown("# 📊 Stephanie Insight Dashboard")
    gr.Dataframe(headers=headers, value=view_insights, interactive=False)

if __name__ == "__main__":
    demo.launch()
