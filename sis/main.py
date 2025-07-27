# sis/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import json
import os
import asyncio

app = FastAPI()

DB_PATH = "insight_data.db"
LOG_DIRECTORY = "../logs Just do one thing"
FILE_POSITIONS = {}  # Tracks last read position in each file

# Define data schema
class InsightPayload(BaseModel):
    timestamp: datetime
    goal: str
    agent: str
    stage: str
    context: dict
    scores: dict
    notes: str = ""

# Initialize database
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        goal TEXT,
        agent TEXT,
        stage TEXT,
        context TEXT,
        scores TEXT,
        notes TEXT
    )''')
    conn.commit()
    conn.close()

# Logging endpoint
@app.post("/log")
async def log_insight(payload: InsightPayload):
    insert_payload(payload)
    return {"status": "ok"}

@app.get("/")
async def index():
    return {"message": "Insight Service is running."}

@app.get("/insights")
async def get_insights():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, goal, agent, stage, notes FROM insights ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return [
        {"timestamp": row[0], "goal": row[1], "agent": row[2], "stage": row[3], "notes": row[4]}
        for row in rows
    ]

def insert_payload(payload: InsightPayload):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO insights (timestamp, goal, agent, stage, context, scores, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (payload.timestamp.isoformat(),
               payload.goal,
               payload.agent,
               payload.stage,
               json.dumps(payload.context),
               json.dumps(payload.scores),
               payload.notes))
    conn.commit()
    conn.close()

async def monitor_log_directory():
    while True:
        if os.path.exists(LOG_DIRECTORY):
            for file in os.listdir(LOG_DIRECTORY):
                if file.endswith(".jsonl"):
                    path = os.path.join(LOG_DIRECTORY, file)
                    try:
                        if path not in FILE_POSITIONS:
                            FILE_POSITIONS[path] = 0
                        with open(path, "r", encoding="utf-8") as f:
                            f.seek(FILE_POSITIONS[path])
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                                    payload = InsightPayload(**data)
                                    insert_payload(payload)
                                except Exception as e:
                                    print(f"Failed to process line in {file}: {e}")
                            FILE_POSITIONS[path] = f.tell()
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")
        await asyncio.sleep(5)

@app.on_event("startup")
async def start_monitor():
    asyncio.create_task(monitor_log_directory())

# Run with: uvicorn insight_service.main:app --reload
