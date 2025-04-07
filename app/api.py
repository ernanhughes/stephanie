from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app.search import run_search
from app.db import connect_db
from psycopg2.extras import RealDictCursor
import html
from datetime import datetime
import logging

# Setup logger
logger = logging.getLogger("stephanie.api")
logging.basicConfig(level=logging.INFO)

import html
import re

def clean_text(text: str) -> str:
    """Clean badly encoded strings, spaced tokens, and escaped sequences."""
    if not text:
        return ""
    # Fix spaced-out characters
    if re.match(r"^(?:[A-Za-z0-9]\s+){5,}", text):
        text = re.sub(r"\s+", "", text)
    # Decode unicode-escaped sequences
    text = text.encode("utf-8").decode("unicode_escape", errors="ignore")
    # Unescape HTML entities
    text = html.unescape(text)
    # Clean up extra quotes
    return text.replace('\\"', '"').strip()



router = APIRouter()

# -------------------------------
# Models
# -------------------------------
class SearchRequest(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"

class UpdateRequest(BaseModel):
    summary: Optional[str]
    tags: Optional[List[str]]

# -------------------------------
# Routes
# -------------------------------

@router.post("/search")
def search_memory(payload: SearchRequest):
    logger.info(f"🔍 /search called with query='{payload.query}', mode='{payload.mode}'")
    results = run_search(payload.query, payload.mode)
    logger.info(f"🔍 /search returning {len(results)} results")
    return [r.__dict__ for r in results]

@router.get("/memory/{id}")
def get_memory(id: int):
    logger.info(f"📥 /memory/{id} requested")
    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM memory WHERE id = %s", (id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        logger.warning(f"⚠️ /memory/{id} not found")
        raise HTTPException(status_code=404, detail="Memory not found")

    return row

@router.patch("/memory/{id}")
def update_memory(id: int, payload: UpdateRequest):
    logger.info(f"✏️ /memory/{id} update request: summary='{payload.summary}', tags={payload.tags}")
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE memory SET
            summary = %s,
            tags = %s
        WHERE id = %s
    """, (payload.summary, payload.tags, id))
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"✅ /memory/{id} successfully updated")
    return {"status": "updated"}

@router.get("/export/{id}")
def export_memory(id: int):
    logger.info(f"📤 /export/{id} requested")
    conn = connect_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM memory WHERE id = %s", (id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        logger.warning(f"⚠️ /export/{id} not found")
        raise HTTPException(status_code=404, detail="Memory not found")

    # Clean everything
    title = clean_text(row["title"])
    summary = clean_text(row.get("summary", ""))
    user_text = clean_text(row.get("user_text", ""))
    ai_text = clean_text(row.get("ai_text", ""))

    # Build markdown
    md = f"# {title}\n\n"
    md += f"**Timestamp:** {row['timestamp']}\n\n"
    md += f"**Tags:** {', '.join(row.get('tags', []))}\n\n"
    md += f"**Summary:**\n{summary}\n\n---\n\n"
    md += f"**User:**\n{user_text}\n\n"
    md += f"**Assistant:**\n{ai_text}\n"

    return {
        "markdown": md,
        "json": row
    }

