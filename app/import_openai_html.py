import os
import sys
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
from embedding import get_embedding
from db import connect_db

def parse_messages_html(path):
    print(f"Parsing HTML file: {path}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    conversations = []
    all_threads = soup.select(".message-group")

    for i, group in enumerate(all_threads):
        user_messages = group.select(".user-message")
        assistant_messages = group.select(".assistant-message")

        user_text = "\n".join([m.get_text(strip=True) for m in user_messages])
        ai_text = "\n".join([m.get_text(strip=True) for m in assistant_messages])

        if not user_text and not ai_text:
            continue

        combined = f"{user_text} {ai_text}".strip()
        if not combined:
            continue

        title = f"OpenAI HTML Chat #{i+1}"
        timestamp = datetime.utcnow().isoformat()

        conversations.append({
            "title": title,
            "timestamp": timestamp,
            "user_text": user_text,
            "ai_text": ai_text,
            "summary": "",
            "tags": ["openai", "html"],
            "source": "openai.html",
            "importance": 0,
            "archived": False,
            "openai_url": None
        })

    return conversations

def already_exists(cur, title, timestamp):
    cur.execute("SELECT 1 FROM memory WHERE title = %s AND timestamp = %s", (title, timestamp))
    return cur.fetchone() is not None

def store_entry(cur, entry):
    combined = (entry["user_text"] + " " + entry["ai_text"]).strip()
    if not combined:
        return

    embedding = get_embedding(combined)
    if embedding is None:
        return

    cur.execute("""
        INSERT INTO memory (title, timestamp, user_text, ai_text, embedding, tags, summary, source, length, importance, archived, openai_url)
        VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
    """, (
        entry["title"],
        entry["timestamp"],
        entry["user_text"],
        entry["ai_text"],
        embedding,
        entry["tags"],
        entry["summary"],
        entry["source"],
        len(combined),
        entry["importance"],
        entry["archived"],
        entry["openai_url"]
    ))

def import_from_html(path):
    conversations = parse_messages_html(path)
    conn = connect_db()
    cur = conn.cursor()

    inserted = 0
    for entry in conversations:
        if not already_exists(cur, entry["title"], entry["timestamp"]):
            store_entry(cur, entry)
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Imported {inserted} memory entries from HTML.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_openai_html.py path/to/messages.html")
    else:
        import_from_html(sys.argv[1])
