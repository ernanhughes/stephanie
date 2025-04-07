import json
import sys
from datetime import datetime
from embedding import get_embedding
from db import connect_db

def parse_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for conv in data:
        messages = conv.get("mapping", {})
        user_texts = []
        ai_texts = []

        for m in messages.values():
            print(m)

            msg = m.get("message")
            if not msg:
                continue  # Skip invalid/null message entries
            author = msg.get("author", {})
            if not author:
                continue        
            content = msg.get("content", {})
            if not content: 
                continue

            role = m.get("message", {}).get("author", {}).get("role")
            content = m.get("message", {}).get("content", {}).get("parts", [])
            text = " ".join(str(content)) if content else ""

            if role == "user":
                user_texts.append(text.strip())
            elif role == "assistant":
                ai_texts.append(text.strip())

        if user_texts or ai_texts:
            conversations.append({
                "title": conv.get("title") or "ChatGPT Conversation",
                "timestamp": datetime.utcfromtimestamp(conv["create_time"]).isoformat() if conv.get("create_time") else datetime.utcnow().isoformat(),
                "user_text": "\n".join(user_texts),
                "ai_text": "\n".join(ai_texts),
                "summary": "",
                "tags": ["openai", "json"],
                "source": "openai.json",
                "importance": 0,
                "archived": False,
                "openai_url": conv.get("id")
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

def import_from_json(path):
    conversations = parse_conversations(path)
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
    print(f"Imported {inserted} memory entries from JSON.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_openai_json.py path/to/conversations.json")
    else:
        import_from_json(sys.argv[1])
