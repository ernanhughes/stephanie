import json
import ollama
from datetime import datetime, timezone
from app.db_utils import connect_db


def get_embedding(text):
    try:
        result = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return result["embedding"]
    except Exception as e:
        print("Embedding error:", e)
        return None

def already_exists(cur, title, timestamp):
    print(f"timestamp: {timestamp}")
    cur.execute("""
        SELECT 1 FROM memory WHERE title = %s AND timestamp = %s
    """, (title, timestamp))
    return cur.fetchone() is not None

def store_entry(cur, entry):
    user_text = entry.get("user_text", "") or ""
    ai_text = entry.get("ai_text", "") or ""
    combined_text = f"{user_text} {ai_text}".strip()
    if not combined_text:
        print(f"⚠️ Skipping empty entry: {entry.get('title', 'Untitled')}")
        return
    embedding = get_embedding(combined_text)
    if embedding is None:
        return

    timestamp = get_timestamp(entry)

    print(f"timestamp:{timestamp}")

    cur.execute("""
        INSERT INTO memory (title, timestamp, user_text, ai_text, embedding, tags, summary, source, length, importance, archived, openai_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        entry.get("title", "Untitled"),
        entry.get("timestamp", timestamp),
        entry.get("user_text", ""),
        entry.get("ai_text", ""),
        embedding,
        entry.get("tags", []),
        entry.get("summary", ""),
        entry.get("source", "manual"),
        entry.get("length", len(entry.get("user_text", "")) + len(entry.get("ai_text", ""))),
        entry.get("importance", 0),
        entry.get("archived", False),
        entry.get("openai_url", None)
    ))

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = connect_db()
    cur = conn.cursor()

    inserted = 0
    for entry in data:
        title = entry.get("title", "Untitled")
        timestamp = get_timestamp(entry)
        if not already_exists(cur, title, timestamp):
            store_entry(cur, entry)
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Done. {inserted} new memories added.")


def get_timestamp(entry):
    """
    Adds a 'timestamp' field to the given entry based on 'update_time' or 'create_time'.
    If neither is present, uses the current UTC time.

    Args:
        entry (dict): A dictionary containing optional 'update_time' or 'create_time' keys.

    Returns:
        None: The function modifies the input dictionary in place.
    """
    def convert_to_iso(timestamp):
        """Converts a Unix timestamp to an ISO 8601 string in UTC."""
        try:
            # Ensure the timestamp is treated as UTC
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        except (TypeError, ValueError):
            # Handle invalid or missing timestamps gracefully
            return None

    result = datetime.now(timezone.utc).isoformat()
    # Try to get the timestamp from 'update_time' or 'create_time'
    if "update_time" in entry and entry["update_time"]:
        result = convert_to_iso(entry["update_time"])
    elif "create_time" in entry and entry["create_time"]:
        result = convert_to_iso(entry["create_time"])
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python embed_and_store.py chat_dump.json")
    else:
        process_file(sys.argv[1])