# tools/chat_importer.py
import json
import glob
import os
from bs4 import BeautifulSoup

def import_conversations(memory, path: str):
    """
    Walks a directory, parses chat files, and imports them into memory as CaseBooks.
    
    Args:
        memory (MemoryTool): The memory tool instance for persistence.
        path (str): The path to the directory containing chat exports.
    """
    for fp in glob.glob(os.path.join(path, "*.html")):
        with open(fp, "r", encoding="utf-8") as f:
            turns = parse_html(f.read())
            if turns:
                bundle = normalize_turns(turns)
                cb = conversation_to_casebook(memory, bundle)
                print(f"Imported: {cb.name} from HTML")
    
    for fp in glob.glob(os.path.join(path, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            turns = parse_json(json.load(f))
            if turns:
                bundle = normalize_turns(turns)
                cb = conversation_to_casebook(memory, bundle)
                print(f"Imported: {cb.name} from JSON")

def parse_html(html_content: str):
    """Parses a ChatGPT-style HTML export."""
    soup = BeautifulSoup(html_content, "html.parser")
    turns = []
    # Simplified parsing logic; needs to be robust to handle different structures.
    for element in soup.select("div.markdown"):
        role = "user" if "user-role" in str(element.parent) else "assistant"
        text = element.get_text(separator="\n").strip()
        turns.append({"role": role, "text": text})
    return turns

def parse_json(json_data: dict):
    """Parses a JSON export from a chat assistant."""
    turns = []
    # Assumes a common format like an array of messages
    if "messages" in json_data:
        for msg in json_data["messages"]:
            role = msg.get("role")
            text = msg.get("content")
            if role and text:
                turns.append({"role": role, "text": text})
    return turns

def normalize_turns(turns: list):
    """Normalizes the text content of turns."""
    for turn in turns:
        text = turn["text"]
        # Basic normalization: strip PII, clean whitespace, etc.
        turn["text"] = text.strip()
    return turns

# Placeholder for the next function, to be developed
def conversation_to_casebook(memory, bundle: list):
    # This will be implemented in the next file.
    # It will create CaseBookORM, CaseORM, and CaseScorableORM instances.
    print(f"Would create CaseBook from bundle with {len(bundle)} turns.")
    return None 