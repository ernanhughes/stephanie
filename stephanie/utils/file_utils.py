import re


def camel_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def get_text_from_file(file_path: str) -> str:
    """Get text from a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def write_text_to_file(path: str, text: str):
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"✅ Successfully wrote to {path}")
    except Exception as e:
        print(f"❌ Failed to write to {path}: {e}")
