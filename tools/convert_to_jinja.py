import os
import re


def convert_to_jinja_format(content: str) -> str:
    """
    Converts Python-style {key} formatting to Jinja-style {{ key }}.
    Only affects keys not already wrapped in double braces.
    """
    return re.sub(r'(?<!{){(\w+)}(?!})', r'{{ \1 }}', content)


def convert_templates_to_jinja(base_dir: str, file_extensions=None):
    """
    Recursively converts all matching files in a directory to Jinja-style placeholders.

    Args:
        base_dir: root directory to scan
        file_extensions: set of file extensions to include
    """
    if file_extensions is None:
        file_extensions = {".txt", ".prompt", ".jinja"}
    converted_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext.lower() in file_extensions:
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        original = f.read()

                    converted = convert_to_jinja_format(original)

                    if converted != original:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(converted)
                        converted_files.append(path)
                        print(f"✅ Converted: {path}")
                    else:
                        print(f"⏭️  Skipped (no change): {path}")

                except Exception as e:
                    print(f"❌ Failed to process {path}: {e}")

    return converted_files


if __name__ == "__main__":
# Change this to your local projectI
    convert_templates_to_jinja("/prompts", ".txt")
