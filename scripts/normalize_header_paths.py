import os
import re

def normalize_python_headers(root_dir="stephanie"):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, ".").replace("\\", "/")
                correct_header = f"# {rel_path}\n"

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Fix existing header if needed
                if lines and lines[0].startswith("#"):
                    current = lines[0].strip().replace("\\", "/")
                    if current != correct_header.strip():
                        lines[0] = correct_header
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                else:
                    # No header exists — insert it
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(correct_header)
                        f.writelines(lines)

if __name__ == "__main__":
    normalize_python_headers("stephanie")
