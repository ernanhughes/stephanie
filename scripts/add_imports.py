import os

TARGET_IMPORT = "from __future__ import annotations\n"

def ensure_future_annotations(root_dir="."):
    modified_files = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            filepath = os.path.join(subdir, file)
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Skip if already present
            if any("from __future__ import annotations" in line for line in lines):
                continue

            insert_at = 0
            # Skip shebang or encoding line if present
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            elif lines and "coding" in lines[0]:
                insert_at = 1

            lines.insert(insert_at, TARGET_IMPORT)

            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(lines)

            modified_files.append(filepath)

    # Log results
    if modified_files:
        print("✅ Added `from __future__ import annotations` to:")
        for path in modified_files:
            print("   -", path)
    else:
        print("No changes needed. All files already have the import.")

if __name__ == "__main__":
    # Change "." to your Stephanie root if running elsewhere
    ensure_future_annotations(".")
