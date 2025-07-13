import os

def annotate_python_files(root_dir="stephanie"):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                # Build correct relative POSIX-style path
                rel_path = os.path.relpath(file_path, ".").replace("\\", "/")
                full_comment = f"# {rel_path}\n"

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                if lines:
                    first_line = lines[0].strip().replace("\\", "/")
                    # If already correct, skip
                    if first_line == full_comment.strip():
                        continue
                    # If already a path-style comment, replace it
                    if first_line.startswith("#") and "stephanie/" in first_line:
                        lines[0] = full_comment
                    else:
                        lines.insert(0, full_comment)
                else:
                    lines = [full_comment]

                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

if __name__ == "__main__":
    annotate_python_files("stephanie")
