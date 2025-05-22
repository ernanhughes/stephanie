import os

def rename_txt_to_j2(directory):
    """
    Recursively renames all .txt files to .j2 in the specified directory and its subdirectories.
    """
    renamed_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                old_path = os.path.join(root, filename)
                new_filename = filename[:-4] + ".j2"
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                renamed_files.append((old_path, new_path))
                print(f"Renamed: {old_path} -> {new_path}")
    return renamed_files

# Example usage:
# Replace '/path/to/prompts' with the actual path where your .txt files are stored.
if __name__ == "__main__":
    prompt_directory = "prompts"  # <-- CHANGE THIS
    renamed = rename_txt_to_j2(prompt_directory)
    print(f"\nTotal files renamed: {len(renamed)}")
