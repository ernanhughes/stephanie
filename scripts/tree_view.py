import os
import sys

MODEL_TYPE_ICONS = {
    "svm": "📊",  # Changed to better reflect SVM/Scorable Vector Model
    "mrq": "🧠",
    "ebt": "🪜"
}

def get_icon(name, is_dir):
    """Return an emoji icon based on the filename or extension."""
    name = name.lower()
    if is_dir:
        return "📁"
    elif "encoder.pt" in name:
        return "🧠"
    elif "model.pt" in name:
        return "🤖"
    elif "tuner.json" in name:
        return "🎚️"
    elif "_scaler.joblib" in name:
        return "📏"
    elif name.endswith("meta.json"):
        return "⚙️"
    elif name.endswith((".yaml", ".yml")):
        return "📄"
    elif name.endswith(".md"):
        return "📘"
    elif name.endswith(".json"):
        return "🗂️"
    elif name.endswith(".pt"):
        return "📦"
    elif name.endswith(".joblib") or name.endswith(".pkl"):
        return "📦"
    elif name.endswith(".onnx"):
        return "📐"
    elif name.endswith(".txt"):
        return "📝"
    elif name.endswith(".py"):
        return "🐍"
    else:
        return "📦"

def get_model_type_icon(name):
    """Return a model type icon if the folder matches a known model type."""
    # Use the local dictionary, falling back to directory icon if not found
    return MODEL_TYPE_ICONS.get(name.lower(), "📁")

def print_tree(root_path, file_obj=sys.stdout, indent="", is_top_level=True):
    """
    Recursively print the directory tree to a given file object with icons.
    Defaults to printing to stdout.
    """
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        print(f"{indent}└── 🔒 Permission Denied", file=file_obj)
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        # Skip hidden files and special directory names
        if entry.startswith(".") or entry.startswith("__"):
            continue
        
        is_dir = os.path.isdir(path)
        # Determine connector style
        connector = "└──" if i == len(entries) - 1 else "├──"

        # Logic for determining the icon
        # Only use model-type icon if the parent is the designated root for model types
        if is_top_level and is_dir:
            icon = get_model_type_icon(entry)
        else:
            icon = get_icon(entry, is_dir)

        print(f"{indent}{connector} {icon}  {entry}", file=file_obj)

        if is_dir:
            extension = "    " if i == len(entries) - 1 else "│   "
            # When recursing, the next level is no longer "top-level" for model-type check
            print_tree(path, file_obj, indent + extension, is_top_level=False)

if __name__ == "__main__":
    output_filename = "file_view.log"
    
    # 1. Check for command-line argument
    if len(sys.argv) < 2:
        print("❌ Error: Please provide the base directory path as a command-line argument.", file=sys.stderr)
        print("Usage: python script_name.py <path/to/directory>", file=sys.stderr)
        sys.exit(1)

    # The first argument (index 0) is the script name itself; index 1 is the path.
    base_dir = sys.argv[1]
    
    # 2. Validate the directory path
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory '{base_dir}' does not exist.", file=sys.stderr)
    else:
        # 3. Process and write the tree
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                # Use os.path.basename() to get a clean name for the root display
                root_name = os.path.basename(base_dir.rstrip('/')) if base_dir.rstrip('/') else base_dir
                print(f"📦 {root_name}", file=f)
                print_tree(base_dir, file_obj=f)
            print(f"✅ Directory tree successfully written to '{output_filename}' from root: '{base_dir}'")
        except Exception as e:
            sys.exit(1)
