# sis/utils/model_tree.py
import os

MODEL_TYPE_ICONS = {
    "svm": "🌀",
    "mrq": "🧠",
    "ebt": "🪜"
}

def get_icon(name, is_dir):
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
    elif name.endswith(".log"):
        return "📋"
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
    return MODEL_TYPE_ICONS.get(name.lower(), "📦")

def build_tree(root_path, indent="", is_top_level=True):
    """Return a string tree for templates instead of printing."""
    lines = []
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        lines.append(indent + "└── 🔒 Permission Denied")
        return lines

    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        is_dir = os.path.isdir(path)
        connector = "└──" if i == len(entries) - 1 else "├──"

        if is_top_level and is_dir:
            icon = get_model_type_icon(entry)
        else:
            icon = get_icon(entry, is_dir)

        lines.append(f"{indent}{connector} {icon}  {entry}")

        if is_dir:
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(build_tree(path, indent + extension, is_top_level=False))
    return lines

def get_model_tree(base_dir="models"):
    if not os.path.exists(base_dir):
        return [f"❌ Directory '{base_dir}' does not exist."]
    return [f"📦 {base_dir}"] + build_tree(base_dir)
