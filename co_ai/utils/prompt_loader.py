# co_ai/utils/prompt_loader.py
import os
from pathlib import Path


def load_prompt_from_file(prompt_path: str, base_dir: str = "prompts") -> str:
    """
    Load a prompt template from a .txt file.
    
    Args:
        prompt_path: Relative path to the prompt file (e.g., "generation_hypothesis.txt")
        base_dir: Base directory where prompts are stored
    
    Returns:
        Loaded prompt as string
    """
    project_root = Path(__file__).parent.parent.parent  # go up to root
    full_path = project_root / base_dir / prompt_path

    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()