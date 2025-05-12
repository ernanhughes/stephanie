import os
import re
from pathlib import Path


# Set base path
PROMPT_DIR = Path("prompts")
IGNORE_FILES = [".DS_Store", "__pycache__"]

AGENT_STRATEGIES = {
    "generation": ["goal_aligned", "out_of_the_box"],
    "evolution": [
        "feasibility", "grafting", "inspiration", "simplification",
        "self_critic"
    ],
    "reflection": [
        "full_review", "deep_verification", "initial_review",
        "novelty_check", "observation_review", "safety_check",
        "simulation_review"
    ],
    "ranking": ["debate"],
    "literature_ranking": ["ranking"],
    "literature_search": ["query", "parse"],
    "meta_review": [
        "expert_summary", "feedback_to_generation", "research_overview",
        "safety_check", "synthesis"
    ],
    "proximity": ["analysis"]
}

def organize_prompts():
    """Main function to organize and rename prompt files"""
    print(f"[INFO] Organizing prompts in {PROMPT_DIR}")

    # Create subdirs if not exist
    for agent in AGENT_STRATEGIES.keys():
        (PROMPT_DIR / agent).mkdir(exist_ok=True)

    # Loop through all .txt files
    for file in PROMPT_DIR.glob("*.txt"):
        if file.name in IGNORE_FILES:
            continue

        # Match agent_strategy.txt pattern
        match = re.match(r"([a-z]+)_([a-z_]+)\.txt", file.name)
        if not match:
            print(f"[ERROR] Could not parse {file.name}")
            continue

        agent_name = match.group(1)
        raw_strategy = match.group(2)

        # Validate agent name
        if agent_name not in AGENT_STRATEGIES:
            print(f"[ERROR] Unknown agent: {agent_name} in {file.name}")
            continue

        # Validate strategy
        valid_strategies = AGENT_STRATEGIES[agent_name]
        if raw_strategy not in valid_strategies:
            print(f"[WARNING] Strategy '{raw_strategy}' not in known list for {agent_name}, but will still move.")
        
        target_dir = PROMPT_DIR / agent_name
        new_name = f"{raw_strategy}.txt"
        target_path = target_dir / new_name

        # Move and rename
        try:
            file.rename(target_path)
            print(f"[MOVED] {file.name} → {target_path}")
        except Exception as e:
            print(f"[ERROR] Failed to move {file.name}: {e}")


if __name__ == "__main__":
    organize_prompts()