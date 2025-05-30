from datasets import load_dataset
import json
from datetime import datetime, timezone

# Load StrategyQA dataset
dataset = load_dataset("ChilleD/StrategyQA", split="train")

# Convert to Co AI goal format
def convert_to_goal_format(example, idx):
    return {
        "id": f"strategyqa_{idx}",
        "goal_text": example["question"],
        "goal_type": "strategyqa",
        "focus_area": "commonsense",
        "source": "strategyqa",
        "answer": example["answer"],
        "facts": example.get("facts", []),
        "created_at": datetime.now(timezone.utc)
    }

# Output file path
output_path = "strategyqa_goals.jsonl"

# Convert and write
with open(output_path, "w", encoding="utf-8") as f:
    for idx, example in enumerate(dataset.select(range(100))):  # Limit to 100 goals
        goal = convert_to_goal_format(example, idx)
        f.write(json.dumps(goal) + "\n")

print(f"Saved {output_path}")
