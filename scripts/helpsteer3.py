from datasets import load_dataset
from datetime import datetime, timezone
import json

# Config
NUM_SAMPLES = 100
GOAL_TYPE = "stem"
PROMPT_TYPE = "one_shot"
OUTPUT_PREFIX = "helpsteer3_stem"

# Helper
def clean_text(t):
    return t.strip().replace("\n", " ").replace("  ", " ")

def build_entries(example, idx):
    goal_id = f"helpsteer3_stem_{idx}"
    run_id_a = f"{goal_id}_a"
    run_id_b = f"{goal_id}_b"
    created_at = datetime.now(timezone.utc)

    # Build goal
    goal = {
        "id": goal_id,
        "goal_text": clean_text(example["context"][0]["content"]),
        "goal_type": "stem",
        "focus_area": example.get("domain", "stem"),
        "source": "helpsteer3",
        "created_at": created_at
    }

    # Runs
    run_a = {
        "goal_id": goal_id,
        "run_id": run_id_a,
        "pipeline": ["strategy_a"],
        "created_at": created_at
    }
    run_b = {
        "goal_id": goal_id,
        "run_id": run_id_b,
        "pipeline": ["strategy_b"],
        "created_at": created_at
    }

    # Scores
    score_map = {
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "Equal": (0.5, 0.5)
    }
    preference = example.get("overall_preference", "Equal")
    score_a, score_b = score_map.get(preference, (0.5, 0.5))

    score_entries = [
        {
            "goal": goal["goal_text"],
            "hypothesis": clean_text(example["response1"]),
            "score": score_a,
            "score_type": "preference_label",
            "run_id": run_id_a,
            "evaluator_name": "helpsteer3",
            "created_at": created_at
        },
        {
            "goal": goal["goal_text"],
            "hypothesis": clean_text(example["response2"]),
            "score": score_b,
            "score_type": "preference_label",
            "run_id": run_id_b,
            "evaluator_name": "helpsteer3",
            "created_at": created_at
        }
    ]

    # Reflection Delta
    delta = {
        "goal_id": goal_id,
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "score_a": score_a,
        "score_b": score_b,
        "score_delta": round(score_a - score_b, 2),
        "pipeline_diff": {
            "only_in_a": ["strategy_a"],
            "only_in_b": ["strategy_b"]
        },
        "strategy_diff": True,
        "model_diff": False,
        "rationale_diff": None,
        "created_at": created_at
    }

    return goal, run_a, run_b, score_entries, delta

# Load dataset
ds = load_dataset("nvidia/HelpSteer3", split="train")

# Filter to STEM + one_shot
filtered = ds.filter(lambda x: x.get("domain").lower() == "stem")
print(f"Filtered {len(filtered)} samples")

# Output
with open(f"{OUTPUT_PREFIX}_goals.jsonl", "w") as fg, \
     open(f"{OUTPUT_PREFIX}_runs.jsonl", "w") as fr, \
     open(f"{OUTPUT_PREFIX}_scores.jsonl", "w") as fs, \
     open(f"{OUTPUT_PREFIX}_deltas.jsonl", "w") as fd:

    for idx, example in enumerate(filtered.select(range(min(NUM_SAMPLES, len(filtered))))):
        goal, run_a, run_b, scores, delta = build_entries(example, idx)
        fg.write(json.dumps(goal) + "\n")
        fr.write(json.dumps(run_a) + "\n")
        fr.write(json.dumps(run_b) + "\n")
        for s in scores:
            fs.write(json.dumps(s) + "\n")
        fd.write(json.dumps(delta) + "\n")

print(f"Wrote {NUM_SAMPLES} STEM + one_shot examples to JSONL files.")
