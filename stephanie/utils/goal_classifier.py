def classify_goal_strategy(goal: dict) -> str:
    desc = goal.get("goal_text", "").lower()
    kw = " ".join(goal.get("keywords", [])).lower()

    if "dataset" in desc or "huggingface" in kw:
        return "dataset-first"
    elif "accuracy" in desc or "evaluation" in kw:
        return "evaluation-centric"
    elif "explain" in desc or "interpret" in kw:
        return "interpretability"
    elif "code" in desc or "repo" in kw:
        return "code-centric"
    elif "survey" in desc or "literature" in kw:
        return "deep_literature"
    else:
        return "default"
