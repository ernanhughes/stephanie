# stephanie/utils/query_generator.py
class GoalQueryGenerator:
    def generate_queries(self, goal: dict, strategy: str) -> list[str]:
        desc = goal.get("goal_text", goal.get("description", ""))
        kw = goal.get("keywords", [])

        if strategy == "deep_literature":
            return kw + [desc + " site:arxiv.org"]
        elif strategy == "dataset-first":
            return kw + [f"Hugging Face dataset {desc}"]
        elif strategy == "code-centric":
            return kw + [f"GitHub {desc}"]
        else:
            return kw or [desc]
