from haystack import component

@component
class GenerationComponent:
    def run(self, goal: str) -> dict:
        print(f"[GenerationComponent] Generating hypotheses for: {goal}")
        return {
            "hypotheses": [
                "Tesla will outperform due to strong Model Y sales.",
                "Tesla's Q4 profit will be impacted by raw material costs."
            ]
        }
