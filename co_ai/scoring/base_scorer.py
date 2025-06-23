class BaseScorer:
    """
    Base interface for any scorer that evaluates a hypothesis given a goal and dimensions.

    Returns:
        A dictionary with dimension names as keys, and for each:
            - score (float)
            - rationale (str)
            - weight (float, optional)
    """
    def score(self, goal: dict, hypothesis: dict, dimensions: list[str]) -> dict:
        raise NotImplementedError("Subclasses must implement the score method.")
