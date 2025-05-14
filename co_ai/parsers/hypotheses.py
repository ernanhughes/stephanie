from co_ai.parsers.response_parser import ResponseParser

class HypothesisParser(ResponseParser):
    """
    Default parser for extracting hypotheses sections from model output.
    """

    def _default_pattern(self) -> str:
        return r"# Hypothesis\s+\d+\s*\n(?:.*?\n)*?(?=(# Hypothesis\s+\d+|\Z))"
