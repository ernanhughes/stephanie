import re

class ResponseParser:
    """
    Base class for parsing LLM responses into structured content.
    Allows regex-based extraction with configurable fallback.
    """

    def __init__(self, cfg: dict, logger=None):
        self.cfg = cfg
        self.logger = logger

    def parse(self, response: str) -> any:
        pattern = self._get_pattern()

        if pattern:
            try:
                matches = re.findall(pattern, response, flags=re.DOTALL)
                if matches:
                    return [m.strip() for m in matches]
            except re.error as e:
                if self.logger:
                    self.logger.log("ParserRegexError", {"error": str(e), "pattern": pattern})

        return self._fallback_parse(response)

    def _get_pattern(self) -> str:
        pattern = self.cfg.get("parse_pattern")
        if pattern is None:
            self.logger.log("ParserConfigError no parse_Patter found in config falling back to default")
            return self._default_pattern()
        return pattern

    def _default_pattern(self) -> str | None:
        return r"(?m)^\s*[\-\*\d]+\.\s+(.*)"  # Override in subclasses if needed

    def _fallback_parse(self, response: str) -> list[str]:
        return [response.strip()] if response.strip() else []
