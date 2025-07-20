# stephanie/agents/recovery_parser.py
import json

from stephanie.agents.base_agent import BaseAgent


class RecoveryParserAgent(BaseAgent):
    def __init__(self, cfg=None, memory=None, logger=None):
        super().__init__(cfg or {}, memory, logger)

    def parse(
        self, raw_text: str, expected_fields: list[str], regex_hint: str = None
    ) -> dict:
        prompt = self.prompt_loader.load_prompt(
            self.cfg,
            {
                "raw_text": raw_text,
                "expected_fields": expected_fields,
                "regex_hint": regex_hint or "None",
            },
        )
        output = self.call_llm(prompt, {})
        try:
            return json.loads(output)
        except Exception as e:
            self.logger and self.logger.log(
                "LLMParseFail",
                {
                    "error": str(e),
                    "raw_output": output,
                },
            )
            return {}
