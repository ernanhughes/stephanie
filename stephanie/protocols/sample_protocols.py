# stephanie/protocols/sample_protocols.py
from stephanie.protocols.base import Protocol


class DirectAnswerProtocol(Protocol):
    def run(self, input_context: dict) -> dict:
        return {"answer": "Yes", "trace": ["answered directly"], "score": 0.85}

class CodeExecutionProtocol(Protocol):
    def run(self, input_context: dict) -> dict:
        code = input_context.get("code")
        return {"result": eval(code), "trace": ["executed code"], "success": True}