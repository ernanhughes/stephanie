# stephanie/protocols/__init__.py
from stephanie.protocols.registry import ProtocolRegistry
from stephanie.protocols.sample_protocols import (CodeExecutionProtocol,
                                                  DirectAnswerProtocol)

protocol_registry = ProtocolRegistry()

protocol_registry.register(
    name="direct_answer",
    protocol=DirectAnswerProtocol(),
    description="Answers directly using LLM",
    input_format={"goal": "string"},
    output_format={"answer": "string"},
    failure_modes=["Incomplete", "Hallucinated"],
    tags=["qa", "llm"],
    capability="question_answering"
)

protocol_registry.register(
    name="code_exec",
    protocol=CodeExecutionProtocol(),
    description="Runs Python code snippets",
    input_format={"code": "string"},
    output_format={"result": "any"},
    failure_modes=["SyntaxError", "RuntimeError"],
    tags=["code", "execution"],
    capability="code_execution"
)