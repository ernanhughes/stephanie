# stephanie/agents/compiler/symbol_mapper.py
from __future__ import annotations

from stephanie.agents.compiler.reasoning_trace import ReasoningNode
from stephanie.services.rules_service import RulesService


class SymbolMapper:
    def __init__(self, cfg, memory, container, logger):
        self.rule_engine = RulesService(cfg, memory, container, logger)

    def tag_node(self, node: ReasoningNode) -> dict:
        tags = self.rule_engine.apply(node.thought)
        node.metadata["tags"] = tags
        return tags
