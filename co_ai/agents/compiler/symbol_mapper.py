# co_ai/compiler/symbol_mapper.py
from co_ai.agents.compiler.reasoning_trace import ReasoningNode
from co_ai.rules.symbolic_rule_applier import SymbolicRuleApplier


class SymbolMapper:
    def __init__(self, cfg, memory, logger):
        self.rule_engine = SymbolicRuleApplier(cfg, memory, logger)

    def tag_node(self, node: ReasoningNode) -> dict:
        tags = self.rule_engine.apply(node.thought)
        node.metadata["tags"] = tags
        return tags