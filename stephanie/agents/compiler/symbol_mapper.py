# stephanie/agents/compiler/symbol_mapper.py
from stephanie.agents.compiler.reasoning_trace import ReasoningNode
from stephanie.rules.symbolic_rule_applier import SymbolicRuleApplier


class SymbolMapper:
    def __init__(self, cfg, memory, logger):
        self.rule_engine = SymbolicRuleApplier(cfg, memory, logger)

    def tag_node(self, node: ReasoningNode) -> dict:
        tags = self.rule_engine.apply(node.thought)
        node.metadata["tags"] = tags
        return tags
