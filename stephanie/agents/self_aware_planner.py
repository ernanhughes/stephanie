# --- agents/self_aware_planner_agent.py ---

import copy

from stephanie.agents.automind import AutoMindAgent
from stephanie.agents.symbolic_tuner import SymbolicTunerAgent
from stephanie.memory.symbolic_rule_store import SymbolicRuleStore


class SelfAwarePlannerAgent:
    def __init__(self, base_config):
        self.base_config = base_config
        self.rule_store = SymbolicRuleStore()

    def run(self, goal):
        # Step 1: Apply best symbolic rule to config
        config = copy.deepcopy(self.base_config)
        presets = self.rule_store.to_yaml_presets(top_n=1)
        if presets:
            config.update(presets[0])

        # Step 2: Execute AutoMindAgent with current config
        agent = AutoMindAgent(config)
        tree = agent.run(goal)

        # Step 3: Update symbolic knowledge from experience
        tuner = SymbolicTunerAgent()
        rules = tuner.run()
        self.rule_store.update_from_tuner(rules)

        return tree
