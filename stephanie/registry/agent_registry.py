# stephanie/registry/agent_registry.py

from stephanie.agents.automind import AutoMindAgent
from stephanie.agents.self_aware_planner import SelfAwarePlannerAgent


class AgentRegistry:
    def __init__(self, config):
        self.config = config
        self.agents = {
            "automind": lambda: AutoMindAgent(config.agents.automind),
            "selfaware": lambda: SelfAwarePlannerAgent(config.agents.selfaware),
            # Add more agents here as needed
        }

    def get(self, name):
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self.agents[name]()
