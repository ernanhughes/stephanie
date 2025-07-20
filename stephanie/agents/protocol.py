from stephanie.agents.base_agent import BaseAgent
from stephanie.protocols import protocol_registry


class ProtocolAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        # Implement agent logic here


        # Get a protocol by name
        da_proto = protocol_registry.get_protocol("direct_answer")

        # Run it dynamically
        context["result"] = {"goal": "Is the Earth flat?"}
        new_context = da_proto.run(context)

        print(new_context)
        # Output: {'answer': 'Yes', 'trace': ['answered directly'], 'score': 0.85}
        return context