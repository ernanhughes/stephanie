from dependency_injector.wiring import Provide, inject

from stephanie.protocols.base import Protocol


class G3PSSolverAgent:
    @inject
    def __init__(self, protocol: Protocol):
        self.protocol = protocol  # This will be a real Protocol instance

    async def run(self, context: dict) -> dict:
        # Now self.protocol is a real object with .run()
        result = self.protocol.run(context)
        return result