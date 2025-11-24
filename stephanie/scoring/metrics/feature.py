class Feature:
    name: str

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

    async def apply(self, scorable, context):
        raise NotImplementedError
