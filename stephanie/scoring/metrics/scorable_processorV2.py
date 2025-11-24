
class ScorableProcessorV2:

    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Load enabled features
        self.features = self._load_features()

    def _load_features(self):
        feats = []
        for name, fcfg in self.cfg.get("features", {}).items():
            if not fcfg.get("enabled", False):
                continue
            cls = locate(fcfg["cls"])   # import dynamically
            feats.append(cls(fcfg, self.memory, self.container, self.logger))
        return feats

    async def process_many(self, scorables, context):
        for sc in scorables:
            await self.process_one(sc, context)
        return scorables

    async def process_one(self, scorable, context):
        for feat in self.features:
            await feat.apply(scorable, context)
        return scorable
