class ELMOrchestrator:

    def __init__(
        self,
        core_evaluator,         # Stephanie's main evaluator
        governance_reducer,     # ELM reducer
        reflection_engine
    ):
        self.core = core_evaluator
        self.governance = governance_reducer
        self.reflector = reflection_engine

    def step(self, context_pack, plan_trace, model_output):

        # 1. Core evaluation
        base_bundle = self.core.evaluate(
            context_pack=context_pack,
            plan_trace=plan_trace,
            output=model_output
        )

        # 2. Governance layer
        governed_bundle = self.governance.evaluate(
            context_pack=context_pack,
            plan_trace=plan_trace,
            output=model_output,
            base_bundle=base_bundle
        )

        # 3. Reflection trigger
        if governed_bundle.reward_vector.failure_signatures:
            return self.reflector.generate(governed_bundle)

        return governed_bundle
