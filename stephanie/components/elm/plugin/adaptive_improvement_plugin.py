class AdaptiveImprovementPlugin:

    def __init__(self, evaluator, comparator, governor, reflection_engine, applier):
        self.evaluator = evaluator
        self.comparator = comparator
        self.governor = governor
        self.reflection_engine = reflection_engine
        self.applier = applier

    def improve(self, context, trace, output, model):

        before_bundle = self.evaluator.evaluate(context, trace, output)

        reflection_trace = self.reflection_engine.generate_reflection(before_bundle)

        improved_output = self.applier.apply_reflection(
            original_output=output,
            reflection=reflection_trace,
            model=model,
            context_pack=context,
        )

        after_bundle = self.evaluator.evaluate(context, trace, improved_output)

        if not self.governor.should_accept_update(before_bundle, after_bundle):
            return output, before_bundle

        if self.comparator.dominates(
            before_bundle,
            after_bundle,
            critical_axes=["hallucination_energy", "hrm_alignment"],
        ):
            return improved_output, after_bundle

        return output, before_bundle
