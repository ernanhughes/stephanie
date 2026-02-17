from stephanie.components.elm.reflection.reflection_trace import ReflectionTrace


class ReflectionApplier:

    def apply_reflection(
        self,
        original_output: str,
        reflection: ReflectionTrace,
        model,
        context_pack,
    ) -> str:
        """
        Applies structured reflection by re-running model
        with corrective constraints.
        """

        if not reflection.failed_axes:
            return original_output

        correction_prompt = (
            "You previously produced the following output:\n\n"
            f"{original_output}\n\n"
            "It had the following issues:\n"
        )

        for axis in reflection.failed_axes:
            correction_prompt += f"- {axis}\n"

        correction_prompt += "\nPlease revise the output to correct these issues.\n"

        return model.generate(
            context=context_pack,
            additional_constraints=correction_prompt,
        )
