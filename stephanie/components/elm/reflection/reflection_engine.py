# components/elm/reflection/reflection_engine.py

from typing import List
from stephanie.data.score_bundle import ScoreBundle
from components.elm.reflection.reflection_trace import ReflectionTrace


class ReflectionEngine:

    def __init__(self, energy_threshold: float = 55.0, hrm_threshold: float = 60.0):
        self.energy_threshold = energy_threshold
        self.hrm_threshold = hrm_threshold

    def generate_reflection(self, bundle: ScoreBundle) -> ReflectionTrace:

        failed_axes: List[str] = []
        instructions = {}
        focus = []

        # --- Hallucination Energy ---
        energy = bundle.get("hallucination_energy")
        if energy and energy.score > self.energy_threshold:
            failed_axes.append("hallucination_energy")
            focus.append("hallucination_energy")
            instructions["grounding"] = (
                "Re-evaluate claims against retrieved context. "
                "Remove speculative statements not supported by evidence."
            )

        # --- HRM Alignment ---
        hrm = bundle.get("hrm_alignment")
        if hrm and hrm.score < self.hrm_threshold:
            failed_axes.append("hrm_alignment")
            focus.append("hrm_alignment")
            instructions["reasoning"] = (
                "Clarify logical steps. Make reasoning explicit. "
                "Avoid implicit assumptions."
            )

        # --- Embedding Margin ---
        margin = bundle.get("embedding_margin")
        if margin and margin.score < 50.0:
            failed_axes.append("embedding_margin")
            focus.append("embedding_margin")
            instructions["alignment"] = (
                "Align output terminology with retrieved context anchors."
            )

        confidence = 1.0 if failed_axes else 0.0

        return ReflectionTrace(
            original_trace_id=bundle.meta.get("trace_id", "unknown"),
            failed_axes=failed_axes,
            correction_instructions=instructions,
            focus_axes=focus,
            confidence=confidence,
        )
