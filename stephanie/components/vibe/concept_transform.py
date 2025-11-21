# stephanie/vibe/vibe_task_example.py
from __future__ import annotations

from stephanie.components.vibe.concept_transform import (
    graph_permute_isomorphic, text_paraphrase, trace_reorder_steps)
from stephanie.components.vibe.vibe import VibeFeatureTask

task = VibeFeatureTask(
    base_state=base_snapshot,
    user_feature="Improve robustness of research explanations",
    f2p_tests=[...],
    p2p_tests=[...],
    rubrics=[writing_rubric],
    perturbations=[
        text_paraphrase(
            id="txt_para_1",
            description="Paraphrase the explanation with different phrasing.",
            intensity=0.6,
            style="technical",
            group_id="text_robustness_1",
        ),
        trace_reorder_steps(
            id="trace_reorder_1",
            description="Reorder non-critical reasoning steps in the PlanTrace.",
            intensity=0.5,
            group_id="reasoning_robustness_1",
        ),
        graph_permute_isomorphic(
            id="graph_perm_1",
            description="Rename nodes in the underlying Nexus graph (isomorphic).",
            intensity=0.7,
            group_id="graph_robustness_1",
        ),
    ],
    context_complexity=0.8,
    aggressiveness_profile={},
)
