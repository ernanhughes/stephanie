# stephanie/agents/data/vpm_builder_agent.py
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import CASEBOOK


class VPMBuilderAgent(BaseAgent):
    """
    Agent that builds a Vector of Policies and Measures (VPM)
    from a given CaseBook and its associated evaluation metrics.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.metrics = cfg.get("metrics", ["alignment", "clarity"])

    async def run(self, context: dict) -> dict:
        casebook = context.get(CASEBOOK)
        if not casebook:
            self.report({"event": "vpm_build_failed", "reason": "No casebook in context."})
            return context

        # Step 1: Fetch evaluations for all cases in the casebook
        case_ids = [c.id for c in casebook.cases]
        all_evaluations = self.memory.evaluations.get_for_cases(case_ids)
        
        # Step 2: Build the VPM matrix
        vpm_matrix = []
        for case_id in case_ids:
            case_metrics = {}
            for eval in all_evaluations:
                if eval.case_id == case_id and eval.attribute.name in self.metrics:
                    case_metrics[eval.attribute.name] = eval.score.value
            
            # Ensure all metrics are present for each case
            row = [case_metrics.get(metric, 0.0) for metric in self.metrics]
            vpm_matrix.append(row)

        vpm = np.array(vpm_matrix)

        # Step 3: Normalize and save
        vpm_normalized = (vpm - vpm.min()) / (vpm.max() - vpm.min()) if vpm.max() > vpm.min() else vpm
        vpm_path = f"vpm_data/{casebook.name}.npy"
        np.save(vpm_path, vpm_normalized)
        
        # Step 4: Create a PNG preview
        plt.figure(figsize=(10, 6))
        plt.imshow(vpm_normalized.T, aspect='auto', cmap='viridis')
        plt.title(f"VPM for CaseBook: {casebook.name}")
        plt.xlabel("Cases")
        plt.ylabel("Metrics")
        plt.yticks(ticks=np.arange(len(self.metrics)), labels=self.metrics)
        preview_path = f"vpm_previews/{casebook.name}.png"
        plt.savefig(preview_path)
        plt.close()
        
        # Step 5: Report outcome
        self.report({
            "event": "vpm_built",
            "casebook_id": casebook.id,
            "metrics": self.metrics,
            "vpm_shape": vpm.shape,
            "vpm_path": vpm_path,
            "preview_path": preview_path,
        })

        return context