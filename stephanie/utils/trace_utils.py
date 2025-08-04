import json
import os

import numpy as np
import torch

from stephanie.data.plan_trace import PlanTrace


def get_trace_score_stats(trace: PlanTrace, dimensions: list[str]) -> torch.Tensor:
    sicql_q_values = []
    sicql_v_values = []
    ebt_energies = []
    ebt_uncertainties = []

    # Collect all bundles (execution steps + final)
    bundles = [step.scores for step in trace.execution_steps] + [trace.final_scores]

    for bundle in bundles:
        for dimension in dimensions:
            result = bundle.results.get(dimension)
            if result:
                # Use getattr with fallback to None
                q = getattr(result, "q_value", None)
                v = getattr(result, "state_value", None)
                e = getattr(result, "energy", None)
                u = getattr(result, "uncertainty", None)

                # Append only valid floats
                if q is not None: 
                    sicql_q_values.append(q)
                if v is not None: 
                    sicql_v_values.append(v)
                if e is not None: 
                    ebt_energies.append(e)
                if u is not None: 
                    ebt_uncertainties.append(u)

    def stats(values):
        valid = [v for v in values if v is not None]
        if not valid:
            return [0.0, 0.0, 0.0]
        return [
            float(np.mean(valid)),
            float(np.std(valid)),
            float(valid[-1]),
        ]

    # Final features vector: [q_stats, v_stats, energy_stats, uncertainty_stats]
    features = (
        stats(sicql_q_values) +
        stats(sicql_v_values) +
        stats(ebt_energies) +
        stats(ebt_uncertainties)
    )

    return torch.tensor(features, dtype=torch.float32)


def load_plan_traces_from_export_dir(export_dir: str) -> list[PlanTrace]:
    traces = []
    if not os.path.exists(export_dir):
        print(f"Export directory {export_dir} does not exist.")
        return traces
    for fname in os.listdir(export_dir):
        if fname.startswith("trace_") and fname.endswith(".json"):
            fpath = os.path.join(export_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    loaded_dict = json.load(f)
                trace = PlanTrace.from_dict(loaded_dict)
                traces.append(trace)
            except Exception as e:
                print(f"Error loading trace from {fname}: {e}")
    return traces
