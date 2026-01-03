# stephanie/agents/symbolic_tuner.py
from __future__ import annotations

import json
from collections import defaultdict

from stephanie.orm import NodeORM


class SymbolicTunerAgent:
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

    def run(self):
        # Group metrics by symbolic config
        results = defaultdict(list)

        nodes = self.memory.session.query(NodeORM).all()
        for node in nodes:
            if not node.valid or node.metric is None:
                continue

            config_str = json.dumps(node.config, sort_keys=True)
            results[config_str].append(node.metric)

        symbolic_rules = []
        for config_str, metrics in results.items():
            avg_metric = sum(metrics) / len(metrics)
            rule = {
                "config": json.loads(config_str),
                "mean_score": round(avg_metric, 4),
                "count": len(metrics),
            }
            symbolic_rules.append(rule)

        symbolic_rules.sort(key=lambda r: r["mean_score"], reverse=True)
        return symbolic_rules[:10]  # Return top 10 symbolic patterns


# Example usage:
# tuner = SymbolicTunerAgent()
# rules = tuner.run()
# for rule in rules:
#     print(rule)
