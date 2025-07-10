# stephanie/core/knowledge_cartridge.py

import json
from datetime import datetime
from hashlib import sha256

from stephanie.agents.knowledge import KnowledgePackage


class KnowledgeCartridge(KnowledgePackage):
    def __init__(self, goal, generation=0, parent_hash=None):
        super().__init__(goal=goal)
        self.generation = generation
        self.created_at = datetime.utcnow().isoformat()
        self.parent_hash = parent_hash
        self.quality_metrics = {}
        self.signature = None
        self._update_signature()
        self.schema["icl_examples"] = []  # Store prompt/response pairs

    def add_icl_example(self, prompt, response, task_type):
        """Add in-context learning example"""
        self.schema["icl_examples"].append({
            "prompt": prompt,
            "response": response,
            "task_type": task_type,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_finding(self, category, content, source, confidence, metadata=None):
        finding = {
            "id": f"{category}_{len(self.schema[category])}",
            "content": content,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.schema[category].append(finding)
        self._update_signature()

    def add_validation_protocol(self, hypothesis_id, protocol):
        self.schema["validation_protocols"][hypothesis_id] = protocol
        self._update_signature()

    def _update_signature(self):
        content_str = json.dumps(self.schema, sort_keys=True)
        self.signature = sha256(content_str.encode()).hexdigest()

    def to_json(self, include_signature=True):
        data = {
            "metadata": {
                "goal": self.goal,
                "generation": self.generation,
                "created_at": self.created_at,
                "parent_hash": self.parent_hash,
                "quality_metrics": self.quality_metrics,
            },
            "schema": self.schema,
        }
        if include_signature:
            data["signature"] = self.signature
        return json.dumps(data, indent=2)

    def to_markdown(self):
        # Replace this with a clean formatter later
        return f"# Report: {self.goal}\n\n## Thesis\n{self.schema['core_thesis']}\n\n..."
