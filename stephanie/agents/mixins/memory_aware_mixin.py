# stephanie/agents/mixins/memory_aware_mixin.py

from dataclasses import dataclass
from typing import Dict, List, Optional

from stephanie.utils.hashing import hash_dict


@dataclass
class MemoryFilterContext:
    """
    Encapsulates dynamic memory filtering rules derived from symbolic rules or configs.
    """
    tags: Optional[List[str]] = None
    dimension: Optional[str] = None
    skill: Optional[str] = None
    top_k: int = 3
    threshold: float = 0.5


class MemoryAwareMixin:
    """
    Mixin that equips any agent with memory-aware capabilities:
    - Inject shared memory into prompt context
    - Use episodic memory (vector DB) for retrieval
    - Support symbolic rule-based memory filtering
    - Auto-log memory usage events
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_memory_key = "shared_memory"
        self.episodic_memory_key = "episodic_memory"
        self.seen_hashes = set()

    def inject_memory_context(self, goal: str, context: dict, **overrides) -> dict:
        """
        Injects both shared and episodic memory into the prompt context.

        Supports symbolic rule overrides like:
            memory_tags=["agent=solver"]
            memory_dimension="clarity"
            memory_skill="multi-hop"
        """
        # Parse filter context from symbolic rules or config overrides
        filter_ctx = self._build_filter_context(context.get("symbolic_rules", {}), **overrides)

        # Retrieve from shared memory (short-term)
        shared_traces = self._get_shared_memory_traces(filter_ctx)

        # Retrieve from episodic memory (long-term)
        episodic_traces = self._get_episodic_memory_traces(goal, filter_ctx, context=context)

        # Deduplicate by content hash
        filtered_shared = [t for t in shared_traces if self._hash_trace(t) not in self.seen_hashes]
        filtered_episodic = [t for t in episodic_traces if self._hash_trace(t) not in self.seen_hashes]

        # Log filter usage
        self._log_memory_filter_used(filter_ctx)

        # Update seen hashes
        for trace in filtered_shared + filtered_episodic:
            self.seen_hashes.add(self._hash_trace(trace))

        # Inject into context
        updated_context = {
            **context,
            "memory": {
                "shared": filtered_shared,
                "episodic": filtered_episodic,
                "filter": filter_ctx.__dict__
            },
            "goal": goal
        }

        return updated_context

    def add_to_shared_memory(self, state: dict, entry: dict) -> None:
        """
        Add a new entry to the shared memory store.

        Entry should include at least:
            - agent name
            - trace / reasoning steps
            - response
            - score
        """
        shared_memory = state.get(self.shared_memory_key, [])
        shared_memory.append(entry)
        shared_memory.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        state[self.shared_memory_key] = shared_memory

    def _build_filter_context(self, symbolic_rules: dict, **overrides) -> MemoryFilterContext:
        """
        Build a memory filter context using symbolic rules and manual overrides.
        """
        ctx = MemoryFilterContext(**overrides)

        # Apply symbolic rule overrides
        if symbolic_rules:
            if "memory_tags" in symbolic_rules:
                ctx.tags = symbolic_rules["memory_tags"]
            if "memory_dimension" in symbolic_rules:
                ctx.dimension = symbolic_rules["memory_dimension"]
            if "memory_skill" in symbolic_rules:
                ctx.skill = symbolic_rules["memory_skill"]
            if "memory_top_k" in symbolic_rules:
                ctx.top_k = symbolic_rules["memory_top_k"]
            if "memory_threshold" in symbolic_rules:
                ctx.threshold = symbolic_rules["memory_threshold"]

        return ctx

    def _get_shared_memory_traces(self, filter_ctx: MemoryFilterContext) -> List[Dict]:
        """
        Get filtered traces from shared memory.
        """
        shared_memory = getattr(self, self.shared_memory_key, [])
        return self._filter_traces(shared_memory, filter_ctx)

    def _get_episodic_memory_traces(self, goal: str, filter_ctx: MemoryFilterContext, context: dict) -> List[Dict]:
        """
        Retrieve similar traces from episodic memory based on current goal.
        """
        memory_obj = context.get("memory", {}).get("episodic")

        if not memory_obj:
            return []

        try:
            results = memory_obj.retrieve_similar(goal, k=filter_ctx.top_k)
            return self._filter_traces(results, filter_ctx)
        except Exception as e:
            if self.logger:
                self.logger.log("EpisodicMemoryRetrievalError", {"error": str(e)})
            return []

    def _filter_traces(self, traces: List[Dict], filter_ctx: MemoryFilterContext) -> List[Dict]:
        """
        Filter traces by tags, dimension, skill, and score threshold.
        """
        if not traces:
            return []

        filtered = traces

        # Filter by tags
        if filter_ctx.tags:
            filtered = [
                t for t in filtered
                if any(tag in t.get("tags", []) for tag in filter_ctx.tags)
            ]

        # Filter by dimension score
        if filter_ctx.dimension:
            filtered = [
                t for t in filtered
                if t.get("dimension_scores", {}).get(filter_ctx.dimension, 0.0) >= filter_ctx.threshold
            ]

        # Sort by dimension score if specified
        if filter_ctx.dimension:
            filtered.sort(
                key=lambda x: x.get("dimension_scores", {}).get(filter_ctx.dimension, 0.0),
                reverse=True
            )

        # Take top-K
        return filtered[:filter_ctx.top_k]

    def _hash_trace(self, context: dict) -> str:
        """
        Generate a unique hash for a trace to prevent duplication.
        """
        content = context.get("trace", "") + context.get("response", "")
        return hash_dict({"content": content})

    def _log_memory_filter_used(self, filter_ctx: MemoryFilterContext) -> None:
        """
        Log which memory filters were applied during this run.
        """
        if self.logger:
            self.logger.log("MemoryFilterUsed", {
                "event": "memory_filter",
                "memory_dimension": filter_ctx.dimension,
                "memory_tags": filter_ctx.tags,
                "memory_skill": filter_ctx.skill,
                "memory_top_k": filter_ctx.top_k,
                "memory_threshold": filter_ctx.threshold
            })