# stephanie/components/information/agents/policy_evolver.py
from __future__ import annotations
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.memcube import MemCubeORM
from stephanie.models.idea import Idea
from stephanie.memory.memcube_store import MemCubeStore
from stephanie.memory.idea_store import IdeaStore
from stephanie.scoring.scorable import ScorableProcessor
from stephanie.services.knowledge_graph_service import KnowledgeGraphService
from stephanie.utils.date_utils import iso_now

log = logging.getLogger(__name__)

@dataclass
class PolicyUpdate:
    """Represents a proposed change to a system policy."""
    component: str
    parameter: str
    old_value: Any
    new_value: Any
    confidence: float  # 0.0–1.0
    reason: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = iso_now()


class PolicyEvolverAgent(BaseAgent):
    """
    Analyzes historical performance data to evolve system policies.

    Responsibilities:
    - Query high-performing artifacts (MemCubes, Ideas) from storage.
    - Identify patterns in inputs, prompts, models, and parameters.
    - Propose policy updates (e.g., increase temperature for idea generation).
    - Persist changes to config or emit events for dynamic adaptation.

    This closes the loop on Stephanie's self-improvement cycle.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger
    ) -> None:
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)

        self.memcube_store: MemCubeStore = self.memory.memcubes
        self.idea_store: IdeaStore = self.memory.ideas
        self.scorable_processor: ScorableProcessor = self.container.get("scorable_processor")
        self.kg: KnowledgeGraphService = self.container.get("knowledge_graph")

        # Configurable thresholds
        self.min_high_score = cfg.get("min_high_score", 0.8)
        self.lookback_days = cfg.get("lookback_days", 7)
        self.max_updates_per_run = cfg.get("max_updates", 5)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint: evolve policies based on recent success.

        Populates context["policy_updates"] with list of changes.
        """
        updates: List[PolicyUpdate] = []

        # 1. Find high-quality MemCubes (blogs that scored well)
        log.info("PolicyEvolver: fetching high-scoring MemCubes...")
        high_memcubes = await self._get_high_scoring_memcubes()
        log.info(f"Found {len(high_memcubes)} high-quality MemCubes")

        # 2. Extract patterns and propose improvements
        if high_memcubes:
            updates.extend(self._analyze_memcube_policies(high_memcubes))

        # 3. Find high-impact ideas (r_final > threshold)
        log.info("PolicyEvolver: fetching top-rated research ideas...")
        top_ideas = await self.idea_store.list_top_ideas(
            min_r_final=self.min_high_score,
            limit=50
        )
        log.info(f"Found {len(top_ideas)} high-potential ideas")

        if top_ideas:
            updates.extend(self._analyze_idea_generation_policies(top_ideas))

        # 4. Apply or suggest updates
        applied = await self._apply_policy_updates(updates[:self.max_updates_per_run])

        # Output results
        context["policy_updates"] = [u.__dict__ for u in applied]
        context["policy_update_count"] = len(applied)

        return context

    # ------------------------------------------------------------------
    # Data Retrieval
    # ------------------------------------------------------------------
    async def _get_high_scoring_memcubes(self) -> List[MemCubeORM]:
        """Fetch MemCubes with high quality scores from recent history."""
        cutoff_date = (datetime.utcnow() - timedelta(days=self.lookback_days)).isoformat()

        # Get all MemCubes created recently
        candidates = await self.memcube_store.get_recent_sincetime(cutoff_date)

        high_quality = []
        for cube in candidates:
            # Extract scores from extra_data (attached by InformationQualityPass)
            scores = cube.extra_data.get("scores", {})
            avg_score = self._safe_mean(scores.values())
            if avg_score >= self.min_high_score:
                high_quality.append(cube)

        return high_quality

    def _safe_mean(self, values: List[float]) -> float:
        vs = [v for v in values if isinstance(v, (int, float)) and 0 <= v <= 1]
        return sum(vs) / len(vs) if vs else 0.0

    # ------------------------------------------------------------------
    # Policy Analysis: MemCube → Blog Quality
    # ------------------------------------------------------------------
    def _analyze_memcube_policies(self, memcubes: List[MemCubeORM]) -> List[PolicyUpdate]:
        """Derive blog/post policies from successful outputs."""
        updates = []

        # Example: Did blogs using web search perform better?
        use_web_count = 0
        total = len(memcubes)
        for m in memcubes:
            source_profile = m.extra_data.get("source_profile", {})
            if source_profile.get("use_web", False):
                use_web_count += 1

        ratio = use_web_count / total
        if ratio > 0.8:
            # Suggest increasing reliance on web for similar topics
            updates.append(PolicyUpdate(
                component="BucketBuilder",
                parameter="default_source_profile.use_web",
                old_value=False,
                new_value=True,
                confidence=0.85,
                reason=f"80% of high-scoring blogs used web sources ({use_web_count}/{total})"
            ))

        # Example: Was higher temperature linked to more engaging writing?
        # You could correlate LLM settings stored in metadata with readability scores

        return updates

    # ------------------------------------------------------------------
    # Policy Analysis: Idea Generation
    # ------------------------------------------------------------------
    def _analyze_idea_generation_policies(self, ideas: List[Idea]) -> List[PolicyUpdate]:
        """Learn from high-r_final ideas to improve future ideation."""
        updates = []

        # Compute average novelty & feasibility of winning ideas
        novelties = [i.novelty_score for i in ideas if i.novelty_score is not None]
        feasibilities = [i.feasibility_score for i in ideas if i.feasibility_score is not None]

        avg_novelty = self._safe_mean(novelties)
        avg_feasibility = self._safe_mean(feasibilities)

        # If most good ideas are highly novel, suggest increasing temperature
        if avg_novelty > 0.75:
            updates.append(PolicyUpdate(
                component="IdeaGenerationHead",
                parameter="settings.temperature",
                old_value=0.7,
                new_value=0.9,
                confidence=0.8,
                reason=f"Top ideas had high novelty ({avg_novelty:.2f}) suggesting need for more exploration"
            ))

        # If feasible ideas dominate, consider relaxing filters slightly
        if avg_feasibility < 0.6:
            updates.append(PolicyUpdate(
                component="CreativeAssociationAgent",
                parameter="min_r_final",
                old_value=0.65,
                new_value=0.60,
                confidence=0.75,
                reason=f"Average feasibility low ({avg_feasibility:.2f}); allow more experimental ideas through"
            ))

        return updates

    # ------------------------------------------------------------------
    # Apply Updates (Simulated or Real)
    # ------------------------------------------------------------------
    async def _apply_policy_updates(self, updates: List[PolicyUpdate]) -> List[PolicyUpdate]:
        """
        In MVP: we only **record** suggested updates.
        A separate process is responsible for actually changing config.
        """
        applied = []

        for update in updates:
            log.info(
                f"PolicyEvolver: proposing update {update.component}.{update.parameter} "
                f"{update.old_value} → {update.new_value} [{update.confidence:.2f}]"
            )

            # Mark as "proposed", not applied
            update.reason += " | STATUS=PROPOSED"
            applied.append(update)

            # TODO: write to PolicyStore / MemCube for human/agent review

        return applied
