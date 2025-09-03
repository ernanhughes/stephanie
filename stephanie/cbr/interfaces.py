# stephanie/cbr/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

Context = Dict[str, Any]
ScorableLike = Dict[str, Any]  # {id, text, ...}

class ContextNamespacer(ABC):
    @abstractmethod
    def ns(self, ctx: Context) -> Context: ...
    @abstractmethod
    def variant_bucket(self, ctx: Context, variant: str) -> Context: ...
    @abstractmethod
    def variant_output_key(self, variant: str) -> str: ...
    @abstractmethod
    def temp_key(self, ctx: Context, key: str, value):  # contextmanager
        ...

class CasebookScopeManager(ABC):
    @abstractmethod
    def home_casebook_id(self, ctx: Context, agent_name: str, tag: str) -> int: ...
    @abstractmethod
    def ensure_scope(self, pipeline_run_id: Optional[str], agent: Optional[str], tag: str): ...
    @abstractmethod
    def get_cases(self, ctx: Context, retrieval_mode: str, tag: str) -> List[Any]: ...

class CaseSelector(ABC):
    @abstractmethod
    def build_reuse_candidates(self, casebook_id: int, goal_id: str, cases: List[Any],
                               budget: int, novelty_k: int, exploration_eps: float) -> List[str]: ...

class RankAndAnalyze(ABC):
    @abstractmethod
    def run(self, ctx: Context, hypotheses: List[ScorableLike]) -> Tuple[List[ScorableLike], Dict, Dict, List[str], Dict]:
        """returns (ranked, corpus, mars_results, recommendations, scores_payload)"""

class RetentionPolicy(ABC):
    @abstractmethod
    def retain(self, ctx: Context, ranked: List[ScorableLike], mars: Dict, scores: Dict) -> Optional[int]: ...

class QualityAssessor(ABC):
    @abstractmethod
    def quality(self, mars_results: Dict, scores_payload: Dict) -> float: ...

class ChampionPromoter(ABC):
    @abstractmethod
    def maybe_promote(self, casebook_id: int, goal_id: str, retained_case_id: Optional[int], quality: float) -> None: ...

class GoalStateTracker(ABC):
    @abstractmethod
    def bump_run_ix(self, casebook_id: int, goal_id: str) -> int: ...
    @abstractmethod
    def should_run_ab(self, run_ix: int, mode: str, period: int) -> bool: ...

class ABValidator(ABC):
    @abstractmethod
    async def run_two(self, ctx: Context, run_cbr, run_baseline) -> Tuple[str, Dict]:
        """Executes (baseline, cbr) variants, compares quality, returns (winner_variant, comparison_report)."""

class MicroLearner(ABC):
    @abstractmethod
    def learn(self, ctx: Context, ranked: List[ScorableLike], mars: Dict) -> None: ...
