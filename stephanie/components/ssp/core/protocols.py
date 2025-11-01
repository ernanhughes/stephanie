# stephanie/components/ssp/core/protocols.py
"""
SSP Protocol Definitions

Shared data structures, type aliases, and protocols used across the SSP system.
These interfaces let you swap implementations (LLM vs MemCube, ATS vs mock, etc.)
without touching the algorithmic core.

Key components:
- EpisodeContext: Context propagated through an SSP episode
- EpisodeResult / VerificationResult / JudgeResult
- SSPMetrics (system-level KPIs)
- Core roles: Proposer, Retriever, Verifier, Solver, Judge
- Services: PromptService, ProgressReporter, EventSink, ArtifactStore, VPMEncoder
- Filters/Rewards/Curriculum protocols
- SSPAlgorithm (orchestrator façade)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Iterable,
    Sequence,
    Tuple,
    runtime_checkable,
)

# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------
EpisodeID = str
Question = str
Answer = str
Snippet = str
Score = float
Difficulty = float


# ---------------------------------------------------------------------
# Episode context and results
# ---------------------------------------------------------------------
class EpisodeContext(TypedDict, total=False):
    """Context passed through the SSP pipeline."""
    pipeline_run_id: str
    session_id: str
    user_id: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]
    # Optional knobs commonly needed
    max_tokens: int
    temperature: float
    top_p: float


@dataclass
class VerificationResult:
    """Result of the verification process."""
    is_valid: bool
    score: float
    reason: str
    filter_results: Dict[str, bool]
    verification_details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class JudgeResult:
    """Compact judge output used across verification/evaluation."""
    score: int                     # 0..100, integer for rubric parity
    rationale: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Complete result of an SSP episode."""
    episode_id: EpisodeID
    seed_answer: Answer
    question: Question
    predicted_answer: Answer
    evidence_docs: List[Snippet]
    proposer_evidence: List[Snippet]
    verified: bool
    verifier_score: float
    solver_steps: int
    difficulty: Difficulty
    proposer_meta: Dict[str, Any]
    verifier_meta: Dict[str, Any]
    solver_meta: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    episode_duration: float = 0.0


@dataclass
class SSPMetrics:
    """Performance metrics for the SSP system."""
    # Episode metrics
    total_episodes: int = 0
    verified_episodes: int = 0
    success_rate: float = 0.0

    # Proposer metrics
    proposer_success_rate: float = 0.0
    avg_question_difficulty: float = 0.0
    evidence_quality: float = 0.0

    # Solver metrics
    solver_accuracy: float = 0.0
    avg_solver_steps: float = 0.0
    evidence_coverage: float = 0.0

    # Verification metrics
    verification_pass_rate: float = 0.0
    avg_verification_score: float = 0.0

    # Self-play metrics
    proposer_adversarial_reward: float = 0.0
    solver_cooperative_reward: float = 0.0
    curriculum_difficulty: float = 0.0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------
# Core role protocols (paper-faithful)
# ---------------------------------------------------------------------
@runtime_checkable
class Proposer(Protocol):
    """
    Searching proposer: crafts a single question from a seed mechanism (answer),
    optionally collecting evidence O(τ) while proposing.
    Returns: (question, meta, evidence_snippets)
    """
    async def propose(self, seed_answer: Answer, context: EpisodeContext) -> Tuple[Question, Dict[str, Any], List[Snippet]]: ...


@runtime_checkable
class Retriever(Protocol):
    """
    Retrieval bridge. History-first, MemCube, web, or LLM-backed—your choice.
    """
    async def retrieve(self, query: str, seed_answer: Answer, context: EpisodeContext, k: int) -> List[Snippet]: ...


@runtime_checkable
class Verifier(Protocol):
    """
    RAG-gated verification (no-search solver): decide if question is valid and solvable
    using only proposer-provided evidence; implements rejection sampling.
    """
    async def verify(self, question: Question, seed_answer: Answer, evidence: Sequence[Snippet], context: EpisodeContext) -> VerificationResult: ...


@runtime_checkable
class Solver(Protocol):
    """
    Deep search solver (e.g., ATS) that can use full retrieval.
    Returns: (predicted_answer, evidence_snippets, steps, meta)
    """
    async def solve(self, question: Question, context: EpisodeContext) -> Tuple[Answer, List[Snippet], int, Dict[str, Any]]: ...


@runtime_checkable
class Judge(Protocol):
    """
    Knowledge judge used within verification/eval. Must obey the strict two-line format elsewhere,
    but here we return a structured object.
    """
    async def judge(self, goal_text: str, user_text: str, assistant_text: str, evidence: Sequence[Snippet], context: EpisodeContext) -> JudgeResult: ...


# ---------------------------------------------------------------------
# Services / infrastructure
# ---------------------------------------------------------------------
@runtime_checkable
class PromptService(Protocol):
    """
    LLM prompt runner used by proposer/verifier/retriever.
    """
    async def run_prompt(self, prompt: str, context: Dict[str, Any]) -> str: ...


@runtime_checkable
class ProgressReporter(Protocol):
    """
    Mirrors ProgressMixin methods so components can report stages/ticks.
    """
    def start(self, task: str, **kw: Any) -> None: ...
    def stage(self, task: str, stage: str, **kw: Any) -> None: ...
    def set(self, task: str, done: int, total: int, **kw: Any) -> None: ...
    def end(self, task: str, **kw: Any) -> None: ...


@runtime_checkable
class EventSink(Protocol):
    """
    Minimal event sink (TreeEventEmitter-compatible).
    """
    def emit(self, event: str, payload: Dict[str, Any]) -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """
    Episode/run artifact persistence (JSON, text, images).
    """
    def save_json(self, path: str, obj: Dict[str, Any]) -> str: ...
    def save_text(self, path: str, text: str) -> str: ...
    def save_image(self, path: str, data: bytes, *, format: Optional[str] = None) -> str: ...


@runtime_checkable
class VPMEncoder(Protocol):
    """
    Encodes per-episode metrics into a VPM artifact (PNG/JSON).
    Returns the artifact path.
    """
    def encode(self, episode_id: EpisodeID, node_scores: List[float], judge_score: int, evidence_snippets: List[str]) -> str: ...


# ---------------------------------------------------------------------
# Filters / curriculum / rewards
# ---------------------------------------------------------------------
@runtime_checkable
class RuleFilter(Protocol):
    """
    Rule-based pre-verification checks (format, leakage, min length, tool-use, etc.).
    Returns per-rule boolean flags.
    """
    def apply(self, question: Question, seed_answer: Answer, evidence: Sequence[Snippet], context: EpisodeContext) -> Dict[str, bool]: ...


class EpisodeTracker(Protocol):
    """Tracks episode state across the pipeline."""
    def start_episode(self, episode_id: EpisodeID, seed_answer: Answer) -> None: ...
    def update_episode(self, episode_id: EpisodeID, **kwargs: Any) -> None: ...
    def complete_episode(self, episode_id: EpisodeID, result: EpisodeResult) -> None: ...
    def get_episode(self, episode_id: EpisodeID) -> Optional[EpisodeResult]: ...


class RewardCalculator(Protocol):
    """Computes proposer/solver rewards over verified episodes."""
    def calculate_rewards(self, verified_episodes: List[EpisodeResult]) -> Dict[str, float]: ...
    def get_reward_history(self) -> List[Dict[str, float]]: ...


class CurriculumManager(Protocol):
    """Manages difficulty progression."""
    def update_curriculum(self, success: bool, difficulty: Difficulty) -> Difficulty: ...
    def get_current_difficulty(self) -> Difficulty: ...
    def reset_curriculum(self) -> None: ...


# ---------------------------------------------------------------------
# Algorithm façade
# ---------------------------------------------------------------------
@runtime_checkable
class SSPAlgorithm(Protocol):
    """
    High-level orchestrator interface so your agent/trainer can be swapped out.
    """
    async def run_episode(self, seed_answer: Answer, context: EpisodeContext) -> EpisodeResult: ...
    async def run_batch(self, seeds: Iterable[Answer], context: EpisodeContext, *, concurrency: int = 1) -> SSPMetrics: ...


