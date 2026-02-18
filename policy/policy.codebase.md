# Project Context: policy
# Path: C:\Users\ernan\Project\stephanie\policy
# Generated for AI Review


==================================================
FILE: base_policy.py
==================================================

from .governance_signal import GovernanceSignal
from .dominance import DominanceEngine
from .calibration import EnergyRegimeEstimator
from .monitor import EnergySpiralDetector
from .decision import Decision


class BasePolicy:

    def __init__(
        self,
        dominance_engine: DominanceEngine,
        regime_estimator: EnergyRegimeEstimator,
        spiral_detector: EnergySpiralDetector,
    ):
        self.dominance_engine = dominance_engine
        self.regime_estimator = regime_estimator
        self.spiral_detector = spiral_detector

    def evaluate(
        self,
        current_signal: GovernanceSignal,
        previous_signal: GovernanceSignal | None = None,
    ) -> Decision:

        self.regime_estimator.update(current_signal.energy)
        self.spiral_detector.update(current_signal.energy)

        regime = self.regime_estimator.regime(current_signal.energy)

        if regime == "critical":
            return Decision.REJECT

        if self.spiral_detector.is_spiraling():
            return Decision.FREEZE

        if previous_signal:
            if not self.dominance_engine.dominates(previous_signal, current_signal):
                return Decision.REJECT

        return Decision.ACCEPT




==================================================
FILE: calibration.py
==================================================

import numpy as np
from collections import deque


class EnergyRegimeEstimator:

    def __init__(self, window: int = 500):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy: float):
        self.history.append(energy)

    def percentile(self, p: float) -> float:
        if not self.history:
            return 1.0
        return float(np.percentile(self.history, p))

    def regime(self, energy: float) -> str:
        if len(self.history) < 50:
            return "uncalibrated"

        p90 = self.percentile(90)
        p98 = self.percentile(98)

        if energy > p98:
            return "critical"
        elif energy > p90:
            return "warning"
        return "stable"


class QuantileEnergyCalibrator:
    """
    FAR-controlled adaptive thresholding.

    tau = Q_negatives(FAR_target)
    """

    def __init__(self, far_target: float = 0.01):
        self.far_target = far_target
        self.tau = None

    def fit(self, negative_energies):
        self.tau = float(np.quantile(negative_energies, self.far_target))

    def accept(self, energy: float) -> bool:
        if self.tau is None:
            raise RuntimeError("Calibrator not fitted")
        return energy <= self.tau




==================================================
FILE: custom_types.py
==================================================

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Verdict(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"

@dataclass(frozen=True)
class SupportDiagnostics:
    """
    Sentence-level support analysis for summarization-style tasks.
    Aggregated over all sentences in a summary.
    """

    sentence_count: int
    paragraph_count: int

    # Entailment-style support scores
    max_entailment: float
    mean_entailment: float
    min_entailment: float

    # Similarity-based fallback metrics
    mean_sim_top1: float
    min_sim_top1: float
    mean_sim_margin: float
    min_sim_margin: float

    # Coverage signals
    mean_coverage: float
    min_coverage: float

    # Energy aggregates
    max_energy: float
    mean_energy: float
    min_energy: float
    high_energy_count: int

    p90_energy: float
    frac_above_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {
                "sentence_count": self.sentence_count,
                "paragraph_count": self.paragraph_count,
            },
            "entailment": {
                "max": self.max_entailment,
                "mean": self.mean_entailment,
                "min": self.min_entailment,
            },
            "similarity": {
                "mean_sim_top1": self.mean_sim_top1,
                "min_sim_top1": self.min_sim_top1,
                "mean_sim_margin": self.mean_sim_margin,
                "min_sim_margin": self.min_sim_margin,
            },
            "coverage": {
                "mean": self.mean_coverage,
                "min": self.min_coverage,
            },
            "energy": {
                "max": self.max_energy,
                "min": self.min_energy,
                "high_count": self.high_energy_count,
                "mean": self.mean_energy,
                "p90": self.p90_energy,
                "frac_above_threshold": self.frac_above_threshold,
            }
        }


@dataclass(frozen=True)
class GeometryDiagnostics:
    """
    Intrinsic geometric properties of claim–evidence interaction.
    All values are computed at SVD time.
    """

    # Spectral structure
    sigma1_ratio: float
    sigma2_ratio: float
    spectral_sum: float
    participation_ratio: float

    effective_rank: int
    used_count: int

    # Alignment
    alignment_to_sigma1: float

    # Similarity geometry
    sim_top1: float
    sim_top2: float
    sim_margin: float

    # Concentration / brittleness
    sensitivity: float

    # Optional raw vector (for offline research only)
    v1: np.ndarray
    entropy_rank: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral": {
                "sigma1_ratio": self.sigma1_ratio,
                "sigma2_ratio": self.sigma2_ratio,
                "spectral_sum": self.spectral_sum,
                "participation_ratio": self.participation_ratio,
                "effective_rank": self.effective_rank,
            },
            "alignment": {
                "alignment_to_sigma1": self.alignment_to_sigma1,
            },
            "similarity": {
                "sim_top1": self.sim_top1,
                "sim_top2": self.sim_top2,
                "sim_margin": self.sim_margin,
            },
            "robustness": {
                "sensitivity": self.sensitivity,
            },
            "support": {
                "effective_rank": self.effective_rank,
                "used_count": self.used_count,
                "entropy_rank": self.entropy_rank,
            },
        }


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    explained: float
    identity_error: float

    evidence_topk: int
    rank_cap: int

    geometry: GeometryDiagnostics

    def is_stable(self, threshold: float = 1e-4) -> bool:
        return self.identity_error < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.energy,
            "explained": self.explained,
            "identity_error": self.identity_error,
            "config": {
                "evidence_topk": self.evidence_topk,
                "rank_cap": self.rank_cap,
            },
            "geometry": self.geometry.to_dict(),
        }



@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation outcome."""

    claim: str
    evidence: List[str]

    energy_result: EnergyResult
    decision_trace: DecisionTrace
    verdict: Verdict
    policy_applied: str

    run_id: str
    split: str
    effectiveness: float

    embedding_info: Dict

    robustness_probe: Optional[List[float]] = None  # Energy under param variations
    difficulty_value: Optional[float] = 0.0
    difficulty_bucket: Optional[str] = None
    support_diagnostics: Optional[SupportDiagnostics] = None
    label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "meta": {
                "run_id": self.run_id,
                "split": self.split,
                "policy": self.policy_applied,
            },
            "claim": self.claim,
            "evidence": self.evidence,
            "difficulty": {
                "value": self.difficulty_value,
                "bucket": self.difficulty_bucket,
            },
            "effectiveness": self.effectiveness,
            "embedding": self.embedding_info,
            "decision": {
                "verdict": self.verdict.value if self.verdict else None,
            },
            "stability": {
                "probe_variance": float(np.var(self.robustness_probe))
                if self.robustness_probe
                else None,
            },
            "label": self.label
        }
        

        if self.energy_result is not None:
            data["energy"] = self.energy_result.to_dict()
            data["is_stable"] = self.energy_result.is_stable() 

        if self.support_diagnostics is not None:
            data["support"] = self.support_diagnostics.to_dict()

        if self.decision_trace is not None:
            data["decision"] = {
                "verdict": self.verdict.value if self.verdict else None,
            }

        return data

@dataclass(frozen=True)
class DecisionTrace:
    """
    Deterministic explanation of a 3-axis geometry-aware policy decision.
    """

    # === Core Energy Axis ===
    energy: float
    alignment: float  # |dot(claim, v1)|

    # === Geometry Axis ===
    participation_ratio: float
    sensitivity: float
    effectiveness: float
    difficulty: float

    # Policy thresholds
    tau_accept: Optional[float]
    tau_review: Optional[float]
    pr_threshold: Optional[float]
    sensitivity_threshold: Optional[float]
    margin_band: Optional[float]

    # Policy metadata
    policy_name: str
    hard_negative_gap: float

    # Final action
    verdict: str  # expected: "accept" | "review" | "reject"

    def to_dict(self) -> dict:
        return asdict(self)


def why_rejected(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REJECT verdicts.
    """
    if trace.verdict != "reject":
        return "Not rejected."

    reasons = []

    # Energy hard-reject (most common)
    if trace.tau_review is not None and trace.energy > trace.tau_review:
        reasons.append("Energy exceeds review threshold.")

    # If you ever make PR/Sensitivity hard-reject in policy, these become exact.
    # Otherwise they are informational: they explain why the sample is risky.
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("High PR (diffuse evidence manifold).")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("High sensitivity (brittle evidence dependence).")

    if trace.effectiveness < 0.05:
        reasons.append("Insufficient effectiveness margin.")

    if not reasons:
        reasons.append("Rejected by policy fallback.")

    return " | ".join(reasons)


def why_reviewed(trace: DecisionTrace) -> str:
    """
    Deterministic explanation for REVIEW verdicts.
    """
    if trace.verdict != "review":
        return "Not reviewed."

    reasons = []

    if (
        trace.margin_band is not None
        and trace.tau_accept is not None
        and abs(trace.energy - trace.tau_accept) <= trace.margin_band
    ):
        reasons.append("Within policy margin band.")

    # These are “human-facing” interpretations; keep them stable.
    if trace.difficulty > 0.4:
        reasons.append("Moderate difficulty region.")

    if trace.effectiveness < 0.25:
        reasons.append("Low effectiveness margin.")

    # Optional: geometry triggers (only if policy actually uses these to trigger REVIEW)
    if trace.pr_threshold is not None and trace.participation_ratio > trace.pr_threshold:
        reasons.append("PR exceeds threshold.")

    if trace.sensitivity_threshold is not None and trace.sensitivity > trace.sensitivity_threshold:
        reasons.append("Sensitivity exceeds threshold.")

    if not reasons:
        reasons.append("Reviewed by policy fallback.")

    return " | ".join(reasons)


==================================================
FILE: decision.py
==================================================

# policy/decision.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class Decision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"
    FREEZE = "freeze"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str
    signals: Dict[str, float]
    metadata: Optional[Dict] = None


==================================================
FILE: dominance.py
==================================================

from typing import Iterable

from policy.governance_signal import GovernanceSignal


class DominanceEngine:
    """
    True Pareto dominance check.
    """

    def __init__(self, critical_axes: Iterable[str]):
        self.critical_axes = set(critical_axes)

    def dominates(self, before: GovernanceSignal, after: GovernanceSignal) -> bool:
        improved_any = False

        for axis in self.critical_axes:
            before_val = getattr(before, axis)
            after_val = getattr(after, axis)

            if after_val < before_val:
                return False  # degraded critical axis

            if after_val > before_val:
                improved_any = True

        return improved_any


==================================================
FILE: energy_spiral_detector.py
==================================================


from collections import deque
import numpy as np

class EnergySpiralDetector:

    def __init__(self, window=100):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy):
        self.history.append(energy)

    def is_spiraling(self):
        if len(self.history) < self.window:
            return False

        y = np.array(self.history)
        x = np.arange(len(y))

        slope = np.polyfit(x, y, 1)[0]

        return slope > 0.002  # upward drift


==================================================
FILE: governance_signal.py
==================================================

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class GovernanceSignal:
    """
    Policy-level representation of AI output quality.

    Independent of Stephanie or Certum.
    """

    energy: float
    energy_p90: float
    high_energy_count: int
    entailment_mean: Optional[float] = None
    entailment_min: Optional[float] = None
    sim_margin: Optional[float] = None
    coverage: Optional[float] = None
    sentence_count: int
    embedding_margin: Optional[float] = None
    alignment: Optional[float] = None
    metadata: Optional[Dict] = None

    @property
    def instability(self) -> float:
        """Return raw instability proxy (1 - energy)."""
        return 1.0 - self.energy


def from_support_diagnostics(sd) -> GovernanceSignal:
    return GovernanceSignal(
        energy=sd.mean_energy,
        energy_p90=sd.p90_energy,
        high_energy_count=sd.high_energy_count,
        entailment_mean=sd.mean_entailment,
        entailment_min=sd.min_entailment,
        sim_margin=sd.mean_sim_margin,
        coverage=sd.mean_coverage,
        sentence_count=sd.sentence_count,
    )


==================================================
FILE: logging.py
==================================================



==================================================
FILE: monitor.py
==================================================

from collections import deque
import numpy as np


class EnergySpiralDetector:

    def __init__(self, window: int = 100):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, energy: float):
        self.history.append(energy)

    def slope(self) -> float:
        if len(self.history) < self.window:
            return 0.0

        y = np.array(self.history)
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    def is_spiraling(self) -> bool:
        return self.slope() > 0.002


==================================================
FILE: policy_container.py
==================================================

from typing import Any, Callable
from .decision import Decision
from .governance_signal import GovernanceSignal
from .base_policy import BasePolicy


class PolicyContainer:
    """
    Wraps ANY system. System never sees policy.
    """

    def __init__(
        self,
        system: Callable[..., Any],
        policy: BasePolicy,
        signal_adapter: Callable[[Any], GovernanceSignal],
    ):
        self.system = system
        self.policy = policy
        self.signal_adapter = signal_adapter
        self.previous_signal: GovernanceSignal | None = None

    def run(self, *args, **kwargs) -> tuple[Decision, Any]:

        result = self.system(*args, **kwargs)

        signal = self.signal_adapter(result)

        decision = self.policy.evaluate(
            current_signal=signal,
            previous_signal=self.previous_signal,
        )

        if decision == Decision.ACCEPT:
            self.previous_signal = signal

        return decision, result


==================================================
FILE: stability_monitor.py
==================================================

from collections import deque
import numpy as np


class StabilityMonitor:

    def __init__(self, window=50, energy_spike=0.6):
        self.window = window
        self.energy_spike = energy_spike
        self.energy_history = deque(maxlen=window)
        self.dominance_history = deque(maxlen=window)

    def update(self, energy, dominance_ok):
        self.energy_history.append(energy)
        self.dominance_history.append(1 if dominance_ok else 0)

    def spiral_detected(self) -> bool:
        if len(self.energy_history) < self.window:
            return False

        mean_energy = np.mean(self.energy_history)
        dominance_rate = np.mean(self.dominance_history)

        if mean_energy > self.energy_spike:
            return True

        if dominance_rate < 0.6:
            return True

        return False


==================================================
FILE: thresholds.py
==================================================

# policy/thresholds.py

from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    energy_safe: float = 0.35
    energy_warning: float = 0.45
    energy_critical: float = 0.55
    dominance_required: float = 0.80


==================================================
FILE: __init__.py
==================================================



==================================================
FILE: adapters\certum_adapter.py
==================================================

from policy.governance_signal import GovernanceSignal


class CertumAdapter:

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def __call__(self, result) -> GovernanceSignal:
        sd = result.support_diagnostics

        # Normalize energy (lower raw -> higher normalized)
        normalized_energy = 1.0 - min(max(sd.mean_energy, 0.0), 1.0)

        return GovernanceSignal(
            energy=normalized_energy,
            embedding_margin=float(sd.mean_sim_margin),
            alignment=float(sd.max_entailment),
            metadata=sd.to_dict(),
        )


==================================================
FILE: adapters\certum_geometry.py
==================================================


from policy.governance_signal import from_support_diagnostics


class CertumEnergyAdapter:
    """
    Wraps ClaimEvidenceGeometry + EntailmentModel
    and returns GovernanceSignal.
    """

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def compute_signal(self, summary_text, evidence_text):
        sd = self.support_analyzer.analyze(summary_text, evidence_text)
        return from_support_diagnostics(sd)


==================================================
FILE: axes\bundle.py
==================================================

# certum/axes/bundle.py

from typing import Dict

from policy.utils.dict_utils import deep_get


class AxisBundle:
    def __init__(self, axes: Dict[str, float]):
        self._axes = axes

    def get(self, name: str) -> float:
        return self._axes.get(name, 0.0)

    def items(self):
        return self._axes.items()

    def __repr__(self):
        return f"AxisBundle({self._axes})"

    # -----------------------------------------------------
    # Factory: build from decision trace
    # -----------------------------------------------------

    @classmethod
    def from_trace(cls, row: Dict) -> "AxisBundle":
        """
        Reconstruct AxisBundle from full report row.
        """

        axes = {
            "energy": deep_get(row, "energy", "value"),
            "participation_ratio": deep_get(
                row, "energy", "geometry", "spectral", "participation_ratio"
            ),
            "sensitivity": deep_get(
                row, "energy", "geometry", "robustness", "sensitivity"
            ),
            "alignment": deep_get(
                row, "energy", "geometry", "alignment", "alignment_to_sigma1"
            ),
            "effectiveness": deep_get(row, "effectiveness"),
        }

        return cls(axes)


==================================================
FILE: axes\difficulty.py
==================================================

class DifficultyAxis:

    name = "difficulty"

    def __init__(self, difficulty_computer):
        self.difficulty_computer = difficulty_computer

    def compute(self, context):
        return self.difficulty_computer.compute(context)


==================================================
FILE: axes\energy.py
==================================================

class EnergyAxis:

    def __init__(self, computer):
        self.computer = computer
        self.name = "energy"

    def compute(self, context):
        result = self.computer.compute(
            context["claim_vec"],
            context["evidence_vecs"],
        )

        context["energy_result"] = result
        return result.energy


==================================================
FILE: axes\engine.py
==================================================

from typing import Any, Dict, List, Tuple

from policy.protocols.axes import AxisCalculator


class AxisEngine:

    def __init__(self, axes: List[AxisCalculator]):
        self.axes = axes

    def compute(
        self,
        claim_vec,
        evidence_vecs,
    ) -> Tuple[Dict[str, float], Any]:

        context: Dict[str, Any] = {
            "claim_vec": claim_vec,
            "evidence_vecs": evidence_vecs,
        }

        axes_values: Dict[str, float] = {}

        for axis in self.axes:
            value = axis.compute(context)
            axes_values[axis.name] = float(value)

        energy_result = context.get("energy_result")

        return axes_values, energy_result


==================================================
FILE: axes\participation_ratio.py
==================================================

class ParticipationRatioAxis:

    name = "participation_ratio"

    def compute(self, context):
        energy_result = context["energy_result"]
        return energy_result.geometry.participation_ratio


==================================================
FILE: axes\sensitivity.py
==================================================

class SensitivityAxis:

    name = "sensitivity"

    def compute(self, context):
        energy_result = context["energy_result"]
        return energy_result.geometry.sensitivity


==================================================
FILE: embedding\embedding_store.py
==================================================

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List

import numpy as np


class EmbeddingStore:

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self._init_pragmas()
        self._init_schema()

    # -------------------------------------------------
    # Schema
    # -------------------------------------------------
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vec BLOB NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(text_hash, model)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash_model
            ON embeddings(text_hash, model)
        """)
        self.conn.commit()

    # -------------------------------------------------
    # Pragmas
    # -------------------------------------------------
    def _init_pragmas(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA mmap_size=30000000000")
        cur.close()

    # -------------------------------------------------
    # Hashing
    # -------------------------------------------------
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -------------------------------------------------
    # Fetch
    # -------------------------------------------------
    def get(self, texts: List[str], model: str):
        if not texts:
            return [], []

        hashes = [self._hash_text(t) for t in texts]

        rows = {}
        chunk = 900

        for i in range(0, len(hashes), chunk):
            sub = hashes[i:i+chunk]
            q = ",".join(["?"] * len(sub))

            sql = f"""
                SELECT text_hash, vec, dim
                FROM embeddings
                WHERE model = ?
                  AND text_hash IN ({q})
            """

            cur = self.conn.cursor()
            cur.execute(sql, [model, *sub])
            for r in cur.fetchall():
                rows[r["text_hash"]] = r
            cur.close()

        vecs = []
        missing_idx = []

        for i, h in enumerate(hashes):
            row = rows.get(h)
            if row is None:
                vecs.append(None)
                missing_idx.append(i)
                continue

            dim = int(row["dim"])
            v = np.frombuffer(row["vec"], dtype=np.float32)

            if v.shape[0] != dim:
                vecs.append(None)
                missing_idx.append(i)
                continue

            vecs.append(v)

        return vecs, missing_idx

    # -------------------------------------------------
    # Insert
    # -------------------------------------------------
    def put(self, texts: List[str], vecs: np.ndarray, model: str):
        now = time.time()
        cur = self.conn.cursor()

        for text, vec in zip(texts, vecs):
            text_hash = self._hash_text(text)

            cur.execute("""
                INSERT OR REPLACE INTO embeddings
                (text, text_hash, model, dim, vec, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                text,
                text_hash,
                model,
                int(vec.shape[0]),
                vec.astype(np.float32).tobytes(),
                now,
            ))

        self.conn.commit()
        cur.close()


==================================================
FILE: embedding\hf_embedder.py
==================================================

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from policy.protocols.embedding_backend import EmbeddingBackend


class HFEmbedder:
    """
    HuggingFace embedder that delegates storage
    to an EmbeddingBackend.
    """

    name: str = "HFEmbedder"

    def __init__(
        self,
        model_name: str,
        backend: EmbeddingBackend,
    ):
        self.model_name = model_name
        self.backend = backend
        self.model = SentenceTransformer(model_name)

        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*embeddings.position_ids.*"
        )

        # Simple in-memory cache (hashable key)
        self._memory_cache: dict[tuple[str, ...], np.ndarray] = {}

    # -------------------------------------------------
    # RAW embedding (no backend interaction)
    # -------------------------------------------------
    def _embed_raw(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {vecs.shape}")

        return vecs

    # -------------------------------------------------
    # Public embed (backend-aware)
    # -------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        # print(f"Embedding {len(texts)} texts with model '{self.model_name}'...")

        key = tuple(texts)
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 1. Load from backend
        vecs, missing_idx = self.backend.get(texts, self.model_name)

        # 2. Compute missing
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vecs = self._embed_raw(missing_texts)

            # Persist
            self.backend.put(missing_texts, new_vecs, self.model_name)

            # Fill into vecs list
            for i, v in zip(missing_idx, new_vecs):
                vecs[i] = v

        # 3. Stack
        result = np.stack(vecs, axis=0)

        # Cache in-memory
        self._memory_cache[key] = result
        return result

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()



==================================================
FILE: embedding\sqlite_embedding_backend.py
==================================================


from policy.embedding.embedding_store import EmbeddingStore


class SQLiteEmbeddingBackend:
    def __init__(self, db_path: str):
        self.store = EmbeddingStore(db_path)

    def get(self, texts, model):
        return self.store.get(texts, model)

    def put(self, texts, vecs, model):
        self.store.put(texts, vecs, model)



==================================================
FILE: embedding\__init__.py
==================================================



==================================================
FILE: experiments\ablations.py
==================================================



==================================================
FILE: experiments\benchmark.py
==================================================

from typing import Callable


class DynamicStabilityBenchmark:

    def __init__(self, container, steps: int = 1000):
        self.container = container
        self.steps = steps

    def run(self, input_generator: Callable):

        decisions = []
        energies = []

        for _ in range(self.steps):
            decision, result = self.container.run(input_generator())
            decisions.append(decision)
            energies.append(result.support_diagnostics.mean_energy)

        return {
            "decisions": decisions,
            "energies": energies,
        }


==================================================
FILE: experiments\pipeline.py
==================================================

# src/certum/evaluation/pipeline.py

from pathlib import Path
import logging

from policy.embedding.hf_embedder import HFEmbedder
from policy.embedding.sqlite_embedding_backend import SQLiteEmbeddingBackend
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.geometry.nli_wrapper import EntailmentModel
from policy.geometry.sentence_support import SentenceSupportAnalyzer
from policy.orchestration.summarization_runner import SummarizationRunner


logger = logging.getLogger(__name__)


# =========================================================
# Evaluation Pipeline Builder
# =========================================================

def run_summarization_pipeline(
    *,
    samples: list,
    embedding_model: str,
    embedding_db: Path,
    nli_model: str,
    entailment_db: Path,
    top_k: int,
    geometry_top_k: int,
    rank_r: int,
    out_path: Path,
) -> list:
    """
    Builds full summarization evaluation stack and executes it.

    Returns structured results list.
    """

    # -----------------------------------------------------
    # Embedding Backend
    # -----------------------------------------------------

    backend = SQLiteEmbeddingBackend(str(embedding_db))
    logger.info(f"Using embedding DB: {embedding_db}")

    embedder = HFEmbedder(
        embedding_model,
        backend=backend,
    )

    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------

    energy_computer = ClaimEvidenceGeometry(
        top_k=geometry_top_k,
        rank_r=rank_r,
    )

    # -----------------------------------------------------
    # Entailment
    # -----------------------------------------------------

    entailment_model = EntailmentModel(
        model_name=nli_model,
        batch_size=32,
        db_path=str(entailment_db),
    )

    # -----------------------------------------------------
    # Sentence Support Analyzer
    # -----------------------------------------------------

    support_analyzer = SentenceSupportAnalyzer(
        embedder=embedder,
        energy_computer=energy_computer,
        entailment_model=entailment_model,
        top_k=top_k,
    )

    # -----------------------------------------------------
    # Summarization Runner
    # -----------------------------------------------------

    summarization_runner = SummarizationRunner(
        support_analyzer=support_analyzer
    )

    logger.info("Running summarization pipeline...")

    results = summarization_runner.run(
        samples,
        out_path=out_path,
    )

    return results


==================================================
FILE: geometry\claim_evidence.py
==================================================

import logging
from typing import List, Tuple

import numpy as np
from scipy.linalg import svd

from policy.custom_types import EnergyResult, GeometryDiagnostics

logger = logging.getLogger(__name__)


class ClaimEvidenceGeometry:
    """
    Computes hallucination energy and full geometric diagnostics
    for a claim–evidence embedding pair.

    Core outputs:
        - Energy (1 - explained variance)
        - Spectral diagnostics
        - Similarity ambiguity
        - Robustness sensitivity
        - Alignment to dominant spectral axis
    """

    def __init__(self, top_k: int = 12, rank_r: int = 8):
        self.top_k = top_k
        self.rank_r = rank_r

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def compute(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> EnergyResult:
        """
        Compute hallucination energy and geometry diagnostics.
        """

        logger.debug("Starting energy computation.")

        claim_vec, evidence_vecs = self._validate_and_prepare(
            claim_vec, evidence_vecs
        )

        if evidence_vecs.size == 0:
            logger.warning("Empty evidence vectors received.")
            return EnergyResult(
                energy=1.0,
                explained=0.0,
                identity_error=1.0,
                evidence_topk=0,
                rank_cap=0,
                geometry=None
            )

        # Normalize
        c = _unit_norm(claim_vec)
        E = _unit_norm_rows(evidence_vecs)

        # Build spectral basis
        basis, effective_rank, S, Vt = self._build_evidence_basis(c, E)

        # ========================================================
        # Spectral Metrics
        # ========================================================

        eps = 1e-12
        sum_sigma = float(np.sum(S))
        sum_sq = float(np.sum(S ** 2))

        sigma1 = float(S[0]) if len(S) > 0 else 0.0
        sigma2 = float(S[1]) if len(S) > 1 else 0.0

        sigma1_ratio = sigma1 / max(sum_sigma, eps)
        sigma2_ratio = sigma2 / max(sum_sigma, eps)

        participation_ratio = (sum_sigma ** 2) / (sum_sq + eps)

        # Entropy rank (more stable than raw rank)
        if sum_sigma > eps:
            p = S / sum_sigma
            p = p[p > eps]
            entropy_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            entropy_rank = 0.0

        # ========================================================
        # Projection Energy
        # ========================================================

        projected = basis.T @ c if basis.shape[1] > 0 else np.zeros(0)
        explained = float(np.dot(projected, projected))
        energy = float(np.clip(1.0 - explained, 0.0, 1.0))
        identity_error = abs(1.0 - (explained + energy))

        # ========================================================
        # Similarity Metrics
        # ========================================================

        sims = E @ c
        sorted_sims = np.sort(sims)[::-1]

        sim_top1 = float(sorted_sims[0]) if len(sorted_sims) > 0 else 0.0
        sim_top2 = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
        sim_margin = sim_top1 - sim_top2

        # ========================================================
        # Sensitivity (concentration proxy)
        # ========================================================

        k = min(self.top_k, E.shape[0])
        idx = np.argsort(-sims)[:k]
        sims_topk = np.maximum(sims[idx], 0.0)

        if np.sum(sims_topk) < eps:
            sensitivity = 1.0
        else:
            sensitivity = float(np.max(sims_topk) / np.sum(sims_topk))

        # ========================================================
        # Alignment to dominant spectral axis
        # ========================================================

        if Vt.shape[0] > 0:
            v1 = Vt[0]
            v1 = v1 / (np.linalg.norm(v1) + eps)
            alignment = float(abs(np.dot(c, v1)))
        else:
            v1 = np.zeros((E.shape[1],), dtype=np.float32)
            alignment = 0.0

        # ========================================================
        # Geometry Object
        # ========================================================

        geometry = GeometryDiagnostics(
            sigma1_ratio=sigma1_ratio,
            sigma2_ratio=sigma2_ratio,
            spectral_sum=sum_sigma,
            participation_ratio=float(participation_ratio),
            effective_rank=int(effective_rank),
            used_count=int(E.shape[0]),
            entropy_rank=entropy_rank,
            alignment_to_sigma1=alignment,
            sim_top1=sim_top1,
            sim_top2=sim_top2,
            sim_margin=sim_margin,
            sensitivity=sensitivity,
            v1=v1.astype(np.float32),
        )

        logger.debug(
            "Energy computed: energy=%.4f explained=%.4f alignment=%.4f",
            energy, explained, alignment
        )

        return EnergyResult(
            energy=energy,
            explained=explained,
            identity_error=identity_error,
            evidence_topk=min(self.top_k, E.shape[0]),
            rank_cap=self.rank_r,
            geometry=geometry,
        )

    # ============================================================
    # ROBUSTNESS PROBE
    # ============================================================

    def compute_robustness_probe(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
        param_variants: List[Tuple[int, int]] = [(8, 6), (12, 8), (20, 10)],
    ) -> List[float]:
        """
        Evaluate energy stability under parameter variation.
        """

        logger.debug("Running robustness probe.")

        probes = []

        for top_k, rank_r in param_variants:
            try:
                computer = ClaimEvidenceGeometry(top_k=top_k, rank_r=rank_r)
                res = computer.compute(claim_vec, evidence_vecs)
                probes.append(res.energy)
            except Exception as e:
                logger.exception("Robustness probe failed: %s", str(e))
                probes.append(1.0)

        return probes

    # ============================================================
    # SENSITIVITY (LOO Spike)
    # ============================================================

    def compute_sensitivity(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> float:
        """
        Leave-one-out brittleness test.
        """

        n = len(evidence_vecs)
        if n <= 1:
            return 1.0

        base_energy = self.compute(claim_vec, evidence_vecs).energy

        max_spike = 0.0
        for i in range(n):
            loo = np.delete(evidence_vecs, i, axis=0)
            loo_energy = self.compute(claim_vec, loo).energy
            spike = loo_energy - base_energy
            max_spike = max(max_spike, spike)

        return float(max(0.0, max_spike))

    # ============================================================
    # BASIS BUILDER
    # ============================================================

    def _build_evidence_basis(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:

        if evidence_vecs.shape[0] == 0:
            return (
                np.zeros((claim_vec.shape[0], 0), dtype=np.float32),
                0,
                np.array([]),
                np.array([]),
            )

        E_topk = evidence_vecs

        try:
            _, S, Vt = svd(E_topk, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.exception("SVD failed.")
            d = evidence_vecs.shape[1]
            return (
                np.zeros((d, 0), dtype=np.float32),
                0,
                np.array([]),
                np.array([]),
            )

        r = min(self.rank_r, Vt.shape[0])
        basis = Vt[:r].T
        effective_rank = int(np.sum(S > 1e-6))

        return basis.astype(np.float32), effective_rank, S, Vt

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_and_prepare(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ):
        claim_vec = np.asarray(claim_vec, dtype=np.float32)
        evidence_vecs = np.asarray(evidence_vecs, dtype=np.float32)

        if claim_vec.ndim != 1:
            raise ValueError(f"claim_vec must be 1D vector, got {claim_vec.shape}")

        if evidence_vecs.ndim == 1:
            evidence_vecs = evidence_vecs.reshape(1, -1)
        elif evidence_vecs.ndim != 2:
            raise ValueError(f"evidence_vecs must be 2D, got {evidence_vecs.shape}")

        if claim_vec.shape[0] != evidence_vecs.shape[1]:
            raise ValueError("Dimension mismatch claim/evidence")

        return claim_vec, evidence_vecs


# ================================================================
# Normalization Utilities
# ================================================================

def _unit_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x)
    return x / max(norm, eps)


def _unit_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


==================================================
FILE: geometry\nli_wrapper.py
==================================================

import hashlib
import sqlite3
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentModel:

    def __init__(
        self,
        model_name: str,
        db_path: str = "nli_cache.db",
        device: str = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.entailment_index = 2

        # Setup SQLite
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS nli_cache (
            key TEXT PRIMARY KEY,
            entailment_prob REAL
        )
        """)
        self.conn.commit()

    def _make_key(self, premise: str, hypothesis: str) -> str:
        combined = premise.strip() + "|||" + hypothesis.strip()
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _lookup_cache(self, keys: List[str]):
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT key, entailment_prob FROM nli_cache WHERE key IN ({placeholders})"
        rows = self.conn.execute(query, keys).fetchall()
        return {k: v for k, v in rows}

    def _insert_cache(self, key_score_pairs):
        self.conn.executemany(
            "INSERT OR REPLACE INTO nli_cache (key, entailment_prob) VALUES (?, ?)",
            key_score_pairs
        )
        self.conn.commit()

    @torch.no_grad()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:

        keys = [self._make_key(p, h) for p, h in pairs]
        cached = self._lookup_cache(keys)

        results = []
        to_compute = []
        compute_indices = []

        # Separate cached vs new
        for i, key in enumerate(keys):
            if key in cached:
                results.append(cached[key])
            else:
                results.append(None)
                to_compute.append(pairs[i])
                compute_indices.append(i)

        # If nothing new → return immediately
        if not to_compute:
            return results

        # Compute missing in batches
        new_scores = []

        for i in range(0, len(to_compute), self.batch_size):
            batch = to_compute[i:i+self.batch_size]
            premises = [p for p, h in batch]
            hypotheses = [h for p, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            entailment_probs = probs[:, self.entailment_index]

            new_scores.extend(entailment_probs.detach().cpu().numpy().tolist())

        # Insert into cache
        new_key_score_pairs = [
            (keys[idx], score)
            for idx, score in zip(compute_indices, new_scores)
        ]
        self._insert_cache(new_key_score_pairs)

        # Fill in results
        for idx, score in zip(compute_indices, new_scores):
            results[idx] = score

        return results


==================================================
FILE: geometry\sentence_support.py
==================================================

import numpy as np

from policy.custom_types import SupportDiagnostics
from policy.utils.text_utils import split_into_paragraphs, split_into_sentences


class SentenceSupportAnalyzer:

    def __init__(self, embedder, energy_computer=None, entailment_model=None, top_k=3):
        self.embedder = embedder
        self.energy_computer = energy_computer
        self.entailment_model = entailment_model
        self.top_k = top_k

    def analyze(self, summary_text, evidence_text):

        paragraphs = split_into_paragraphs(evidence_text)
        para_vecs = self.embedder.embed(paragraphs)

        sentences = split_into_sentences(summary_text)

        if not sentences:
            return None

        # ---------------------------------------
        # Collect top-k paragraph candidates
        # ---------------------------------------

        all_pairs = []
        sentence_meta = []

        sentence_energies = []
        sim_top1_vals = []
        sim_margin_vals = []
        coverage_vals = []

        for sent in sentences:

            sent_vec = self.embedder.embed([sent])[0]

            sims = para_vecs @ sent_vec
            idx = np.argsort(-sims)[:self.top_k]

            candidates = [paragraphs[i] for i in idx]

            # Save similarity metrics
            sims_sorted = np.sort(sims)[::-1]
            sim_top1_vals.append(float(sims_sorted[0]))
            sim_margin_vals.append(
                float(sims_sorted[0] - sims_sorted[1])
                if len(sims_sorted) > 1 else 0.0
            )

            coverage_vals.append(float(np.mean(sims > 0.3)))

            # Energy (optional)
            if self.energy_computer:
                result = self.energy_computer.compute(
                    claim_vec=sent_vec,
                    evidence_vecs=para_vecs
                )
                sentence_energies.append(result.energy)
            else:
                sentence_energies.append(0.0)

            # Collect entailment pairs
            if self.entailment_model:
                for p in candidates:
                    all_pairs.append((p, sent))
                sentence_meta.append(len(candidates))

        # ---------------------------------------
        # Batched entailment
        # ---------------------------------------

        entailment_scores = []

        if self.entailment_model and all_pairs:
            entailment_scores = self.entailment_model.score_pairs(all_pairs)

        # ---------------------------------------
        # Restructure entailment scores
        # ---------------------------------------

        entailment_max = []
        entailment_mean = []
        entailment_min = []

        pointer = 0

        for count in sentence_meta:
            sent_scores = entailment_scores[pointer:pointer+count]
            pointer += count

            if sent_scores:
                entailment_max.append(max(sent_scores))
                entailment_mean.append(np.mean(sent_scores))
                entailment_min.append(min(sent_scores))
            else:
                entailment_max.append(0.0)
                entailment_mean.append(0.0)
                entailment_min.append(0.0)

        # ---------------------------------------
        # Aggregate into SupportDiagnostics
        # ---------------------------------------

        sentence_energies = np.array(sentence_energies)
        sim_top1_vals = np.array(sim_top1_vals)
        sim_margin_vals = np.array(sim_margin_vals)
        coverage_vals = np.array(coverage_vals)

        entailment_max = np.array(entailment_max) if entailment_max else np.zeros(len(sentences))
        entailment_mean = np.array(entailment_mean) if entailment_mean else np.zeros(len(sentences))
        entailment_min = np.array(entailment_min) if entailment_min else np.zeros(len(sentences))

        threshold = 0.5  # This can be tuned based on validation data

        return SupportDiagnostics(
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),

            max_entailment=float(np.max(entailment_max)),
            mean_entailment=float(np.mean(entailment_mean)),
            min_entailment=float(np.min(entailment_min)),

            mean_sim_top1=float(np.mean(sim_top1_vals)),
            min_sim_top1=float(np.min(sim_top1_vals)),
            mean_sim_margin=float(np.mean(sim_margin_vals)),
            min_sim_margin=float(np.min(sim_margin_vals)),

            mean_coverage=float(np.mean(coverage_vals)),
            min_coverage=float(np.min(coverage_vals)),

            max_energy=float(np.max(sentence_energies)),
            mean_energy=float(np.mean(sentence_energies)),
            min_energy = float(np.min(sentence_energies)),
            high_energy_count = int(np.sum(sentence_energies > threshold)),
            p90_energy=float(np.percentile(sentence_energies, 90)),
            frac_above_threshold=float(np.mean(sentence_energies > 0.5)),
        )


==================================================
FILE: geometry\support_types.py
==================================================

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SupportDiagnostics:
    """
    Sentence-level support analysis for summarization-style tasks.
    Aggregated over all sentences in a summary.
    """

    sentence_count: int
    paragraph_count: int

    # Entailment-style support scores
    max_entailment: float
    mean_entailment: float
    min_entailment: float

    # Similarity-based fallback metrics
    mean_sim_top1: float
    min_sim_top1: float
    mean_sim_margin: float
    min_sim_margin: float

    # Coverage signals
    mean_coverage: float
    min_coverage: float

    # Energy aggregates
    max_energy: float
    mean_energy: float
    min_energy: float
    high_energy_count: int

    p90_energy: float
    frac_above_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": {
                "sentence_count": self.sentence_count,
                "paragraph_count": self.paragraph_count,
            },
            "entailment": {
                "max": self.max_entailment,
                "mean": self.mean_entailment,
                "min": self.min_entailment,
            },
            "similarity": {
                "mean_sim_top1": self.mean_sim_top1,
                "min_sim_top1": self.min_sim_top1,
                "mean_sim_margin": self.mean_sim_margin,
                "min_sim_margin": self.min_sim_margin,
            },
            "coverage": {
                "mean": self.mean_coverage,
                "min": self.min_coverage,
            },
            "energy": {
                "max": self.max_energy,
                "min": self.min_energy,
                "high_count": self.high_energy_count,
                "mean": self.mean_energy,
                "p90": self.p90_energy,
                "frac_above_threshold": self.frac_above_threshold,
            }
        }


==================================================
FILE: orchestration\summarization_runner.py
==================================================

import json
import time
from pathlib import Path

from tqdm import tqdm

from policy.custom_types import EvaluationResult


class SummarizationRunner:

    def __init__(self, support_analyzer):
        self.support_analyzer = support_analyzer

    def run(self, samples, out_path):

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        start_time = time.time()

        with out_path.open("w", encoding="utf-8") as f:

            for i, sample in enumerate(tqdm(samples, desc="Analyzing summaries")):

                support_diag = self.support_analyzer.analyze(
                    summary_text=sample["claim"],
                    evidence_text=sample["evidence"][0],
                )

                evaluation = EvaluationResult(
                    claim=sample["claim"],
                    evidence=sample["evidence"],
                    energy_result=None,
                    decision_trace=None,
                    verdict=None,
                    policy_applied="support_analysis",
                    run_id="summary_v1",
                    split="test",
                    effectiveness=0.0,
                    embedding_info={},
                    support_diagnostics=support_diag,
                    label=sample["label"],
                )

                results.append(evaluation)

                # ✅ Stream write immediately
                f.write(json.dumps(evaluation.to_dict()) + "\n")

                # Optional: flush every 50 samples
                if i % 50 == 0:
                    f.flush()

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(samples)} samples in {elapsed:.2f}s")

        return results

    def _write_jsonl(self, results, out_path: str):

        def convert(o):
            import numpy as np

            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=convert)


    def _write_csv(self, results, csv_path):
        import pandas as pd

        rows = []
        for r in results:
            row = r.support_diagnostics.to_dict()
            row["label"] = r.verdict  # or sample label I
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)


==================================================
FILE: protocols\axes.py
==================================================

from typing import Any, Dict, Protocol


class AxisCalculator(Protocol):
    """
    Structural typing: any object with these members is an Axis.
    """

    name: str

    def compute(self, context: Dict[str, Any]) -> float:
        ...


==================================================
FILE: protocols\calibration.py
==================================================

from typing import Any, Dict, List, Optional, Protocol

import numpy as np


class Calibrator(Protocol):
    """
    Threshold calibration interface.

    Responsible for:
        - Computing energy-based thresholds
        - Enforcing target FAR
        - Producing calibration statistics

    Must NOT:
        - Load datasets
        - Apply final policy
        - Write files

    Pure calibration logic only.
    """

    def run_sweep(
        self,
        *,
        claims: List[str],
        evidence_sets: List[List[str]],
        evidence_vecs: List[np.ndarray],
        percentiles: List[int],
        neg_mode: str,
        seed: int,
        neg_offset: Optional[int] = None,
        claim_vec_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Executes calibration sweep.

        Returns:
            {
                "tau_energy": float,
                "tau_pr": float,
                "tau_sensitivity": float,
                "hard_negative_gap": float,
                ...
            }
        """
        ...


==================================================
FILE: protocols\embedder.py
==================================================

from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    """
    Computes embeddings (may internally use backend).
    """

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns (n, d) float32 embeddings.
        """

    def dimension(self) -> int:
        """
        Embedding dimensionality.
        """


==================================================
FILE: protocols\embedding_backend.py
==================================================

from typing import List, Protocol, Tuple

import numpy as np


class EmbeddingBackend(Protocol):
    """
    Storage contract for embeddings.
    Does NOT compute embeddings.
    """

    def get(
        self,
        texts: List[str],
        model: str,
    ) -> Tuple[List[np.ndarray | None], List[int]]:
        """
        Retrieve embeddings.

        Returns:
            vecs: aligned to input order (None where missing)
            missing_indices: indices of texts not found
        """
        ...

    def put(
        self,
        texts: List[str],
        vecs: np.ndarray,
        model: str,
    ) -> None:
        """
        Persist embeddings.
        """
        ...


==================================================
FILE: protocols\evidence_store.py
==================================================

from typing import List, Protocol, Tuple

import numpy as np


class EvidenceStore(Protocol):
    """
    Generic supporting data interface.
    Dataset-agnostic.
    """

    def has(self, element_id: str) -> bool:
        ...

    def get_texts(self, element_ids: List[str]) -> List[str]:
        ...

    def get_texts_and_vectors(
        self,
        element_ids: List[str],
        model: str,
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Returns:
            texts
            vectors
            missing_ids
        """


==================================================
FILE: protocols\gate.py
==================================================

from typing import List, Optional, Protocol

import numpy as np

from policy.custom_types import EnergyResult, EvaluationResult
from policy.geometry.claim_evidence import ClaimEvidenceGeometry
from policy.protocols.embedder import Embedder


class Gate(Protocol):
    """
    Deterministic policy execution interface.

    A Gate implementation must:
        - Accept precomputed diagnostics
        - Apply a policy
        - Return EvaluationResult

    It must NOT:
        - Perform embedding
        - Compute energy
        - Perform calibration
        - Tune thresholds
    """

    def compute_energy(
        claim: str,
        evidence_texts: List[str],
        *,
        claim_vec: Optional[np.ndarray] = None,
        evidence_vecs: Optional[np.ndarray] = None,
    ) -> EnergyResult:
        """
        Compute hallucination energy WITHOUT policy decision.
        Used during calibration sweeps.
        """
        ...

    def evaluate(self, embedder: Embedder,
        energy_computer: ClaimEvidenceGeometry,) -> EvaluationResult:
        """
        Executes deterministic policy decision.

        Expected inputs:
            {
                "run_id": str,
                "claim": str,
                "evidence": List[str],
                "axes": AxisBundle,
                "energy_result": EnergyResult,
                "effectiveness": float,
                "embedding_info": dict,
                "split": str,
            }

        Returns:
            EvaluationResult
        """
        ...


==================================================
FILE: protocols\geometry.py
==================================================

# certum/protocols/geometry.py

from typing import List, Protocol, Tuple

import numpy as np

from policy.custom_types import EnergyResult


class GeometryComputer(Protocol):
    """
    Claim–Evidence geometric analysis interface.

    Responsible for:
        - Projection-based residual computation
        - Spectral diagnostics
        - Similarity ambiguity metrics
        - Sensitivity diagnostics
        - Robustness probes

    Must NOT:
        - Apply policy
        - Perform thresholding
        - Load data
        - Perform embedding

    Pure geometric computation only.
    """

    def compute(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> EnergyResult:
        """
        Compute projection energy and full geometry diagnostics.
        """
        ...

    def compute_robustness_probe(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
        param_variants: List[Tuple[int, int]],
    ) -> List[float]:
        """
        Evaluate stability of energy under parameter variation.
        """
        ...

    def compute_sensitivity(
        self,
        claim_vec: np.ndarray,
        evidence_vecs: np.ndarray,
    ) -> float:
        """
        Leave-one-out brittleness probe.
        """
        ...


==================================================
FILE: protocols\policy.py
==================================================

# certum/protocols/policy.py

from typing import Optional, Protocol

from policy.axes.bundle import AxisBundle
from policy.custom_types import Verdict


class Policy(Protocol):

    # Required attributes
    tau_accept: float
    tau_review: Optional[float]
    hard_negative_gap: float

    # Required behavior
    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:
        ...

    @property
    def name(self) -> str:
        ...


==================================================
FILE: utils\dict_utils.py
==================================================

def deep_get(d: dict, *keys, default=0.0):
    """
    Safely retrieve nested dict values.

    Example:
        deep_get(row, "energy", "geometry", "spectral", "participation_ratio")
    """
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


==================================================
FILE: utils\text_utils.py
==================================================

import re


def clean_wiki_markup(text: str) -> str:
    # [[Page|Display]] → Display
    text = re.sub(r"\[\[[^\|\]]+\|([^\]]+)\]\]", r"\1", text)

    # [[Page]] → Page
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Replace underscores with spaces
    text = text.replace("_", " ")

    return text

def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def split_into_paragraphs(text: str):
    """
    Split document into paragraph-level chunks.
    Handles both double-newline and fallback chunking.
    """
    # First try real paragraph breaks
    paras = re.split(r"\n\s*\n", text)

    # If document has no paragraph breaks (CNN style), fallback to length chunking
    if len(paras) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunk_size = 5  # 5 sentences per pseudo-paragraph
        paras = [
            " ".join(sentences[i:i+chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]

    # Clean empty
    paras = [p.strip() for p in paras if p.strip()]

    return paras

