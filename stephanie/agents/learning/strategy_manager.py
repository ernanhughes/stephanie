# stephanie/agents/learning/strategy_manager.py
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Tuple
import json
import random
import time
import math
import logging

from stephanie.utils.json_sanitize import dumps_safe

_logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Strategy:
    """
    Tunable knobs for verification. Immutable so we can reason about versions.
    Use StrategyManager.set_state(...) to change.
    """
    verification_threshold: float = 0.85
    skeptic_weight: float = 0.34
    editor_weight: float = 0.33
    risk_weight: float = 0.33
    version: int = 1

class StrategyManager:
    """
    Owns strategy knobs, runs lightweight A/B enrollment and validation,
    and records evolution breadcrumbs into casebooks as scorables.
    """

    def __init__(self, cfg, memory, container, logger):
        self.cfg, self.memory, self.container, self.logger = cfg, memory, container, logger

        self.casebook_tag: str = cfg.get("casebook_action", "blog")
        self.min_gain: float = float(cfg.get("min_gain", 0.01))
        self.high_gain: float = float(cfg.get("high_gain", 0.03))

        # A/B validation knobs
        self.history_limit: int = int(cfg.get("strategy_test_history", 20))
        self.min_samples: int = int(cfg.get("ab_min_samples", 10))
        self.window_seconds: Optional[int] = (
            int(cfg["ab_window_seconds"]) if "ab_window_seconds" in cfg else None
        )

        # Deterministic, seedable randomness (or you can hash case_id)
        seed = cfg.get("rng_seed")
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Current state (immutable dataclass)
        self.state: Strategy = Strategy(
            verification_threshold=float(cfg.get("verification_threshold", 0.85)),
            skeptic_weight=float(cfg.get("skeptic_weight", 0.34)),
            editor_weight=float(cfg.get("editor_weight", 0.33)),
            risk_weight=float(cfg.get("risk_weight", 0.33)),
            version=1,
        )

        self._evolution_log: List[Dict[str, Any]] = []

    # ---------- Public helpers ----------
    def get_state(self) -> Strategy:
        return self.state

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self.state)

    def set_state(self, **fields) -> None:
        """Safely replace fields (keeps immutability)."""
        self.state = replace(self.state, **fields)

    def bump_version(self) -> None:
        self.set_state(version=self.state.version + 1)

    # ---------- Recording ----------
    def record_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change"):
        payload = {**self.as_dict(), "tag": tag, "timestamp": time.time()}
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_state",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag},
                )
        except Exception:
            # Never block on telemetry
            pass

    # ---------- Evolution logic ----------
    def propose(self, avg_gain: float) -> Strategy:
        """
        Create a proposed strategy (copy) based on observed avg_gain.
        """
        s = self.state
        if avg_gain < self.min_gain:
            change = 0.06 if avg_gain < 0.005 else 0.03
            return replace(
                s,
                skeptic_weight=min(0.60, s.skeptic_weight + change),
                editor_weight=max(0.20, s.editor_weight - change / 2),
                risk_weight=max(0.20, s.risk_weight - change / 2),
            )
        elif avg_gain > self.high_gain:
            return replace(s, verification_threshold=max(0.80, s.verification_threshold - 0.01))
        return s  # no change

    def _pick_group(self, case_id: Optional[int]) -> str:
        """
        Deterministic-ish group selection.
        If rng_seed provided, uses seeded RNG; else hash case_id fallback.
        """
        if case_id is not None and self.cfg.get("ab_hash_assign", True) and self.cfg.get("rng_seed") is None:
            # Stable across runs on same case_id
            g = "B" if (hash((case_id, self.casebook_tag)) & 1) else "A"
            return g
        return "B" if self.rng.random() < 0.5 else "A"

    def evolve(self, iterations: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        """
        Called at the end of a section’s refinement loop.
        Computes avg gain, proposes changes, enrolls next unit in A or B,
        records breadcrumb, and (if B) updates current state.
        """
        if len(iterations) < 2:
            self.record_state(context, "pre_change")
            return

        gains = [
            (iterations[i]["score"] - iterations[i - 1]["score"])
            for i in range(1, len(iterations))
            if "score" in iterations[i] and "score" in iterations[i - 1]
        ]
        avg_gain = (sum(gains) / len(gains)) if gains else 0.0

        old = self.as_dict()
        self.record_state(context, "pre_change")
        proposed = self.propose(avg_gain)

        case_id = (context or {}).get("case_id")
        group = self._pick_group(case_id)

        if group == "B" and proposed != self.state:
            # adopt proposed knobs for the NEXT unit
            self.state = proposed
            self.bump_version()

        self._record_ab(context, old, proposed, group, avg_gain)

        if group == "B":
            self._evolution_log.append(
                {"avg_gain": round(avg_gain, 4), "old": old, "new": self.as_dict(), "timestamp": time.time()}
            )

    def _record_ab(self, context, old, new: Strategy, group: str, avg_gain: float):
        payload = {
            "test_group": group,
            "avg_gain": float(avg_gain),
            "old_strategy": old,
            "new_strategy": asdict(new),
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id,
                    role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": group},
                )
        except Exception:
            pass

    def track_section(self, case, iterations):
        """
        Optional compact attribute (avg_gain, iters, version) for dashboards/queries.
        """
        try:
            if not iterations:
                return
            gains = [
                (iterations[i]["score"] - iterations[i - 1]["score"])
                for i in range(1, len(iterations))
                if "score" in iterations[i] and "score" in iterations[i - 1]
            ]
            avg_gain = (sum(gains) / len(gains)) if gains else 0.0
            payload = {
                "avg_gain": round(avg_gain, 6),
                "iteration_count": len(iterations),
                "strategy": self.as_dict(),
                "timestamp": time.time(),
            }
            self.memory.casebooks.set_case_attr(case.id, "strategy_evolution", value_json=payload)
        except Exception:
            pass

    # ---------- A/B effectiveness ----------
    def _ab_results(self) -> List[Dict[str, Any]]:
        """
        Pull recent A/B enrollments + final scores from scorables.
        Applies history/window filters to avoid stale leakage.
        """
        now = time.time()
        results: List[Dict[str, Any]] = []
        casebooks = self.memory.casebooks.get_casebooks_by_tag(self.casebook_tag) or []

        for cb in reversed(casebooks):  # bias toward latest casebooks
            for case in reversed(self.memory.casebooks.get_cases_for_casebook(cb.id) or []):
                group, ts, perf = None, 0.0, None
                for s in self.memory.casebooks.list_scorables(case.id) or []:
                    if s.role == "strategy_ab_enroll":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            group = rec.get("test_group")
                            ts = float(rec.get("timestamp", 0.0))
                        except Exception:
                            pass
                    elif s.role == "metrics":
                        try:
                            rec = json.loads(s.text) if isinstance(s.text, str) else (s.text or {})
                            final = (rec or {}).get("final_scores") or {}
                            if "overall" in final:
                                perf = float(final["overall"])
                        except Exception:
                            pass

                if group in ("A", "B") and isinstance(perf, (int, float)):
                    if self.window_seconds is not None and (now - ts) > self.window_seconds:
                        continue
                    results.append({"group": group, "performance": perf, "timestamp": ts, "case_id": case.id})
                if len(results) >= self.history_limit:
                    break
            if len(results) >= self.history_limit:
                break

        results.sort(key=lambda r: r["timestamp"], reverse=True)
        return results[: self.history_limit]

    @staticmethod
    def _mean_stdev(xs: List[float]) -> Tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        m = sum(xs) / len(xs)
        if len(xs) == 1:
            return m, 0.0
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m, math.sqrt(max(v, 0.0))

    @staticmethod
    def _cohens_d(a: List[float], b: List[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        ma, sa = StrategyManager._mean_stdev(a)
        mb, sb = StrategyManager._mean_stdev(b)
        # pooled SD (unbiased)
        sp_num = ((len(a) - 1) * (sa ** 2) + (len(b) - 1) * (sb ** 2))
        sp_den = (len(a) + len(b) - 2)
        if sp_den <= 0 or sp_num <= 0:
            return 0.0
        sp = math.sqrt(sp_num / sp_den)
        if sp == 0:
            return 0.0
        return (mb - ma) / sp

    @staticmethod
    def _welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
        """
        Returns (t_stat, p_two_sided) using Welch's t-test (approx, no deps).
        """
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return 0.0, 1.0
        ma, sa = StrategyManager._mean_stdev(a)
        mb, sb = StrategyManager._mean_stdev(b)
        sa2, sb2 = sa ** 2, sb ** 2
        denom = math.sqrt((sa2 / na) + (sb2 / nb)) or 1e-9
        t = (mb - ma) / denom
        # Welch–Satterthwaite dof
        num = (sa2 / na + sb2 / nb) ** 2
        den = ((sa2 / na) ** 2) / (na - 1) + ((sb2 / nb) ** 2) / (nb - 1)
        dof = max(num / den, 1.0) if den > 0 else 1.0

        # two-sided p via survival of Student's t (approx using normal fallback)
        # For small samples this is rough; good enough for telemetry.
        # Convert to normal as an approximation:
        # p ≈ 2 * (1 - Φ(|t|)), Φ normal CDF
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
        return t, p

    def validate_ab(self, context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Aggregate recent A/B samples and return a small stats bundle.
        Does not mutate state; caller can decide to commit/revert elsewhere.
        """
        tests = self._ab_results()
        if not tests or len(tests) < self.min_samples:
            return None

        perf_a = [r["performance"] for r in tests if r["group"] == "A"]
        perf_b = [r["performance"] for r in tests if r["group"] == "B"]
        if not perf_a or not perf_b:
            return None

        mean_a, sd_a = self._mean_stdev(perf_a)
        mean_b, sd_b = self._mean_stdev(perf_b)
        delta = mean_b - mean_a
        d = self._cohens_d(perf_a, perf_b)
        t, p = self._welch_ttest(perf_a, perf_b)

        out = {
            "samples_A": len(perf_a),
            "samples_B": len(perf_b),
            "mean_A": mean_a,
            "mean_B": mean_b,
            "sd_A": sd_a,
            "sd_B": sd_b,
            "delta_B_minus_A": delta,
            "cohens_d": d,
            "welch_t": t,
            "p_value_two_sided": p,
            "timestamp": time.time(),
        }
        try:
            self.logger.log("StrategyAB_Validation", out)
        except Exception:
            pass
        return out

    def _evolve_strategy(self, iters: List[Dict[str, Any]], context: Optional[Dict[str, Any]]):
        if len(iters) < 2:
            # still record a point so we can analyze later
            self._record_strategy_state(context, tag="pre_change")
            return

        gains = [iters[i]["score"] - iters[i-1]["score"] for i in range(1, len(iters))]
        avg_gain = sum(gains) / len(gains) if gains else 0.0

        old_strategy = {
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
        }

        # always record the current state (for later comparison)
        self._record_strategy_state(context, tag="pre_change")

        # propose (don’t apply yet)
        proposed = self._propose_strategy_changes(avg_gain)

        # A/B enroll for *next* work unit
        #  - A keeps current knobs
        #  - B uses proposed knobs
        if random.random() < 0.5:
            # switch to proposed for the next work unit
            self.strategy = proposed
            group = "B"
            # bump version only when actually switching
            self.strategy.version += 1
        else:
            group = "A"

        # record the assignment
        self._record_strategy_test(context, old_strategy=old_strategy, new_strategy=proposed, test_group=group, avg_gain=avg_gain)

        # keep the in-memory evolution event so your longitudinal view can see changes
        if group == "B":
            new_strategy = {
                "verification_threshold": self.strategy.verification_threshold,
                "skeptic_weight": self.strategy.skeptic_weight,
                "editor_weight": self.strategy.editor_weight,
                "risk_weight": self.strategy.risk_weight,
                "version": self.strategy.version,
            }
            event = {
                "avg_gain": round(avg_gain, 4),
                "change_amount": None,  # already in deltas above
                "old": old_strategy,
                "new": new_strategy,
                "iteration_count": len(iters),
                "timestamp": time.time(),
            }
            self._evolution_log.append(event)
            _logger.info(f"LfL_Strategy_Evolved(AB): {event}")
            if context is not None:
                context.setdefault("strategy_evolution", []).append(event)

    def _propose_strategy_changes(self, avg_gain: float) -> Strategy:
        """Return a *copy* of the current strategy with proposed adjustments applied."""
        proposed = Strategy(
            verification_threshold=self.strategy.verification_threshold,
            skeptic_weight=self.strategy.skeptic_weight,
            editor_weight=self.strategy.editor_weight,
            risk_weight=self.strategy.risk_weight,
            version=self.strategy.version,
        )
        change_amount = 0.06 if avg_gain < self.cfg.get("min_gain", 0.01) and avg_gain < 0.005 else 0.03
        if avg_gain < self.cfg.get("min_gain", 0.01):
            proposed.skeptic_weight = min(0.60, proposed.skeptic_weight + change_amount)
            proposed.editor_weight  = max(0.20, proposed.editor_weight - change_amount / 2)
            proposed.risk_weight    = max(0.20, proposed.risk_weight   - change_amount / 2)
        elif avg_gain > self.cfg.get("high_gain", 0.03):
            proposed.verification_threshold = max(0.80, proposed.verification_threshold - 0.01)
        return proposed

    def _record_strategy_test(self, context: Optional[Dict[str, Any]], old_strategy: Dict[str, Any],
                            new_strategy: Strategy, test_group: str, avg_gain: float) -> None:
        """Record that we entered A or B for upcoming work."""
        payload = {
            "test_group": test_group,
            "avg_gain": avg_gain,
            "old": old_strategy,
            "new": {
                "verification_threshold": new_strategy.verification_threshold,
                "skeptic_weight": new_strategy.skeptic_weight,
                "editor_weight": new_strategy.editor_weight,
                "risk_weight": new_strategy.risk_weight,
                "version": new_strategy.version,
            },
            "timestamp": time.time(),
        }
        try:
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_ab_enroll",
                    text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"group": test_group}
                )
        except Exception:
            pass


    def _record_strategy_state(self, context: Optional[Dict[str, Any]], tag: str = "pre_change") -> None:
        """Record current strategy knobs so we can compare later."""
        payload = {
            "tag": tag,
            "verification_threshold": self.strategy.verification_threshold,
            "skeptic_weight": self.strategy.skeptic_weight,
            "editor_weight": self.strategy.editor_weight,
            "risk_weight": self.strategy.risk_weight,
            "version": self.strategy.version,
            "timestamp": time.time(),
        }
        try:
            # If a current case_id is available in context, attach it; otherwise, store on the pipeline run
            case_id = (context or {}).get("case_id")
            if case_id:
                self.memory.casebooks.add_scorable(
                    case_id=case_id, role="strategy_state", text=dumps_safe(payload),
                    pipeline_run_id=(context or {}).get("pipeline_run_id"),
                    meta={"tag": tag}
                )
        except Exception:
            pass
