# stephanie/zeromodel/vpm_controller.py
# VPM Controller — trend-aware, goal-aware, bandit-ready control loop
from __future__ import annotations

import json
import math
import statistics as stats
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ========= public API =========

class Signal(Enum):
    EDIT = auto()        # apply local, minimal diffs
    RESAMPLE = auto()    # rerun with new exemplars / different seeds
    ESCALATE = auto()    # escalate to stronger model / human checkpoint
    STOP = auto()        # stop improving (stable & above thresholds)
    SPINOFF = auto()     # fork dropped/novel content to a new artifact
    HOLD = auto()        # hold state (cooldown / wait for external event)

@dataclass
class Thresholds:
    mins: Dict[str, float]                # {"coverage":0.8, ...}
    stop_margin: float = 0.02             # extra margin to declare STOP
    edit_margin: float = 0.01             # tolerance before EDIT re-triggers

@dataclass
class Policy:
    # windows & smoothing
    window: int = 5
    ema_alpha: float = 0.4
    edit_margin: float = 0.05   # <-- add this default
    patience: int = 3
    escalate_after: int = 2
    # oscillation & cooldowns
    oscillation_window: int = 6
    oscillation_threshold: int = 3        # direction flips to flag oscillation
    cooldown_steps: int = 1               # HOLD after RESAMPLE/ESCALATE to avoid thrash
    # novelty → spinoff
    spinoff_dim: str = "novelty"
    stickiness_dim: str = "stickiness"
    spinoff_gate: Tuple[float, float] = (0.75, 0.45)  # (novelty>=, stickiness<=)
    # regression & outlier guards
    max_regressions: int = 2
    zscore_clip_dims: List[str] = field(default_factory=lambda: ["coverage", "coherence", "correctness", "tests_pass_rate"])
    zscore_clip_sigma: float = 3.5
    # local vs global gaps
    local_gap_dims: List[str] = field(default_factory=lambda: ["citation_support","entity_consistency","lint_clean","type_safe"])
    # action cap
    max_steps: int = 50
    # goal awareness (optional; controller works without goals too)
    goal_kind: Optional[str] = None       # "text" or "code"
    goal_name: Optional[str] = None       # e.g., "academic_summary"
    goal_min_score: float = 0.75
    goal_allow_unmet: int = 0

@dataclass
class VPMRow:
    unit: str                               # e.g., "pkg.impl:l2_normalize" or "text:Section"
    kind: str                               # "code" or "text"
    timestamp: float                        # epoch seconds
    dims: Dict[str, float]                  # metric → value
    step_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    signal: Signal
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)

# ========= controller =========

class VPMController:
    """
    Goal- and trend-aware controller that:
      - gates on thresholds with hysteresis,
      - smooths noise and guards outliers,
      - detects stagnation, regressions, oscillations,
      - triggers EDIT / RESAMPLE / ESCALATE / STOP / SPINOFF / HOLD,
      - optionally consults a goal score (via injected scorer),
      - integrates with a bandit for exemplar routing,
      - persists state and accepts simple dicts (add_vpm_row) for compatibility.
    """

    def __init__(
        self,
        thresholds_code: Thresholds,
        thresholds_text: Thresholds,
        policy: Policy = Policy(),
        *,
        bandit_choose: Optional[Callable[[List[str]], str]] = None,
        bandit_update: Optional[Callable[[str, float], None]] = None,
        logger: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        goal_scorer: Optional[Callable[[str, str, Dict[str, float]], Dict[str, Any]]] = None,
        state_path: Optional[str] = None,
    ):
        self.thr_code = thresholds_code
        self.thr_text = thresholds_text
        self.p = policy
        self.bandit_choose = bandit_choose
        self.bandit_update = bandit_update
        self.log = logger or (lambda ev, d: None)
        self.goal_scorer = goal_scorer
        self.history: Dict[str, List[VPMRow]] = {}         # unit → rows
        self.resample_counts: Dict[str, int] = {}          # unit → count
        self.cooldown_until_step: Dict[str, int] = {}      # unit → step_idx boundary
        self.last_signal: Dict[str, Signal] = {}           # unit → last signal
        self.osc_dir_hist: Dict[str, List[int]] = {}       # unit → [+1/-1] changes
        self.state_path = Path(state_path) if state_path else None
        self._load_state()

    # ---- compatibility entrypoint (used by orchestrator) ----
    def add_vpm_row(self, vpm_row: Dict[str, Any], unit: str) -> Decision:
        """
        Accepts a simple dict row (as emitted by improvers) and a unit id.
        Infers kind from available dims.
        """
        kind = "code" if "tests_pass_rate" in vpm_row else "text"
        row = VPMRow(
            unit=unit,
            kind=kind,
            timestamp=time.time(),
            dims={k: float(vpm_row[k]) for k in vpm_row if isinstance(vpm_row[k], (int, float))},
            step_idx=vpm_row.get("step_idx"),
            meta=vpm_row if isinstance(vpm_row, dict) else {}
        )
        return self.add(row)

    # ---- primary entrypoint ----
    def add(self, row: VPMRow, *, candidate_exemplars: Optional[List[str]] = None) -> Decision:
        # append & clip history
        h = self.history.setdefault(row.unit, [])
        h.append(self._clipped(row))
        if len(h) > 100:
            self.history[row.unit] = h[-100:]

        thr = self._thresholds_for(row.kind)
        window = h[-self.p.window:] if h else h
        trend = self._trend(window)
        self._track_oscillation(row.unit, trend)

        # 0) max-steps stop
        if (row.meta.get("total_steps", row.step_idx or 0) >= self.p.max_steps):
            return self._decide(row, Signal.STOP, "Max steps reached", {})

        # cooldown after disruptive actions
        if self._in_cooldown(row.unit, row.step_idx):
            return self._decide(row, Signal.HOLD, "Cooldown", {"until_step": self.cooldown_until_step[row.unit]})

        # 1) STOP if stable above thresholds (hysteresis)
        if self._stable_above(window, thr, margin=thr.stop_margin):
            # optional goal gate: only STOP if goal score passes (when configured)
            if self._goal_ok_if_configured(row):
                return self._decide(row, Signal.STOP, "Stable above thresholds (goal OK)", {})
            return self._decide(row, Signal.EDIT, "Stable but goal not met yet", {"why":"goal"})

        # 2) SPINOFF: high novelty + low stickiness
        if self._should_spinoff(row):
            return self._decide(row, Signal.SPINOFF, "High novelty with low stickiness", {
                "novelty": row.dims.get(self.p.spinoff_dim),
                "stickiness": row.dims.get(self.p.stickiness_dim)
            })

        # 3) Too many regressions → RESAMPLE
        if self._regressions(window) > self.p.max_regressions:
            self._bump_resamples(row.unit)
            return self._resample(row, "Too many regressions", candidate_exemplars)

        # 4) LOCAL vs GLOBAL gaps
        gaps = self._gaps(row, thr)
        local_gaps = [g for g in gaps if g in self.p.local_gap_dims]
        global_fail = (len(gaps) > 0 and len(local_gaps) < len(gaps))

        if local_gaps:
            return self._decide(row, Signal.EDIT, "Local gaps", {"gaps": local_gaps})

        # 5) STAGNATION on core dims → RESAMPLE (then possibly ESCALATE later)
        if self._stagnating(window, thr):
            self._bump_resamples(row.unit)
            return self._resample(row, "Stagnation on core dims", candidate_exemplars)

        # 6) GLOBAL failure + worsening trend → ESCALATE (after a few resamples)
        if global_fail and self._worsening(row.unit, trend):
            if self.resample_counts.get(row.unit, 0) >= self.p.escalate_after:
                self._set_cooldown(row.unit, row.step_idx)
                return self._decide(row, Signal.ESCALATE, "Global fail & worsening after resamples", {})
            else:
                self._bump_resamples(row.unit)
                return self._resample(row, "Global fail & worsening (resample first)", candidate_exemplars)

        # 7) Below mins for patience window → RESAMPLE
        if not self._recently_above(window, thr, patience=self.p.patience):
            self._bump_resamples(row.unit)
            return self._resample(row, "Below thresholds for patience window", candidate_exemplars)

        # 8) default: EDIT small gaps
        return self._decide(row, Signal.EDIT, "Default edit to close residual gaps", {"gaps": gaps})

    # ========= internals =========

    def _thresholds_for(self, kind: str) -> Thresholds:
        return self.thr_code if kind == "code" else self.thr_text

    def _clipped(self, row: VPMRow) -> VPMRow:
        """Clip extreme outliers on selected dims using rolling z-score."""
        if not self.p.zscore_clip_dims:
            return row
        h = self.history.get(row.unit, [])
        for d in self.p.zscore_clip_dims:
            v = row.dims.get(d)
            if v is None or len(h) < 4:
                continue
            series = [w.dims.get(d) for w in h if w.dims.get(d) is not None]
            if len(series) < 4:
                continue
            mu, sd = stats.mean(series), (stats.pstdev(series) or 1e-6)
            z = abs((v - mu) / sd)
            if z > self.p.zscore_clip_sigma:
                row.dims[d] = mu + self.p.zscore_clip_sigma * (1 if v > mu else -1) * sd
        return row

    def _stable_above(self, window: List[VPMRow], thr: Thresholds, margin: float) -> bool:
        if not window:
            return False
        dims = list(thr.mins.keys())
        recent = window[-self.p.patience:]
        for w in recent:
            for d in dims:
                v = self._val(w, d)
                if v is None or v < thr.mins[d] + margin:
                    return False
        return True

    def _recently_above(self, window: List[VPMRow], thr: Thresholds, patience: int) -> bool:
        dims = list(thr.mins.keys())
        recent = window[-patience:]
        for w in recent:
            if all((self._val(w, d) or 0) >= thr.mins[d] for d in dims):
                return True
        return False

    def _should_spinoff(self, row: VPMRow) -> bool:
        nov = row.dims.get(self.p.spinoff_dim)
        stk = row.dims.get(self.p.stickiness_dim)
        if nov is None or stk is None:
            return False
        return (nov >= self.p.spinoff_gate[0]) and (stk <= self.p.spinoff_gate[1])

    def _gaps(self, row: VPMRow, thr: Thresholds) -> List[str]:
        gaps = []
        for k, t in thr.mins.items():
            v = self._val(row, k)
            if v is None:
                continue
            if v < t - self.p.edit_margin:
                gaps.append(k)
        return gaps

    def _regressions(self, window: List[VPMRow]) -> int:
        if len(window) < 2:
            return 0
        regs = 0
        dims = set(window[-1].dims.keys())
        for i in range(1, len(window)):
            prev, cur = window[i-1], window[i]
            dips = sum(1 for d in dims if d in prev.dims and d in cur.dims and cur.dims[d] < prev.dims[d] - 1e-6)
            regs += (1 if dips >= max(1, len(dims)//4) else 0)
        return regs

    def _trend(self, window: List[VPMRow]) -> Dict[str, float]:
        if len(window) < 2:
            return {}
        n = len(window)
        t = list(range(n))
        out: Dict[str, float] = {}
        dims = set().union(*(w.dims.keys() for w in window))
        for d in dims:
            y = [w.dims.get(d) for w in window if w.dims.get(d) is not None]
            if len(y) < 2:
                continue
            out[d] = (y[-1] - y[0]) / (n - 1)
        return out

    def _track_oscillation(self, unit: str, trend: Dict[str, float]):
        """Track sign flips in average slope to detect oscillations."""
        if not trend:
            return
        avg = sum(trend.values()) / max(1, len(trend))
        dir_ = 1 if avg > 0 else -1
        hist = self.osc_dir_hist.setdefault(unit, [])
        if hist and hist[-1] != dir_:
            hist.append(dir_)
        elif not hist:
            hist.append(dir_)
        if len(hist) > self.p.oscillation_window:
            self.osc_dir_hist[unit] = hist[-self.p.oscillation_window:]

    def _worsening(self, unit: str, trend: Dict[str, float]) -> bool:
        if not trend:
            return False
        vals = list(trend.values())
        neg = sum(1 for v in vals if v < -0.003)
        return neg >= max(1, len(vals)//2) or self._oscillating_unit(unit)

    def _oscillating_unit(self, unit: str) -> bool:
        hist = self.osc_dir_hist.get(unit, [])
        if len(hist) < self.p.oscillation_window:
            return False
        flips = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i-1])
        return flips >= self.p.oscillation_threshold

    def _stagnating(self, window: List[VPMRow], thr: Thresholds) -> bool:
        if len(window) < self.p.patience + 1:
            return False
        recent = window[-(self.p.patience+1):]
        core = [k for k in thr.mins.keys() if k in recent[-1].dims]
        for d in core:
            series = [w.dims.get(d, 0.0) for w in recent]
            if series[-1] - series[0] > 0.005:
                return False
        return True

    def _val(self, row: VPMRow, key: str) -> Optional[float]:
        v = row.dims.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _decide(self, row: VPMRow, signal: Signal, reason: str, params: Dict[str, Any]) -> Decision:
        dec = Decision(signal=signal, reason=reason, params=params, snapshot={
            "unit": row.unit,
            "kind": row.kind,
            "step_idx": row.step_idx,
            "dims": row.dims
        })
        self.last_signal[row.unit] = signal

        # bandit credit: on EDIT/STOP reward current exemplar; on RESAMPLE pick next
        eid = row.meta.get("exemplar_id")
        if eid and self.bandit_update and signal in (Signal.EDIT, Signal.STOP):
            try:
                self.bandit_update(eid, self._reward(row))
            except Exception:
                pass

        self._persist_state()
        self.log("decision", {"unit": row.unit, "signal": signal.name, "reason": reason, **params})
        return dec

    def _reward(self, row: VPMRow) -> float:
        core = ["coverage","correctness","coherence","tests_pass_rate","type_safe","lint_clean"]
        vals = [row.dims[d] for d in core if d in row.dims]
        if not vals:
            vals = list(row.dims.values())
        return float(sum(vals)/len(vals)) if vals else 0.0

    def _resample(self, row: VPMRow, why: str, candidates: Optional[List[str]]) -> Decision:
        params: Dict[str, Any] = {"why": why}
        if candidates and self.bandit_choose:
            try:
                chosen = self.bandit_choose(candidates)
                params["exemplar_id"] = chosen
            except Exception:
                pass
        self._set_cooldown(row.unit, row.step_idx)
        return self._decide(row, Signal.RESAMPLE, why, params)

    def _bump_resamples(self, unit: str):
        self.resample_counts[unit] = self.resample_counts.get(unit, 0) + 1

    def _set_cooldown(self, unit: str, step_idx: Optional[int]):
        if step_idx is None:
            return
        self.cooldown_until_step[unit] = step_idx + self.p.cooldown_steps

    def _in_cooldown(self, unit: str, step_idx: Optional[int]) -> bool:
        if step_idx is None:
            return False
        until = self.cooldown_until_step.get(unit)
        return until is not None and step_idx < until

    # -------- goal awareness --------

    def _goal_ok_if_configured(self, row: VPMRow) -> bool:
        if not (self.p.goal_kind and self.p.goal_name and self.goal_scorer):
            return True
        try:
            eval_ = self.goal_scorer(self.p.goal_kind, self.p.goal_name, row.dims)
            score = float(eval_.get("score", 0.0))
            unmet = eval_.get("unmet", [])
            return (score >= self.p.goal_min_score) and (len(unmet) <= self.p.goal_allow_unmet)
        except Exception:
            return True  # fail-open to not block pipeline

    # -------- persistence --------

    def _persist_state(self):
        if not self.state_path:
            return
        try:
            data = {
                "resample_counts": self.resample_counts,
                "cooldown_until_step": self.cooldown_until_step,
                "last_signal": {k: v.name for k, v in self.last_signal.items()},
                "osc_dir_hist": self.osc_dir_hist,
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_state(self):
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            self.resample_counts = data.get("resample_counts", {})
            self.cooldown_until_step = data.get("cooldown_until_step", {})
            self.last_signal = {k: Signal[v] for k, v in data.get("last_signal", {}).items()}
            self.osc_dir_hist = data.get("osc_dir_hist", {})
        except Exception:
            # ignore corrupted state
            self.resample_counts = {}
            self.cooldown_until_step = {}
            self.last_signal = {}
            self.osc_dir_hist = {}

# ========= convenience builders =========

def default_controller(state_path: Optional[str] = "./runs/vpm_state.json") -> VPMController:
    thr_code = Thresholds(
        mins={
            "tests_pass_rate": 1.0,
            "coverage": 0.70,
            "type_safe": 1.0,
            "lint_clean": 1.0,
            "complexity_ok": 0.8
        },
        stop_margin=0.0,
        edit_margin=0.0
    )
    thr_text = Thresholds(
        mins={
            "coverage": 0.80,
            "correctness": 0.75,
            "coherence": 0.75,
            "citation_support": 0.65,
            "entity_consistency": 0.80
        },
        stop_margin=0.02,
        edit_margin=0.01
    )
    return VPMController(thr_code, thr_text, Policy(), state_path=state_path)

# ========= example usage =========
if __name__ == "__main__":
    ctrl = default_controller()

    def row(step, cov, cor, coh, cit, ent) -> VPMRow:
        return VPMRow(
            unit="Blog:Method", 
            kind="text",
            timestamp=time.time(),
            step_idx=step,
            dims=dict(coverage=cov, correctness=cor, coherence=coh, citation_support=cit, entity_consistency=ent, novelty=0.78, stickiness=0.46),
            meta={"exemplar_id": "ex_pack_A", "total_steps": step}
        )

    frames = [
        row(1, 0.62, 0.60, 0.64, 0.30, 0.70),
        row(2, 0.70, 0.66, 0.70, 0.55, 0.78),
        row(3, 0.74, 0.70, 0.72, 0.60, 0.80),
        row(4, 0.81, 0.76, 0.77, 0.67, 0.85),
        row(5, 0.82, 0.77, 0.78, 0.68, 0.86),
    ]
    for f in frames:
        dec = ctrl.add(f, candidate_exemplars=["ex_pack_A","ex_pack_B","ex_pack_C"])
        print(f"step {f.step_idx}: {dec.signal.name} — {dec.reason} {dec.params}")
