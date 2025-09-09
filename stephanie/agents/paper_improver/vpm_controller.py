# stephanie/agents/paper_improver/vpm_controller.py

# vpm_controller.py
# A trend-aware controller for VPM rows that emits control signals to drive the loop.
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Tuple
import math
import time
import statistics as stats

# ========= public API =========

class Signal(Enum):
    """Controller decisions for the next step in the trajectory."""
    EDIT = auto()        # apply local, minimal diffs
    RESAMPLE = auto()    # rerun with new exemplars / different seeds
    ESCALATE = auto()    # escalate to stronger model / human checkpoint
    STOP = auto()        # stop improving (stable & above thresholds)
    SPINOFF = auto()     # fork dropped/novel content to a new artifact

@dataclass
class Thresholds:
    """Per-dimension minimum requirements (text or code)."""
    mins: Dict[str, float]                      # e.g., {"coverage":0.8, "correctness":0.75, ...}
    # Optional bands for hysteresis (require higher to STOP than to remain STOP)
    stop_margin: float = 0.02                   # extra margin to declare STOP
    edit_margin: float = 0.01                   # tolerance before EDIT triggers again

@dataclass
class Policy:
    """Control policy knobs."""
    # lookback and smoothing
    window: int = 5                              # how many recent frames to consider
    ema_alpha: float = 0.4                       # exponential smoothing for trends
    patience: int = 3                            # consecutive fails before RESAMPLE
    escalate_after: int = 2                      # consecutive RESAMPLEs before ESCALATE
    # novelty → spinoff
    spinoff_dim: str = "novelty"                 # when high novelty + low stickiness → spin off
    stickiness_dim: str = "stickiness"           # requires producer to log this; else ignored
    spinoff_gate: Tuple[float, float] = (0.75, 0.45)  # (novelty>=, stickiness<=)
    # regression guard
    max_regressions: int = 2                     # in window
    # score weighting to detect "local gaps" vs "global failure"
    local_gap_dims: List[str] = field(default_factory=lambda: ["citation_support","entity_consistency","lint_clean","type_safe"])
    # action limits
    max_steps: int = 50

@dataclass
class VPMRow:
    """Single frame from an improver (code or text)."""
    # Common
    unit: str                          # e.g., "pkg.impl:l2_normalize" or "Method"
    kind: str                          # "code" or "text"
    timestamp: float                   # epoch seconds
    dims: Dict[str, float]             # metric → value (0..1 or scalar like FKGL)
    # Optional metadata
    step_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    signal: Signal
    reason: str
    # optional “action params”, e.g., which exemplar family to try next
    params: Dict[str, Any] = field(default_factory=dict)
    # snapshot for auditability
    snapshot: Dict[str, Any] = field(default_factory=dict)

class VPMController:
    """
    Enhanced controller that:
      - applies threshold gating with hysteresis,
      - tracks rolling windows and EMAs,
      - detects stagnation vs. local gaps,
      - triggers RESAMPLE, EDIT, ESCALATE, STOP, SPINOFF,
      - provides hooks to route exemplars via a bandit.
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
    ):
        self.thr_code = thresholds_code
        self.thr_text = thresholds_text
        self.p = policy
        self.bandit_choose = bandit_choose
        self.bandit_update = bandit_update
        self.log = logger or (lambda ev, d: None)
        self.history: Dict[str, List[VPMRow]] = {}      # unit → rows
        self.resample_counts: Dict[str, int] = {}       # unit → count
        self.last_signal: Dict[str, Signal] = {}        # unit → last signal

    # ---- public method ----
    def add(self, row: VPMRow, *, candidate_exemplars: Optional[List[str]] = None) -> Decision:
        """Ingest a VPM row and decide next action."""
        h = self.history.setdefault(row.unit, [])
        h.append(row)
        if len(h) > 50:  # trim unbounded history
            self.history[row.unit] = h[-50:]

        thr = self._thresholds_for(row.kind)
        window = h[-self.p.window:] if len(h) >= 1 else h
        ema = self._ema_series(window)
        trend = self._trend(window)

        # 0) safety stop: steps or all required dims missing
        if (row.meta.get("total_steps", row.step_idx or 0) >= self.p.max_steps):
            return self._decide(row, Signal.STOP, "Max steps reached", {})

        # 1) STOP (hysteresis): all dims above mins + margin for last K frames
        if self._stable_above(window, thr, margin=thr.stop_margin):
            return self._decide(row, Signal.STOP, "Stable above thresholds", {"hysteresis": thr.stop_margin})

        # 2) SPINOFF: high novelty, low stickiness (if dims provided)
        if self._should_spinoff(row):
            return self._decide(row, Signal.SPINOFF, "High novelty with low stickiness", {"dim": self.p.spinoff_dim})

        # 3) REGRESSION guard: if too many dips in window, RESAMPLE
        if self._regressions(window) > self.p.max_regressions:
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Too many regressions", {"why": "regressions"})

        # 4) LOCAL vs GLOBAL failure
        gaps = self._gaps(row, thr)
        local_gaps = [g for g in gaps if g in self.p.local_gap_dims]
        global_fail = (len(gaps) > 0 and len(local_gaps) < len(gaps))  # several core dims below

        # 4a) LOCAL gaps → EDIT (prefer edit-policy)
        if local_gaps:
            return self._decide(row, Signal.EDIT, "Local gaps", {"gaps": local_gaps})

        # 4b) STAGNATION on core dims → RESAMPLE
        if self._stagnating(window, thr):
            params = {}
            if candidate_exemplars and self.bandit_choose:
                chosen = self.bandit_choose(candidate_exemplars)
                params["exemplar_id"] = chosen
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Stagnation on core dims", params)

        # 4c) GLOBAL failure and worsening trend → ESCALATE
        if global_fail and self._worsening(trend):
            if self.resample_counts.get(row.unit, 0) >= self.p.escalate_after:
                return self._decide(row, Signal.ESCALATE, "Global fail & worsening after resamples", {})
            else:
                self._bump_resamples(row.unit)
                return self._decide(row, Signal.RESAMPLE, "Global fail & worsening (resample first)", {})

        # 5) default: EDIT until thresholds are met or patience exceeded
        # if below mins for patience frames -> RESAMPLE
        if not self._recently_above(window, thr, patience=self.p.patience):
            self._bump_resamples(row.unit)
            return self._decide(row, Signal.RESAMPLE, "Below thresholds for patience window", {})

        # otherwise keep editing
        return self._decide(row, Signal.EDIT, "Default edit to close small gaps", {"gaps": gaps})

    # ========= internals =========

    def _thresholds_for(self, kind: str) -> Thresholds:
        return self.thr_code if kind == "code" else self.thr_text

    def _stable_above(self, window: List[VPMRow], thr: Thresholds, margin: float) -> bool:
        if not window:
            return False
        dims = thr.mins.keys()
        for w in window[-self.p.patience:]:  # require last K frames above
            for d in dims:
                v = self._val(w, d)
                if v is None:
                    return False
                if v < thr.mins[d] + margin:
                    return False
        return True

    def _recently_above(self, window: List[VPMRow], thr: Thresholds, patience: int) -> bool:
        """At least once in last K frames above all mins (prevents premature RESAMPLE)."""
        dims = thr.mins.keys()
        recent = window[-patience:]
        for w in recent:
            if all((self._val(w, d) or 0) >= thr.mins[d] for d in dims):
                return True
        return False

    def _should_spinoff(self, row: VPMRow) -> bool:
        if self.p.spinoff_dim not in row.dims or self.p.stickiness_dim not in row.dims:
            return False
        nov = row.dims[self.p.spinoff_dim]
        stk = row.dims[self.p.stickiness_dim]
        return (nov >= self.p.spinoff_gate[0]) and (stk <= self.p.spinoff_gate[1])

    def _gaps(self, row: VPMRow, thr: Thresholds) -> List[str]:
        gaps = []
        for k, t in thr.mins.items():
            v = self._val(row, k)
            if v is None:
                continue
            # hysteresis on EDIT: don't thrash if close
            if v < t - self.p.edit_margin:
                gaps.append(k)
        return gaps

    def _regressions(self, window: List[VPMRow]) -> int:
        """Count metric dips vs previous frame for core dims present in all frames."""
        if len(window) < 2:
            return 0
        regs = 0
        dims = set(window[-1].dims.keys())
        for i in range(1, len(window)):
            prev, cur = window[i-1], window[i]
            dips = sum(1 for d in dims if d in prev.dims and d in cur.dims and cur.dims[d] < prev.dims[d] - 1e-6)
            regs += (1 if dips >= max(1, len(dims)//4) else 0)
        return regs

    def _stagnating(self, window: List[VPMRow], thr: Thresholds) -> bool:
        """No improvement on core dims for 'patience' frames."""
        if len(window) < self.p.patience + 1:
            return False
        recent = window[-(self.p.patience+1):]
        core = [k for k in thr.mins.keys() if k in recent[-1].dims]
        improved = False
        for d in core:
            series = [w.dims.get(d, 0.0) for w in recent]
            if series[-1] - series[0] > 0.005:
                improved = True
                break
        return not improved

    def _trend(self, window: List[VPMRow]) -> Dict[str, float]:
        """Simple linear slope estimate per dim in window (normalized by length)."""
        if len(window) < 2:
            return {}
        n = len(window)
        t = list(range(n))
        trends = {}
        dims = set().union(*(w.dims.keys() for w in window))
        for d in dims:
            y = [w.dims.get(d, float('nan')) for w in window]
            y = [v for v in y if not math.isnan(v)]
            if len(y) < 2:
                continue
            # slope ~ last - first over n
            trends[d] = (y[-1] - y[0]) / (n - 1)
        return trends

    def _worsening(self, trend: Dict[str, float]) -> bool:
        """If majority of tracked dims have negative slope beyond small epsilon."""
        if not trend:
            return False
        vals = list(trend.values())
        neg = sum(1 for v in vals if v < -0.003)
        return neg >= max(1, len(vals)//2)

    def _ema_series(self, window: List[VPMRow]) -> Dict[str, float]:
        """Exponential moving average for info/debug; not used directly in gates."""
        if not window:
            return {}
        alpha = self.p.ema_alpha
        acc: Dict[str, float] = {}
        for w in window:
            for k, v in w.dims.items():
                if k not in acc:
                    acc[k] = v
                else:
                    acc[k] = alpha * v + (1 - alpha) * acc[k]
        return acc

    def _bump_resamples(self, unit: str):
        self.resample_counts[unit] = self.resample_counts.get(unit, 0) + 1

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

        # bandit bookkeeping: update on STOP/EDIT improvements if we know exemplar used
        eid = row.meta.get("exemplar_id")
        if eid and self.bandit_update:
            reward = self._reward(row) if signal in (Signal.STOP, Signal.EDIT) else 0.0
            try:
                self.bandit_update(eid, reward)
            except Exception:
                pass

        self.log("decision", {"unit": row.unit, "signal": signal.name, "reason": reason, **params})
        return dec

    def _reward(self, row: VPMRow) -> float:
        """Define reward for bandit as average of selected dims (can be customized)."""
        # prioritize “core” dims commonly present
        core = ["coverage","correctness","coherence","tests_pass_rate","type_safe","lint_clean"]
        vals = [row.dims[d] for d in core if d in row.dims]
        if not vals:
            vals = list(row.dims.values())
        return float(sum(vals)/len(vals)) if vals else 0.0


# ========= convenience builders =========

def default_controller() -> VPMController:
    thr_code = Thresholds(
        mins={
            "tests_pass_rate": 1.0,
            "coverage": 0.70,
            "type_safe": 1.0,
            "lint_clean": 1.0,
            "complexity_ok": 0.8
        },
        stop_margin=0.0,  # exact for code
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
    return VPMController(thr_code, thr_text, Policy())

# ========= example usage =========
if __name__ == "__main__":
    ctrl = default_controller()

    # simulate some text VPM frames
    def row(step, cov, cor, coh, cit, ent) -> VPMRow:
        return VPMRow(
            unit="Blog:Method",
            kind="text",
            timestamp=time.time(),
            step_idx=step,
            dims=dict(coverage=cov, correctness=cor, coherence=coh, citation_support=cit, entity_consistency=ent, novelty=0.7, stickiness=0.5),
            meta={"exemplar_id": "ex_pack_A"}
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
