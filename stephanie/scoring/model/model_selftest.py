# stephanie/scoring/model/model_selftest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, List

import math
import torch


def _tstats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return {"n": 0.0}
    return {
        "n": float(x.numel()),
        "shape0": float(x.shape[0]) if x.ndim >= 1 else 0.0,
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "l2": float(torch.norm(x).item()),
        "finite_frac": float(torch.isfinite(x).float().mean().item()),
    }


@dataclass
class SelfTestResult:
    ok: bool
    name: str
    details: Dict[str, Any]


class ModelSelfTest:
    """
    Generic self-test harness for Stephanie models.
    You provide:
      - build_inputs() -> kwargs for model(...)
      - extract_debug(aux) -> dict of tensors/scalars to inspect
    """

    def __init__(
        self,
        *,
        name: str,
        build_inputs: Callable[[], Dict[str, Any]],
        extract_debug: Callable[[Any], Dict[str, Any]],
        device: str = "cpu",
        n_trials: int = 16,
        seed: int = 1337,
        warn_only: bool = True,
    ) -> None:
        self.name = name
        self.build_inputs = build_inputs
        self.extract_debug = extract_debug
        self.device = device
        self.n_trials = int(max(3, n_trials))
        self.seed = seed
        self.warn_only = warn_only

    @torch.no_grad()
    def run(self, model: torch.nn.Module) -> SelfTestResult:
        model = model.to(self.device)
        model.eval()

        torch.manual_seed(self.seed)

        scores: List[float] = []
        debug_accum: Dict[str, List[Dict[str, Any]]] = {}

        # Run multiple trials with fresh random inputs
        for i in range(self.n_trials):
            kw = self.build_inputs()
            # move tensors to device
            for k, v in kw.items():
                if torch.is_tensor(v):
                    kw[k] = v.to(self.device)
            out = model(**kw)

            # Support (score, aux) or dict-like outputs
            if isinstance(out, tuple) and len(out) == 2:
                score, aux = out
            else:
                # allow direct score tensor
                score, aux = out, {}

            # gather score stats
            s = (
                float(score.detach().float().mean().item())
                if torch.is_tensor(score)
                else float(score)
            )
            scores.append(s)

            dbg = self.extract_debug(aux)
            for key, val in dbg.items():
                if torch.is_tensor(val):
                    entry = _tstats(val)
                else:
                    entry = {"value": val}
                debug_accum.setdefault(key, []).append(entry)

        # Analyze score distribution
        s_tensor = torch.tensor(scores, dtype=torch.float32)
        s_mean = float(s_tensor.mean().item())
        s_std = float(s_tensor.std(unbiased=False).item())
        s_min = float(s_tensor.min().item())
        s_max = float(s_tensor.max().item())

        # Heuristics for “broken”
        # - near-constant output
        const_like = s_std < 1e-4
        # - saturated near 0 or near 1
        near_zero = s_mean < 1e-3 and s_max < 5e-3
        near_one = s_mean > 1.0 - 1e-3 and s_min > 1.0 - 5e-3
        # - non-finite scores
        nonfinite = not math.isfinite(s_mean) or not math.isfinite(s_std)

        ok = (
            (not nonfinite)
            and (not const_like)
            and (not near_zero)
            and (not near_one)
        )

        details: Dict[str, Any] = {
            "score": {
                "mean": s_mean,
                "std": s_std,
                "min": s_min,
                "max": s_max,
                "const_like": const_like,
                "near_zero": near_zero,
                "near_one": near_one,
                "nonfinite": nonfinite,
                "samples": scores[: min(8, len(scores))],
            },
            "debug": debug_accum,
        }

        # if warn_only, we still return ok flag but don't raise
        return SelfTestResult(ok=ok, name=self.name, details=details)


def summarize_selftest(res: SelfTestResult) -> str:
    s = res.details.get("score", {})
    lines = [
        f"[{res.name}] ok={res.ok}",
        f"  score: mean={s.get('mean'):.6f} std={s.get('std'):.6f} min={s.get('min'):.6f} max={s.get('max'):.6f}",
        f"  flags: const_like={s.get('const_like')} near_zero={s.get('near_zero')} near_one={s.get('near_one')} nonfinite={s.get('nonfinite')}",
    ]
    # Show a couple of debug keys if present
    dbg = res.details.get("debug", {})
    for k in list(dbg.keys())[:4]:
        last = dbg[k][-1]
        if "mean" in last:
            lines.append(
                f"  {k}: mean={last['mean']:.4f} std={last['std']:.4f} min={last['min']:.4f} max={last['max']:.4f} finite={last['finite_frac']:.3f}"
            )
        else:
            lines.append(f"  {k}: {last}")
    return "\n".join(lines)
