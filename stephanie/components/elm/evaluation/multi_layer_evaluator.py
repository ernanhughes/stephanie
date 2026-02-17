# elm/evaluation/multi_layer_evaluator.py

from typing import List, Dict, Any
from datetime import datetime

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult

from elm.providers.base import SignalProvider, SignalResult


class MultiLayerEvaluator:
    """
    Stephanie-compatible reducer.

    Produces:
        stephanie.scoring.score_bundle.ScoreBundle
    """

    def __init__(self, providers: List[SignalProvider]):
        self.providers = providers

    def evaluate(
        self,
        context_pack,
        plan_trace,
        output,
        **kwargs
    ) -> ScoreBundle:

        results: Dict[str, ScoreResult] = {}
        meta: Dict[str, Any] = {
            "trace_id": getattr(plan_trace, "trace_id", None),
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        for provider in self.providers:
            signal: SignalResult = provider.compute(
                context_pack=context_pack,
                plan_trace=plan_trace,
                output=output,
                **kwargs
            )

            for axis, value in signal.axis_values.items():

                dim_name = axis.value  # dimension string

                results[dim_name] = ScoreResult(
                    dimension=dim_name,
                    score=float(value) * 100.0,   # Stephanie uses 0-100 scale
                    weight=1.0,
                    rationale="ELM signal",
                    source=provider.__class__.__name__,
                    target_type="plan_trace",
                    prompt_hash=None,
                    attributes={
                        "confidence": signal.confidence,
                        "failures": signal.failure_signatures,
                    }
                )

        return ScoreBundle(results=results, meta=meta)
