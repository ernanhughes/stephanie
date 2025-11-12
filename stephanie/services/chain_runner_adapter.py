# stephanie/services/chain_runner_adapter.py
from __future__ import annotations

from typing import Callable, Optional

from PIL import Image

from stephanie.components.nexus.utils.visual_thought import VisualThoughtOp
from stephanie.services.chain_sampler import (ChainResult, ChainStep,
                                              default_visual_bootstrap_ops)


def run_chain_with_visual_injection(
    underlying_runner: Callable[[str, Optional[Image.Image], int], ChainResult],
    question: str,
    image: Optional[Image.Image],
    force_visual: bool,
    seed: int,
) -> ChainResult:
    """
    If force_visual, prepend a step with a deterministic visual op set.
    Assumes underlying_runner accepts (question, image, seed) and returns ChainResult.
    """
    res = underlying_runner(question, image, seed)
    if not force_visual or image is None:
        return res

    # Inject a bootstrap step if first step has no visual ops
    if not res.steps or (res.steps and not res.steps[0].visual_ops):
        ops: list[VisualThoughtOp] = default_visual_bootstrap_ops(image)
        injected = ChainStep(text="(visual-thought bootstrap)", visual_ops=ops)
        res.steps = [injected] + res.steps
        # Optional: adjust scores/metadata to reflect interleaving
        res.meta = {**res.meta, "forced_visual_bootstrap": True}
    return res


def make_run_chain_fn(underlying_runner: Callable[[str, Optional[Image.Image], int], ChainResult]):
    def run_chain_fn(question: str, image: Optional[Image.Image], force_visual: bool, seed: int) -> ChainResult:
        res = underlying_runner(question, image, seed)
        if force_visual and image is not None:
            if not res.steps or not res.steps[0].visual_ops:
                ops: list[VisualThoughtOp] = default_visual_bootstrap_ops(image)
                injected = ChainStep(text="(visual-thought bootstrap)", visual_ops=ops)
                res.steps = [injected] + res.steps
                res.meta = {**(res.meta or {}), "forced_visual_bootstrap": True}
        return res
    return run_chain_fn
