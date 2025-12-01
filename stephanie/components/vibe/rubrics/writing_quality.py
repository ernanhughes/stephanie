from __future__ import annotations

import logging
from typing import List

from stephanie.memcube.rubric import (AntiHackingCriterion, CriterionScope,
                                      CriterionType, RubricCriterion,
                                      RubricMemCube, RubricScope)

log = logging.getLogger(__name__)


def build_writing_quality_rubric() -> RubricMemCube:
    """
    Canonical rubric for evaluating research/blog writing quality.
    Targets: clarity, structure, correctness, depth, actionability, vibe.
    """

    criteria: List[RubricCriterion] = [
        RubricCriterion(
            name="clarity",
            display_name="Clarity",
            description=(
                "Rate how clear and easy to understand this writing is for a "
                "technical reader familiar with the domain but not this specific topic. "
                "Consider sentence structure, avoidance of ambiguity, and whether "
                "key terms are briefly explained when needed."
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_clarity",
        ),
        RubricCriterion(
            name="structure",
            display_name="Structure & Flow",
            description=(
                "Rate the structure and flow of the explanation. "
                "Are ideas ordered logically, with a clear beginning, middle, and end? "
                "Do transitions between paragraphs and sections feel natural and coherent?"
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_structure",
        ),
        RubricCriterion(
            name="technical_correctness",
            display_name="Technical Correctness & Faithfulness",
            description=(
                "Rate how accurate and faithful the writing is to the underlying "
                "concepts or source material. Penalize hallucinations, incorrect claims, "
                "or distortions of the method. The explanation should not introduce "
                "mechanisms the source does not support."
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_correctness",
        ),
        RubricCriterion(
            name="depth",
            display_name="Depth & Insight",
            description=(
                "Rate how well the writing goes beyond surface-level description. "
                "Does it explain why the idea matters, key tradeoffs, limitations, "
                "and how it connects to the broader system (e.g., Stephanie/Nexus/MemCubes)?"
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_depth",
        ),
        RubricCriterion(
            name="actionability",
            display_name="Actionability",
            description=(
                "Rate how actionable this writing is for an implementer. "
                "After reading, would a competent engineer know what to build, change, "
                "or test next in the system? Penalize vague advice that can't be turned "
                "into concrete steps."
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_actionability",
        ),
        RubricCriterion(
            name="vibe",
            display_name="Vibe & Voice Alignment",
            description=(
                "Rate how well the tone matches the desired 'Stephanie research partner' voice: "
                "calm, precise, non-hype, honest about limitations, and focused on system impact. "
                "Penalize over-selling, vagueness, or inconsistent tone with the rest of the blog."
            ),
            type=CriterionType.SCORE_0_100,
            scope=CriterionScope.RESPONSE,
            dimension="writing_vibe",
        ),
    ]

    anti_hacking = AntiHackingCriterion.default_for_writing()  # or .default()

    rubric = RubricMemCube(
        name="writing_quality_v1",
        version="1.0",
        scope=RubricScope.WRITING,
        description=(
            "Rubric for evaluating the quality of research/blog writing produced "
            "for the Stephanie system, with dimensions for clarity, structure, "
            "technical correctness, depth, actionability, and vibe alignment."
        ),
        criteria=criteria,
        anti_hacking=[anti_hacking],
        meta={
            "domain": "writing",
            "intended_use": "research_summaries_and_blog_posts",
        },
    )

    log.info("Built WritingQualityRubric (writing_quality_v1)")
    return rubric
