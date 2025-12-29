# stephanie/components/arena/blog/paper_blog_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.components.arena.blog.section_runner import \
    ArenaBlogSectionRunner
from stephanie.components.arena.blog.synthesizer import SectionSynthesizer
from stephanie.components.arena.sources.paper_sections import (
    PaperSectionCandidateSource, PaperSectionRef)


@dataclass
class OutlineSection:
    heading: str
    prompt: str   # this becomes problem_text


class PaperBlogArenaPipeline:
    """
    Single responsibility:
      outline + paper sections -> blog markdown (+ provenance)
    """

    def __init__(
        self,
        *,
        candidate_source: PaperSectionCandidateSource,
        section_runner: ArenaBlogSectionRunner,
        synthesizer: SectionSynthesizer,
        logger: Any = None,
    ):
        self.candidate_source = candidate_source
        self.section_runner = section_runner
        self.synthesizer = synthesizer
        self.log = logger

    async def run(
        self,
        *,
        topic: str,
        outline: List[OutlineSection],
        paper_sections: List[PaperSectionRef],
        emit: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        max_candidates: int = 32,
    ) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx["topic"] = topic

        candidates = self.candidate_source.build_candidates(paper_sections, max_candidates=max_candidates)

        sections_out: List[Dict[str, Any]] = []
        md_parts: List[str] = [f"# {topic}\n"]

        for i, sec in enumerate(outline):
            problem_text = sec.prompt or f"Write the '{sec.heading}' section for a blog post about {topic}."

            sel = await self.section_runner.select_supports(
                problem_text=problem_text,
                candidates=candidates,
                context=ctx,
                emit=emit,
                run_meta={"section_idx": i, "heading": sec.heading, "topic": topic},
            )

            final_text = await self.synthesizer.synthesize(
                problem_text=problem_text,
                supports=sel["supports"],
                context=ctx | {"heading": sec.heading},
            )

            md_parts.append(f"## {sec.heading}\n\n{final_text}\n")

            sections_out.append(
                {
                    "heading": sec.heading,
                    "problem_text": problem_text,
                    "final_text": final_text,
                    "winner": sel["winner"],
                    "supports": sel["supports"],  # provenance for citations / audit
                }
            )

        return {
            "topic": topic,
            "markdown": "\n".join(md_parts).strip(),
            "sections": sections_out,
        }
