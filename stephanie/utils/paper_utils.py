# stephanie/utils/paper_utils.py
from __future__ import annotations

SECTION_GOALS = {
    "abstract": "Produce a faithful, concise overview; no speculation; emphasize contributions and results.",
    "introduction": "Frame the problem, stakes, and novelty; motivate why it matters; minimal formulae.",
    "related work": "Map prior approaches into 2–4 buckets; contrast this paper’s edge; cite sparingly but precisely.",
    "methods": "Explain the core mechanism step-by-step; define symbols; include a minimal worked example if possible.",
    "experiments": "Summarize datasets, baselines, metrics; highlight deltas; include 1–2 ablation insights.",
    "results": "Report main numbers and what they mean; avoid over-claiming; connect to the stated goals.",
    "discussion": "Synthesize implications, limitations, and future directions; note failure modes candidly.",
    "conclusion": "Re-state contribution in one line; one practical takeaway; one open question."
}

def section_goal_text(section_name: str, paper_title: str, fallback: str | None = None) -> str:
    key = (section_name or "").strip().lower()
    hint = SECTION_GOALS.get(key, fallback or "Write an accurate, insightful, blog-friendly summary of this section.")
    return (
        f"You are an expert technical blog writer. Paper: “{paper_title}”. "
        f"Section: “{section_name}”. Goal: {hint} "
        "Keep it accurate, grounded in the section text; prefer plain language; do not invent results."
    )

def section_quality(section_name: str) -> dict:
    base = {
        "coverage": 0.80, "correctness": 0.85, "coherence": 0.80,
        "citation_support": 0.90, "readability": 0.85, "novelty": 0.75
    }
    s = (section_name or "").lower()
    # Slight, sensible nudges by section
    if s in {"methods", "method", "approach"}:
        base["coverage"] = 0.85
        base["correctness"] = 0.90
        base["readability"] = 0.83
    elif s in {"introduction"}:
        base["readability"] = 0.90
        base["coherence"] = 0.85
    elif s in {"experiments", "results"}:
        base["correctness"] = 0.90
        base["citation_support"] = 0.92
    elif s in {"related work"}:
        base["coverage"] = 0.83
        base["citation_support"] = 0.93
    return base

def build_paper_goal_text(title: str) -> str:
    return (
        f"As an expert technical blog writer, process the paper “{title}” into "
        f"section-wise, insightful, accurate blog drafts that summarize each section, "
        f"highlight contributions, limitations, and implications for practice, and "
        f"link claims to verifiable evidence."
    )

def build_paper_goal_meta(title: str, paper_id: str, domains: list[str] | None = None) -> dict:
    return {
        "type": "document_section_blog_goal",
        "paper_id": paper_id,
        "title": title,
        "audience": "experienced ML/AI practitioners",
        "tone": "succinct, precise, citation-aware",
        "quality_standards": {
            "coverage": 0.8,
            "correctness": 0.85,
            "coherence": 0.8,
            "citation_support": 0.9,
            "readability": 0.85,
            "novelty": 0.75
        },
        "blog_requirements": {
            "include": [
                "what the section claims",
                "why it matters",
                "how it compares to prior work",
                "evidence snippets or equation refs when available",
                "practical implications or example use"
            ],
            "avoid": [
                "hallucinated equations or datasets",
                "casual tone",
                "uncited strong claims"
            ]
        },
        "domains": domains or []
    }

def system_guidance_from_goal(goal_text: str, quality: dict) -> str:
    qs = ", ".join(f"{k}≥{v}" for k, v in quality.items())
    return (
        f"SYSTEM GUIDANCE:\n"
        f"- Role: {goal_text}\n"
        f"- Quality thresholds: {qs}\n"
        f"- Style: succinct, precise, expert tone; cite evidence where possible.\n"
        f"- Output JSON strictly matching signature schema."
    )


def build_section_goal_text(title: str, section_name: str) -> str:
    return (
        f"Write an insightful, evidence-grounded blog draft for the “{section_name}” "
        f"section of the paper “{title}”. Capture key claims, support with citations "
        f"from the section, and end with practical implications."
    )

def build_section_goal_meta(paper_id: str, section_name: str) -> dict:
    return {
        "type": "document_section_goal",
        "paper_id": paper_id,
        "section_name": section_name
    }
