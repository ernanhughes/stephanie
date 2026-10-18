"""Unit tests for the paper-to-blog generation helpers.

These cover the deterministic logic that previously caused an off-topic,
badly-headed blog (silent cross-paper fallback, sentence headings, duplicate
sections, mangled references) without needing an LLM or a database.
"""

from __future__ import annotations

from types import SimpleNamespace

from stephanie.components.information.agents.paper_blog_generator import (
    PaperBlogGeneratorAgent,
    dedupe_sections,
    normalize_arxiv_id,
    partition_sections,
    sanitize_heading,
)


def _agent():
    # Instance without running __init__ (no cfg/memory/container needed for these helpers).
    return PaperBlogGeneratorAgent.__new__(PaperBlogGeneratorAgent)


def test_normalize_arxiv_id_handles_common_variants():
    for value in (
        "2506.21734",
        "2506.21734v2",
        "arXiv:2506.21734",
        "arxiv:2506.21734v1",
        "https://arxiv.org/abs/2506.21734",
        " 2506.21734 ",
    ):
        assert normalize_arxiv_id(value) == "2506.21734"
    assert normalize_arxiv_id(None) == ""
    assert normalize_arxiv_id("") == ""
    # Distinct papers stay distinct.
    assert normalize_arxiv_id("2506.21734") != normalize_arxiv_id("2505.04993")


def test_sanitize_heading_turns_sentences_into_headings():
    # A whole sentence is reduced to its first sentence, not echoed verbatim.
    out = sanitize_heading(
        "Vision-Language Models (VLMs) remain constrained to reasoning within "
        "a fixed modality. This paper changes that."
    )
    assert out == "Vision-Language Models (VLMs) remain constrained to reasoning within a fixed modality"

    # Markdown/quotes are stripped and the result is single-line.
    assert sanitize_heading("## **Hello** `World`\n\nsecond line") == "Hello World"

    # Long headings are cut on a word boundary and never end on punctuation.
    long = "word " * 40
    trimmed = sanitize_heading(long, max_chars=30)
    assert len(trimmed) <= 30
    assert not trimmed.endswith(("-", ":", ",", ".", " "))
    assert not trimmed.endswith("wor")  # not cut mid-word

    # Empty input yields the fallback.
    assert sanitize_heading("   ", fallback="Section 3") == "Section 3"


def test_dedupe_sections_removes_repeats_and_near_repeats():
    a = SimpleNamespace(title="Chain of Thought Prompting", text="A" * 500, section_index=0)
    dup = SimpleNamespace(title="Chain of Thought Prompting", text="B" * 500, section_index=1)
    near = SimpleNamespace(title="Chain of Thought Prompting Improves Reasoning", text="C" * 500, section_index=2)
    other = SimpleNamespace(title="Universal Transformer Architecture", text="D" * 500, section_index=3)

    kept = dedupe_sections([a, dup, near, other])
    titles = [s.title for s in kept]
    assert "Chain of Thought Prompting" in titles
    assert "Universal Transformer Architecture" in titles
    assert len(kept) == 2


def test_section_heading_prefers_real_headings_over_generated_sentences():
    agent = _agent()
    with_meta = SimpleNamespace(
        meta={"heading": "Method"},
        title="We propose a method that does many things. It is long and sentence like.",
        section_index=0,
    )
    assert agent._section_heading(with_meta) == "Method"

    no_meta = SimpleNamespace(meta={}, title="A real short heading", section_index=4)
    assert agent._section_heading(no_meta) == "A real short heading"

    empty = SimpleNamespace(meta={}, title="", section_index=4)
    assert agent._section_heading(empty) == "Section 5"


def test_force_heading_overrides_model_heading_and_adds_source_label():
    agent = _agent()
    body = "## A sentence the model wrongly emitted\n\nThe real body text."
    out = agent._force_heading(body, title="Method", level=3, source_note="Other Paper (2505.04993)")
    lines = out.splitlines()
    assert lines[0] == "### Method"
    assert "*Source: Other Paper (2505.04993).*" in out
    # The model's own heading is gone; the body survives.
    assert "A sentence the model wrongly emitted" not in out
    assert "The real body text." in out


def test_partition_sections_scopes_root_and_labels_related():
    root = SimpleNamespace(paper_arxiv_id="2506.21734", title="Root A", text="x")
    root_v = SimpleNamespace(paper_arxiv_id="arXiv:2506.21734v2", title="Root B", text="y")
    other = SimpleNamespace(paper_arxiv_id="2505.04993", title="Other", text="z")
    no_id = SimpleNamespace(paper_arxiv_id=None, title="No id", text="w")

    root_sections, related = partition_sections(
        [root, root_v, other, no_id], "2506.21734"
    )
    assert [s.title for s in root_sections] == ["Root A", "Root B"]
    # Other papers AND id-less sections are related (never silently "root").
    assert [s.title for s in related] == ["Other", "No id"]


def test_partition_sections_repairs_sections_mislabeled_with_the_root_id():
    # Real pipeline artefact: every section is stamped with the ROOT id, but the
    # namespaced section_id proves which paper it came from.
    root = SimpleNamespace(paper_arxiv_id="2506.21734", section_id="2506.21734::sec-0", title="Root", text="a")
    bacteria = SimpleNamespace(paper_arxiv_id="2506.21734", section_id="2412.07423::sec-12", title="Bacteria", text="b")
    sudoku = SimpleNamespace(paper_arxiv_id="2506.21734", section_id="2505.16135::sec-3", title="Sudoku", text="c")

    root_sections, related = partition_sections([root, bacteria, sudoku], "2506.21734")
    assert [s.title for s in root_sections] == ["Root"]
    assert [s.title for s in related] == ["Bacteria", "Sudoku"]


def test_reference_extraction_groups_wrapped_lines():
    agent = _agent()
    ref_text = (
        "References\n"
        "[1] Alice Smith, Bob Jones. A study of things. In Proceedings of\n"
        "the Conference on Stuff, 2024.\n"
        "[2] Carol Ng. Another title. arXiv preprint arXiv:2505.04993, 2025.\n"
    )
    sections = [SimpleNamespace(title="References", text=ref_text, summary=None)]
    refs = agent._extract_reference_lines_from_sections(sections)
    assert len(refs) == 2
    assert refs[0].startswith("[1] Alice Smith")
    # The wrapped continuation is part of entry 1, not a separate/mangled entry.
    assert "Conference on Stuff, 2024." in refs[0]
    assert refs[1].startswith("[2] Carol Ng")
