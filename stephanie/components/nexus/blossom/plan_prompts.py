# stephanie/components/nexus/blossom/plan_prompts.py
from __future__ import annotations

from typing import Dict

try:
    # Prefer Jinja if available (you already use it elsewhere)
    from jinja2 import BaseLoader, Environment
    _JINJA = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
except Exception:  # pragma: no cover
    _JINJA = None


# ---------- Common snippets ----------
JSON_POLICY = """
Return ONLY valid, minified JSON. Do not include markdown, backticks, or commentary.
""".strip()

PLAN_SCHEMA = r"""
{
  "plan_id": "<uuid-or-stable-id>",
  "goal": "{{ goal_text }}",
  "assumptions": ["<short statements>"],
  "steps": [
    {
      "step_id": "s1",
      "intent": "<what this step achieves>",
      "action": "<plain-language action or tool call>",
      "inputs": {"from_context": ["..."], "literals": {"...": "..."}},
      "expected_output_format": "{{ expected_formats | default('symbolic_prompt_plan') }}",
      "success_criteria": ["<observable check>"],
      "risk_flags": ["hallucination_risk|tool_mismatch|time_cost|coverage_gap"]
    }
  ],
  "constraints": {{ constraints | tojson if constraints is not none else "[]" }},
  "tool_hints": {{ tools | tojson if tools is not none else "[]" }},
  "notes": []
}
""".strip()

SHARPEN_BLOCK = """
SHARPEN (GROWS → CRITIC → REFLECT):
- GROWS: propose 2 micro-variants, pick the better one for the final JSON.
- CRITIC: identify weak spots (ambiguity, unverifiable claims, tool mismatch); fix them.
- REFLECT: re-check against constraints and success_criteria; keep only grounded content.
Apply sharpening silently; output is the FINAL JSON only.
""".strip()


# ---------- Templates ----------
TEMPLATES: Dict[str, str] = {
    # 1) Create a first plan
    "draft_plan": """
You are a planning engine producing a compact, testable plan towards a goal.

Goal:
{{ goal_text }}

Context (optional, redacted for safety beyond plan needs):
{{ context | default("") }}

Constraints:
{{ constraints | default([]) }}

Available tools / skills (optional):
{{ tools | default([]) }}

{{ JSON_POLICY }}

JSON schema (illustrative, adapt fields as needed):
{{ PLAN_SCHEMA }}

Instructions:
- Break the work into 3–7 steps.
- Each step MUST have observable success_criteria.
- Prefer verifiable actions over vague reasoning.
- Keep assumptions explicit and minimal.
{% if sharpen %}{{ SHARPEN_BLOCK }}{% endif %}
""".strip(),

    # 2) Improve a plan with feedback & metrics deltas
    "improve_plan": """
You are revising a plan toward the same goal using feedback and metric deltas.

Goal:
{{ goal_text }}

Current plan (JSON):
{{ current_plan_json }}

Feedback signals (free-text & structured):
{{ feedback | default("") }}

Observed deltas (example keys: overall, faithfulness, coverage, novelty, risk):
{{ deltas | default({}) }}

Required outcome: an UPDATED plan JSON following this schema:
{{ PLAN_SCHEMA }}

Instructions:
- Keep step IDs stable if the intent is unchanged; add new steps with new IDs.
- Tighten success_criteria to be measurable.
- Reduce risk_flags by changing actions/inputs, not just wording.
- Keep 3–9 total steps; merge duplicates; remove dead steps.
{% if sharpen %}{{ SHARPEN_BLOCK }}{% endif %}
{{ JSON_POLICY }}
""".strip(),

    # 3) Debug a failing step given an error/log
    "debug_plan": """
You are fixing one failing step within a plan.

Goal:
{{ goal_text }}
OK
Plan (JSON):
{{ current_plan_json }}

Failing step_id:
{{ failing_step_id }}

Error / logs / traces:
{{ error_text | default("") }}

Required: output JSON with fields:
{
  "plan_patch": {
    "replace_steps": [ { /* fully-rendered steps to REPLACE the failing one (and any tightly coupled dependents) */ } ],
    "insert_steps_after": [ { "after_step_id": "s?", "step": { /* new step */ } } ],
    "remove_step_ids": ["..."]
  },
  "notes": ["why these fixes address the failure", "new risks introduced (if any)"]
}

Rules:
- Prefer minimal patch (surgical).
- Ensure downstream step inputs still exist.
- Keep success_criteria testable.
{% if sharpen %}{{ SHARPEN_BLOCK }}{% endif %}
{{ JSON_POLICY }}
""".strip(),

    # 4) Branch (Blossom): propose divergent continuations from a node/partial trace
    "branch_thoughts": """
You are generating DIVERGENT continuations ("blossom branches") from a partial reasoning trace.

Goal:
{{ goal_text }}

Partial trace (ordered steps or thoughts):
{{ partial_trace_json }}

Branch count (K):
{{ k | default(4) }}

Constraints & style guards:
{{ constraints | default([]) }}

Required output (JSON array length K):
[
  {
    "branch_id": "<id>",
    "continuation_steps": [ { "step_id": "...", "intent": "...", "action": "...", "expected_output_format": "...", "success_criteria": ["..."] } ],
    "novelty_tags": ["retrieval_shift|decomposition_swap|verification_first|speculative_probe|tool_change"],
    "risk_flags": ["hallucination_risk"|"tool_mismatch"|"over_depth"],
    "rationale": "<2-3 sentences>",
    "estimated_cost_tokens": 0
  }
]

Guidance:
- At least one branch must emphasize VERIFICATION first.
- At least one branch must try a TOOL or modality change (if tools available).
- Keep each continuation <= 3 steps.
{% if sharpen %}{{ SHARPEN_BLOCK }}{% endif %}
{{ JSON_POLICY }}
""".strip(),

    # 5) Merge (GoT): synthesize 2–5 branches into one superior plan
    "merge_branches": """
You are merging multiple candidate branches into a single superior plan that preserves strengths and removes weaknesses.

Goal:
{{ goal_text }}

Current base plan (JSON):
{{ base_plan_json }}

Candidate branches (JSON array):
{{ branches_json }}

Metrics (per-branch, optional):
{{ metrics_json | default([]) }}

Required output JSON:
{
  "merged_plan": {{ PLAN_SCHEMA }},
  "merge_notes": {
    "kept_from": {"branch_id": ["feature1","feature2"]},
    "dropped_risks": {"branch_id": ["risk1","risk2"]},
    "conflict_resolutions": ["short records of conflicts and chosen resolution"],
    "expected_lift": {"overall": 0.0, "faithfulness": 0.0, "coverage": 0.0, "novelty": 0.0}
  }
}

Rules:
- Resolve conflicts by deferring to faithfulness > coverage > novelty (priority order).
- Do not exceed 9 steps in merged_plan.
{% if sharpen %}{{ SHARPEN_BLOCK }}{% endif %}
{{ JSON_POLICY }}
""".strip(),

    # 6) Evaluate a (partial) state for scoring
    "evaluate_state": """
Score the current state (plan or partial outputs) on normalized scales in [0,1].

Goal:
{{ goal_text }}

State to evaluate (JSON):
{{ state_json }}

Dimensions (provide all even if estimated): ["overall","faithfulness","coverage","clarity","novelty","risk_inverse"]

Output JSON:
{
  "scores": {"overall": 0.0, "faithfulness": 0.0, "coverage": 0.0, "clarity": 0.0, "novelty": 0.0, "risk_inverse": 0.0},
  "rationale": "1-3 sentences linking observations to scores",
  "improvement_hints": ["concise, testable hints"]
}

Rules:
- Penalize unverifiable claims (faithfulness).
- Prefer actionable plans (clarity).
- Keep rationale short and concrete.
{{ JSON_POLICY }}
""".strip(),

    # 7) Compare two filmstrips (A vs B) with winner & reasons
    "compare_filmstrips": """
Compare two reasoning filmstrips for the same goal and pick a winner.

Goal:
{{ goal_text }}

Filmstrip A (JSON):
{{ film_a_json }}

Filmstrip B (JSON):
{{ film_b_json }}

Scoring policy (weights in [0,1], sum≈1):
{{ weights | default({"faithfulness":0.35,"coverage":0.25,"clarity":0.2,"novelty":0.1,"risk_inverse":0.1}) }}

Output JSON:
{
  "winner": "A|B",
  "scores": {
    "A": {"overall": 0.0, "faithfulness": 0.0, "coverage": 0.0, "clarity": 0.0, "novelty": 0.0, "risk_inverse": 0.0},
    "B": {"overall": 0.0, "faithfulness": 0.0, "coverage": 0.0, "clarity": 0.0, "novelty": 0.0, "risk_inverse": 0.0}
  },
  "rationale": ["3-5 bullet points referencing concrete steps"],
  "diff_summary": {"strengths_A": ["..."], "strengths_B": ["..."], "weaknesses_A": ["..."], "weaknesses_B": ["..."]}
}
{{ JSON_POLICY }}
""".strip(),
}


def render(name: str, sharpen: bool = True, **kwargs) -> str:
    """
    Render a prompt by name. `sharpen=True` appends the GROWS→CRITIC→REFLECT block.
    Usage:
        text = render("draft_plan", goal_text="...", context="...", tools=[...])
    """
    if name not in TEMPLATES:
        raise KeyError(f"Unknown plan prompt: {name}")

    data = dict(kwargs)
    data.setdefault("goal_text", kwargs.get("goal_text", "").strip())
    data.setdefault("expected_formats", kwargs.get("expected_formats", "symbolic_prompt_plan"))
    data.setdefault("constraints", kwargs.get("constraints"))
    data.setdefault("tools", kwargs.get("tools"))
    data.setdefault("sharpen", sharpen)
    data.setdefault("JSON_POLICY", JSON_POLICY)
    data.setdefault("PLAN_SCHEMA", PLAN_SCHEMA)
    data.setdefault("SHARPEN_BLOCK", SHARPEN_BLOCK)

    template = TEMPLATES[name]
    if _JINJA:
        return _JINJA.from_string(template).render(**data)
    # fallback: naive format replacement
    return template.format(**{k: (v if isinstance(v, str) else str(v)) for k, v in data.items()})


