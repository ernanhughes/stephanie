# stephanie/trajectory/composer.py
from __future__ import annotations

def _render_span(span_turns):
    lines = []
    for t in span_turns:
        if t.user_message and t.user_message.text:
            lines.append(f"User: {t.user_message.text.strip()}")
        if t.assistant_message and t.assistant_message.text:
            lines.append(f"You: {t.assistant_message.text.strip()}")
    return "\n".join(lines)


GUIDED_PROMPT = """You are writing as Ernan. Study how the examples reason, the tone they use, and the structure they follow.
Adapt to this section without copying phrasing.

{examples}

--- SECTION ---
{section}

Write Ernan's paragraph(s) for this section. Ground to the section. Keep it tight.
Use a {target_move} structure. {image_instruction}"""

def compose_from_spans(llm, section_text: str, spans: list, max_examples: int = 3,
                       target_move: str = "VOICE", require_image_context: bool = False) -> str:
    examples = []
    for i, s in enumerate(spans[:max_examples]):
        span_text = _render_span(s['span_turns'])
        move_str = f" (Reasoning Move: {', '.join(set(s.get('moves', [])))})"
        image_str = " (Visually Grounded)" if s.get('has_image') else ""
        examples.append(f"Example {i+1}{move_str}{image_str}:\n{span_text}")
    
    examples_str = "\n\n".join(examples)
    
    # 👇 Build image instruction
    image_instruction = ""
    if require_image_context:
        image_instruction = "Incorporate visual metaphors or imagery where appropriate."
    
    prompt = GUIDED_PROMPT.format(
        examples=examples_str, 
        section=section_text,
        target_move=target_move,
        image_instruction=image_instruction
    )
    return llm.generate(prompt, max_tokens=800)