"""Bounded, role-filtered chat context construction."""


def build_chat_messages(history, latest_text: str, settings) -> list[dict[str, str]]:
    """Use a bounded window of user/assistant history and the current user input."""
    system_prompt = str(settings.get("system_prompt", "")).strip()
    context_budget = max(1, int(settings.get("max_context_chars", 24000)))
    system_prompt = system_prompt[: min(2000, context_budget)]
    if len(system_prompt) + len(latest_text) > context_budget:
        raise ValueError("Message and system prompt exceed the configured context budget")

    per_message_limit = max(1, int(settings.get("max_context_message_chars", 6000)))
    history_limit = max(0, int(settings.get("history_limit", 24)))
    result = []
    used_chars = 0
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
        used_chars += len(system_prompt)

    eligible = [
        message
        for message in history
        if getattr(message, "role", None) in {"user", "assistant"}
        and getattr(message, "text", None)
    ][-history_limit:] if history_limit else []

    selected = []
    for message in reversed(eligible):
        content = _history_content(message, per_message_limit)
        if used_chars + len(latest_text) + len(content) > context_budget:
            continue
        selected.append({"role": message.role, "content": content})
        used_chars += len(content)

    result.extend(reversed(selected))
    result.append({"role": "user", "content": latest_text})
    return result


def _history_content(message, limit: int) -> str:
    text = str(message.text)
    meta = getattr(message, "meta", None) or {}
    if not isinstance(meta, dict) or meta.get("runtime") != "worker":
        return text[-limit:]
    attribution = (
        "[External Worker result; not verified by Stephanie. "
        f"status={meta.get('status', 'unknown')}; "
        f"execution_id={meta.get('worker_execution_id', 'unknown')}; "
        f"worker_request_id={meta.get('worker_request_id', 'unknown')}; "
        f"requested_model={meta.get('requested_model', 'unknown')} ]\n"
    )
    attribution = attribution[:limit]
    available = max(0, limit - len(attribution))
    return attribution + (text[-available:] if available else "")
