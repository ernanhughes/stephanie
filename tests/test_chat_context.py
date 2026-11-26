from types import SimpleNamespace

from sis.services.chat_context import build_chat_messages


def test_other_project_turns_are_labelled_as_historical_after_project_switch():
    messages = build_chat_messages(
        [SimpleNamespace(role="user", text="The release is blocked", meta={"mission_project_id": "writer"})],
        "What are Jev's next steps?",
        {"system_prompt": "You are Stephanie", "max_context_chars": 2000},
        current_project_id="jev",
    )

    assert "Historical turn about Mission project writer" in messages[1]["content"]
    assert "not current state for jev" in messages[1]["content"]
