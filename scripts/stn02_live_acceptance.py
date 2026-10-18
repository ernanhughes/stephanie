"""Run a real authenticated SIS→LLMService→ChatStore conversation acceptance.

This command writes one clearly titled conversation, four messages and two
ChatTurns to the configured database. No database or provider access occurs
unless ``--execute`` is provided. Authentication credentials are generated in
memory for the isolated TestClient session and are never printed or persisted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _readback(conversation_id: int) -> int:
    if not os.environ.get("STEPHANIE_DB_PASSWORD"):
        raise SystemExit("Set STEPHANIE_DB_PASSWORD before readback")
    from hydra import compose, initialize_config_dir
    from stephanie.core.logging.json_logger import JSONLogger
    from stephanie.memory.memory_tool import MemoryTool

    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str((repo_root / "config").resolve())):
        cfg = compose(config_name="sis")
    memory = MemoryTool(cfg=cfg, logger=JSONLogger("logs/stn02_readback.jsonl"))
    try:
        conversation = memory.chats.get_conversation(conversation_id)
        messages = memory.chats.get_messages(conversation_id)
        turns = memory.chats.get_turns_for_conversation(conversation_id)
        identity = memory.database_identity()
        if (
            conversation is None
            or len(messages) != 4
            or [item.role for item in messages]
            != ["user", "assistant", "user", "assistant"]
            or len(turns) != 2
            or not identity["matches"]
        ):
            raise RuntimeError("Persisted multi-turn SIS conversation failed fresh-process readback")
        user_meta = messages[0].meta or {}
        assistant_meta = messages[-1].meta or {}
        if user_meta.get("sis_owner") != "stn02-live-acceptance":
            raise RuntimeError("Conversation owner metadata was not persisted")
        if assistant_meta.get("execution_status") != "completed":
            raise RuntimeError("Assistant completion state was not persisted")
        print("STN02_READBACK_JSON=" + json.dumps({
            "database": identity["orm"]["database"],
            "database_identity_matches": identity["matches"],
            "conversation_id": conversation_id,
            "conversation_title": conversation.title,
            "message_ids_and_order": [
                {"id": item.id, "role": item.role, "order_index": item.order_index}
                for item in messages
            ],
            "turn_ids": [turn.id for turn in turns],
            "provider": assistant_meta.get("provider"),
            "model": assistant_meta.get("model"),
            "usage": assistant_meta.get("usage"),
            "latency_ms": assistant_meta.get("latency_ms"),
            "assistant_answers_nonempty": all(bool(item.text) for item in messages if item.role == "assistant"),
            "fresh_process_readback": True,
        }, sort_keys=True))
    finally:
        memory.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create and persist a real two-turn conversation")
    parser.add_argument("--readback-conversation-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.readback_conversation_id is not None:
        return _readback(args.readback_conversation_id)
    if not args.execute:
        parser.print_help()
        print("\nNo app import, database connection, or model call performed.")
        return 0

    if not os.environ.get("STEPHANIE_DB_PASSWORD"):
        raise SystemExit("Set STEPHANIE_DB_PASSWORD before --execute")

    from sis.security import hash_password

    ephemeral_password = secrets.token_urlsafe(32)
    os.environ["SIS_AUTH_USERNAME"] = "stn02-live-acceptance"
    os.environ["SIS_AUTH_PASSWORD_HASH"] = hash_password(ephemeral_password)
    os.environ["SIS_SESSION_SECRET"] = secrets.token_urlsafe(48)
    # TestClient uses the same loopback host/origin that the local SIS entrypoint allows.
    os.environ["SIS_ALLOWED_HOSTS"] = "localhost,127.0.0.1,::1"
    os.environ["SIS_ALLOWED_ORIGINS"] = (
        "http://localhost:8000,http://127.0.0.1:8000,http://[::1]:8000"
    )
    os.environ["SIS_COOKIE_SECURE"] = "false"

    from fastapi.testclient import TestClient

    import sis.main as sis_main

    app = sis_main.app
    app.state.cfg.chat_runtime.enabled = True
    # These assertions only open the endpoint inside this one-process acceptance
    # client; the checked-in config remains false/false and still needs operator
    # verification before a real network-served app can enable chat.
    app.state.cfg.chat_runtime.readiness.model_verified = True
    app.state.cfg.chat_runtime.readiness.live_persistence_verified = True
    base_url = "http://127.0.0.1:8000"
    origin = base_url
    title = f"STN02 real model acceptance {uuid4()}"
    request_id = uuid4().hex

    with TestClient(app, base_url=base_url) as client:
        login_page = client.get("/login")
        csrf_match = re.search(r'name="_csrf" value="([^"]+)"', login_page.text)
        if login_page.status_code != 200 or csrf_match is None:
            raise RuntimeError("Could not load the SIS sign-in form")
        login = client.post(
            "/login",
            data={
                "_csrf": csrf_match.group(1),
                "username": "stn02-live-acceptance",
                "password": ephemeral_password,
                "next": "/chats",
            },
            headers={"Origin": origin},
            follow_redirects=False,
        )
        if login.status_code != 303 or not client.cookies.get("sis_session"):
            raise RuntimeError("Local SIS authentication acceptance failed")

        chats_page = client.get("/chats")
        csrf_match = re.search(
            r'<meta name="csrf-token" content="([^"]+)"', chats_page.text
        )
        if chats_page.status_code != 200 or csrf_match is None:
            raise RuntimeError("Authenticated SIS chat page did not render")
        csrf_token = csrf_match.group(1)
        headers = {"Origin": origin, "X-CSRF-Token": csrf_token}

        created = client.post(
            "/chats",
            json={"title": title},
            headers=headers,
        )
        if created.status_code != 201:
            raise RuntimeError(f"SIS conversation creation failed ({created.status_code})")
        conversation_id = created.json()["conversation"]["id"]

        response = client.post(
            f"/chats/{conversation_id}/messages",
            json={"text": "Say only OK."},
            headers={**headers, "Idempotency-Key": request_id},
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Real SIS chat execution failed: "
                f"HTTP {response.status_code}; error={response.json().get('detail')}"
            )
        response_data = response.json()
        if not response_data.get("assistant_message", {}).get("text", "").strip():
            raise RuntimeError("Real model response was empty")

        followup = client.post(
            f"/chats/{conversation_id}/messages",
            json={"text": "Say only OK once more."},
            headers={**headers, "Idempotency-Key": uuid4().hex},
        )
        if followup.status_code != 200:
            raise RuntimeError(
                "Real multi-turn SIS chat failed: "
                f"HTTP {followup.status_code}; error={followup.json().get('detail')}"
            )
        response_data = followup.json()
        if not response_data.get("assistant_message", {}).get("text", "").strip():
            raise RuntimeError("Real follow-up model response was empty")

    # Run a separate Python process after the app lifespan has closed its
    # MemoryTool/session factory to prove that the conversation survives restart.
    readback = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--readback-conversation-id", str(conversation_id)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if readback.returncode:
        raise RuntimeError("Fresh-process conversation readback failed")
    marker = next(
        (line.partition("=")[2] for line in reversed(readback.stdout.splitlines())
         if line.startswith("STN02_READBACK_JSON=")),
        None,
    )
    if not marker:
        raise RuntimeError("Fresh-process readback returned no acceptance record")
    persisted = json.loads(marker)
    print(json.dumps({
        "authenticated_sis_route": True,
        "csrf_protected_mutations": True,
        **persisted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
