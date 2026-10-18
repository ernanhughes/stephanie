"""Bounded live inference check through Hydra and the existing LLMService.

No model call or database connection occurs unless ``--execute`` is provided.
The test uses the configured SIS model and does not write a ChatStore row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from stephanie.core.logging.json_logger import JSONLogger
from stephanie.memory.memory_tool import MemoryTool
from stephanie.orm.base import database_url_from_config
from stephanie.services.registry_loader import load_services_profile
from stephanie.services.service_container import ServiceContainer


async def run_smoke(cfg, *, timeout: int, max_tokens: int) -> dict:
    memory = None
    container = None
    try:
        logger = JSONLogger("logs/stn02_model_smoke.jsonl")
        memory = MemoryTool(cfg=cfg, logger=logger)
        container = ServiceContainer(cfg=cfg, memory=memory, logger=logger)
        load_services_profile(
            container,
            cfg=cfg,
            memory=memory,
            logger=logger,
            profile_path="config/services/default.yaml",
        )
        llm = container.get("llm")
        model = dict(cfg.chat_runtime.model)
        model["timeout"] = timeout
        model["max_tokens"] = max_tokens
        result = await llm.achat(
            [{
                "role": "system",
                "content": "/no_think You are a model connectivity check. Follow the user's exact output request.",
            }, {
                "role": "user",
                "content": "Say only OK.",
            }],
            context={"purpose": "stn02_bounded_model_smoke"},
            agent_cfg={"name": "stn02_model_smoke", "model": model},
            agent_name="stn02_model_smoke",
            add_to_history=False,
        )
        answer = str(result.get("text") or "").strip()
        if not answer:
            raise RuntimeError("Configured model returned an empty response")
        return {
            "provider": result.get("provider"),
            "model": result.get("model_name"),
            "response_id": result.get("response_id"),
            "usage": result.get("usage"),
            "latency_ms": result.get("latency_ms"),
            "finish_reason": result.get("finish_reason"),
            "response_preview": answer[:160],
            "database": memory.database_identity()["orm"]["database"],
        }
    finally:
        try:
            if container is not None:
                await container.shutdown()
        finally:
            if memory is not None:
                memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Call the configured model")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        print("\nNo model call or database connection performed.")
        return 0
    if not 1 <= args.timeout <= 120 or not 1 <= args.max_tokens <= 1024:
        parser.error("timeout must be 1–120 seconds and max-tokens 1–1024")

    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        version_base=None, config_dir=str((repo_root / "config").resolve())
    ):
        cfg = compose(config_name="sis")
    if not database_url_from_config(cfg.db).password:
        raise SystemExit("Set STEPHANIE_DB_PASSWORD before --execute")
    try:
        result = asyncio.run(run_smoke(cfg, timeout=args.timeout, max_tokens=args.max_tokens))
    except Exception as exc:
        from stephanie.utils.secret_utils import redact_secrets

        print(json.dumps({
            "verified": False,
            "error_type": type(exc).__name__,
            "error": redact_secrets(str(exc)),
        }, sort_keys=True))
        return 2
    print(json.dumps({"verified": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
