# tools/prompt_echo_worker.py
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import Any, Dict

# --- Windows fix: force selector loop so pyzmq asyncio works BEFORE importing any zmq.asyncio
if sys.platform.startswith("win"):
    import asyncio as _asyncio
    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus
from stephanie.services.bus.zmq_broker import ZmqBrokerGuard

log = logging.getLogger("echo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

SUBMIT = "prompts.submit"
RESULTS_WC = "results.prompts.>"

# --- simple pending map for topic-style result collection
_pending: Dict[str, asyncio.Future] = {}


# ---- Worker handler: simulates your agent/LLM --------------------------------
async def on_prompt(payload: Dict[str, Any]):
    """Handle 'prompts.submit' jobs. Echo back a result after a tiny delay."""
    job_id = payload.get("job_id") or uuid.uuid4().hex
    prompt = payload.get("prompt", "")
    result_subject = payload.get("result_subject")

    log.info(f"[Worker] received job_id={job_id} prompt={prompt!r} result_subject={result_subject}")

    # Simulate doing some work (e.g., LLM call)
    await asyncio.sleep(0.1)

    resp = {
        "job_id": job_id,
        "status": "ok",
        "text": f"ECHO: {prompt}",
        "meta": {"worker": "prompt_echo_worker"},
    }
    # IMPORTANT:
    # Our ZMQ adapter auto-publishes to 'result_subject' if present when it sees we return a dict.
    # If you wanted to publish explicitly, you could inject the bus and call bus.publish(result_subject, resp).
    return resp


# ---- Result listener: resolves futures for topic-style test -------------------
async def on_result(payload: Dict[str, Any]):
    jid = payload.get("job_id")
    fut = _pending.get(jid)
    if fut and not fut.done():
        fut.set_result(payload)
    log.info(f"[Client] result received for job_id={jid}: {payload}")


# ---- Round-trip tests ---------------------------------------------------------
async def test_topic_roundtrip(bus: HybridKnowledgeBus) -> None:
    """Publish with result_subject and await via wildcard subscription."""
    # Subscribe once to results wildcard (idempotent under our adapter)
    await bus.subscribe(RESULTS_WC, on_result)
    await asyncio.sleep(0.05)  # small yield to ensure sub is active

    job_id = uuid.uuid4().hex
    result_subject = f"results.prompts.{job_id}"
    payload = {"job_id": job_id, "prompt": "Hello via TOPIC", "result_subject": result_subject}

    # Prepare future and publish
    fut = asyncio.get_event_loop().create_future()
    _pending[job_id] = fut

    log.info(f"[Client] TOPIC send job_id={job_id} → {SUBMIT} ; expecting on {result_subject}")
    await bus.publish(SUBMIT, payload)

    # Wait for result from results wildcard handler
    res = await asyncio.wait_for(fut, timeout=5.0)
    log.info(f"[Client] TOPIC roundtrip OK job_id={job_id} -> {res}")
    _pending.pop(job_id, None)


async def test_rpc_roundtrip(bus: HybridKnowledgeBus) -> None:
    """Request/reply direct RPC (no result_subject)."""
    payload = {"prompt": "Hello via RPC"}
    log.info(f"[Client] RPC send → {SUBMIT}")
    res = await bus.request(SUBMIT, payload, timeout=5.0)
    if not res:
        raise RuntimeError("RPC roundtrip failed: no response")
    log.info(f"[Client] RPC roundtrip OK -> {res}")


# ---- Main orchestration -------------------------------------------------------
async def main():
    # Ensure local ZMQ broker is running (no-op if already up)
    await ZmqBrokerGuard.ensure_started()

    # Build Hybrid bus configured for ZMQ
    bus = HybridKnowledgeBus(
        {
            "enabled": True,
            "backend": "zmq",        # <-- use our ZMQ backend
            "fallback": "none",      # no fallback; we WANT to test ZMQ
            "stream": "stephanie",   # ignored by ZMQ; kept for compat
            "connect_timeout_s": 3.0,
        },
        logger=log,
    )

    # Connect + ready
    ok = await bus.connect()
    log.info(f"[HybridBus] connect -> {ok} (backend={bus.get_backend()})")
    ok_ready = await bus.wait_ready(2.0)
    log.info(f"[HybridBus] wait_ready -> {ok_ready} (backend={bus.get_backend()})")

    # Subscribe the worker to prompts
    log.info(f"[Worker] subscribing to {SUBMIT}")
    await bus.subscribe(SUBMIT, on_prompt)

    # Give the worker loop a tick to attach
    await asyncio.sleep(0.05)

    # Run both round-trips
    await test_topic_roundtrip(bus)
    await test_rpc_roundtrip(bus)

    # Health snapshot
    health = bus.health_check()
    log.info(f"[Health] {health}")

    # Done
    await bus.close()
    log.info("[Done] ZMQ end-to-end test completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
