"""Live smoke check for the Worker free-model bridge.

Prints what the local Worker advertises and whether any free route is actually
admissible, then what the bridge would choose. Never raises for a missing
Worker — it reports and exits 0 so it can be used as a preflight.

    venv\\Scripts\\python.exe scripts\\worker_free_model_check.py
    venv\\Scripts\\python.exe scripts\\worker_free_model_check.py --probe 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stephanie.services.worker_model_bridge import (  # noqa: E402
    DEFAULT_WORKER_BASE_URL,
    WorkerModelBridge,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Worker free OpenCode model availability")
    parser.add_argument("--base-url", default=DEFAULT_WORKER_BASE_URL)
    parser.add_argument("--probe", type=int, default=3, help="max routes to probe (1..16)")
    args = parser.parse_args()

    bridge = WorkerModelBridge(base_url=args.base_url)

    print(f"Worker: {args.base_url}")
    try:
        health = bridge.health()
        print(f"  health: {json.dumps(health)[:200]}")
    except Exception as exc:
        print(f"  health: UNAVAILABLE ({exc})")
        print("\nWorker is not reachable; Stephanie will use its normal provider (Ollama).")
        return 0

    try:
        routes = bridge.discover_free_models()
    except Exception as exc:
        print(f"  discovery failed: {exc}")
        return 0

    print(f"  free routes advertised: {len(routes)}")
    for route in routes:
        print(f"    - {route.id} (data_policy={route.data_policy})")

    if not routes:
        print("\nNo free routes; Stephanie will use its normal provider (Ollama).")
        return 0

    print(f"\nProbing up to {args.probe} route(s)…")
    admissible = None
    for route in routes[: max(1, min(16, args.probe))]:
        try:
            result = bridge.probe(route.id)
            row = (result.get("results") or [{}])[0]
            obs = row.get("observation") or {}
            state = row.get("provider_state") or {}
            print(
                f"  {route.id}: status={obs.get('status')} "
                f"http={(obs.get('failure') or {}).get('http_status')} "
                f"admissible={state.get('request_admissible')}"
            )
        except Exception as exc:
            print(f"  {route.id}: probe error ({exc})")

    admissible = bridge.select_free_route(force=True)
    print()
    if admissible is None:
        print("RESULT: no admissible free route -> prefer_free falls back to Ollama.")
    else:
        print(f"RESULT: would use free route {admissible.id}")
    print(f"bridge status: {json.dumps(bridge.status(), default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
