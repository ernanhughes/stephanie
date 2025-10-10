# Agentic Tree Search (MCTS‑lite) — How to Use in Stephanie

This page explains where the agent lives in the repo, how to call it, how to plug in telemetry ("Arena"), and how to adapt the metric verifier for your tasks.

## Where to put this file

Place this file at:

```
/docs/agents/agentic_tree_search.md
```

Then add a link from your root `README.md` (Docs → Agents) and your `docs/README.md` if you have one.

Suggested README snippet:

```md
- [Agentic Tree Search](docs/agents/agentic_tree_search.md): LLM‑driven plan/code search with MCTS‑lite.
```

Also export the class from your agents package for easier imports by adding to `stephanie/agents/__init__.py`:

```python
from .agentic_tree_search import AgenticTreeSearch  # noqa: F401
```

---

## Overview

**AgenticTreeSearch** is a search‑over‑plans/code agent. It:

* drafts a plan from a goal (LLM),
* generates runnable code (LLM),
* executes it in an isolated temp dir (not production‑safe),
* parses stdout/stderr for a numeric metric,
* explores/improves/debugs using a **UCB1** policy (MCTS‑lite), and
* returns the best plan/code/metric found.

It only requires your `BaseAgent.llm(prompt: str) -> str`.

---

## Quickstart

```python
from stephanie.agents.agentic_tree_search import AgenticTreeSearch

# Your BaseAgent: must implement `await llm(prompt) -> str`
agent = MyOpenAIAgent(model="gpt-4o-mini")

ats = AgenticTreeSearch(
    agent=agent,
    N_init=4,
    max_iterations=120,
    time_limit=60*30,         # 30 minutes
    no_improve_patience=25,   # stop if plateau
    H_debug=0.4,
    H_greedy=0.5,             # exploit best node sometimes
)

context = {
    "goal": {"goal_text": "Train a classifier for XYZ and print val_accuracy."},
    "knowledge": [
        "Use stratified CV",
        "Try LightGBM baseline",
        "Early stopping on val loss",
    ],
}

best = await ats.run(context)
print("best metric:", best["final_solution"]["metric"])  # normalized [0,1]
print("code snippet:\n", best["final_solution"]["code"][:800])
```

> **Note:** The included executor is **not** a security boundary. Use Docker/gVisor/Firecracker/WASM for real isolation.

---

## Adapting the metric verifier

The default verifier extracts any of: `accuracy`, `val_accuracy`, `f1`, `auc`, `rmse`, `mae`, `score`, `metric`.

* Higher‑is‑better metrics are kept as‑is and clipped to `[0,1]`.
* Lower‑is‑better metrics (rmse/mae) are negated and normalized to `[0,1]` using `1/(1+abs(val))`.

### Example: parse `PASSED=17/20`

```python
import re
from stephanie.agents.agentic_tree_search import AgenticTreeSearch, OutputVerifier

class UTVerifier(OutputVerifier):
    def extract_metric(self, text: str):
        m = re.search(r"PASSED=(\d+)/(\d+)", text)
        if not m:
            return None
        got, tot = map(int, m.groups())
        return got / max(tot, 1)

ats = AgenticTreeSearch(agent, metric_fn=lambda m: 0.0 if m is None else m)
ats.verifier = UTVerifier()
```

### Example: force a specific final line `FINAL_SCORE=0.873`

Change your harness to print a single, reliable line and keep the default verifier by adding this pattern to `_METRIC_PATTERNS`.

---

## Telemetry: "Arena" live leaderboard

The agent emits events via `emit_cb(event: str, payload: dict)`. Use it to stream a live leaderboard (best nodes first), or render a tree.

### Minimal JSONL logger

```python
import json, time, aiofiles

async def arena_logger(event: str, payload: dict):
    async with aiofiles.open("arena_stream.jsonl", "a") as f:
        await f.write(json.dumps({"ts": time.time(), "event": event, **payload}) + "\n")

ats = AgenticTreeSearch(agent=agent, emit_cb=arena_logger)
```

### Simple web viewer (FastAPI + WebSocket)

Server:

```python
# arena_server.py
from fastapi import FastAPI, WebSocket
from typing import Set
import asyncio, json, time

app = FastAPI()
clients: Set[WebSocket] = set()

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    finally:
        clients.discard(ws)

async def broadcast(msg: dict):
    for ws in list(clients):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            clients.discard(ws)
```

Hook it up:

```python
from arena_server import broadcast

async def arena_ws_cb(event: str, payload: dict):
    await broadcast({"ts": time.time(), "event": event, **payload})

ats = AgenticTreeSearch(agent=agent, emit_cb=arena_ws_cb)
```

Client HTML (drop in `docs/arena.html`):

```html
<!doctype html>
<meta charset="utf-8" />
<title>Agent Arena</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; }
  table { border-collapse: collapse; width: 100%; }
  td, th { border-bottom: 1px solid #eee; padding: 6px 8px; }
  .bug { color: #c00; }
</style>
<table id="tbl"><thead>
<tr><th>id</th><th>parent</th><th>type</th><th>metric</th><th>visits</th><th>value</th><th>bug</th></tr>
</thead><tbody></tbody></table>
<script>
const ws = new WebSocket("ws://localhost:8000/ws");
const byId = {};
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.event === "node") { byId[m.id] = m; render(); }
};
function render() {
  const tbody = document.querySelector("#tbl tbody");
  tbody.innerHTML = Object.values(byId)
    .sort((a,b)=> (b.metric??0)-(a.metric??0))
    .map(n => `<tr>
      <td>${n.id.slice(0,8)}</td>
      <td>${(n.parent_id||"").slice(0,8)}</td>
      <td>${n.type}</td>
      <td>${n.metric?.toFixed?.(4) ?? ""}</td>
      <td>${n.visits ?? ""}</td>
      <td>${n.value?.toFixed?.(4) ?? ""}</td>
      <td class="bug">${n.bug?"bug":""}</td>
    </tr>`).join("");
}
</script>
```

---

## CLI harness (optional)

If you want a quick CLI entrypoint, add `cli_agentic_search.py` under `tools/`:

```python
import asyncio, json, argparse
from stephanie.agents.agentic_tree_search import AgenticTreeSearch
from my_agents import MyOpenAIAgent

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", help="Goal text for the agent")
    args = ap.parse_args()

    agent = MyOpenAIAgent()
    ats = AgenticTreeSearch(agent, N_init=4, max_iterations=60, no_improve_patience=20)
    ctx = {"goal": {"goal_text": args.goal}, "knowledge": []}
    out = await ats.run(ctx)
    print(json.dumps(out["final_solution"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

Update `pyproject.toml` to expose a console script:

```toml
[project.scripts]
steph-ats = "tools.cli_agentic_search:main"
```

---

## Safety & Ops

* **Execution sandbox**: the built‑in executor is convenient but not safe. For production, run code inside a container or microVM with: no network, read‑only FS, CPU/mem/pid ulimits, and timeouts.
* **Budgets**: use `max_iterations`, `time_limit`, and `no_improve_patience` so runs terminate predictably.
* **Determinism**: set `random_seed`, fix LLM temperature, and log seeds in Arena for reproducibility.

---

## API Reference (short)

```python
AgenticTreeSearch(
  agent: BaseAgent,
  max_iterations: int = 500,
  time_limit: int = 86400,
  N_init: int = 5,
  C_ucb: float = 1.2,
  H_debug: float = 0.5,
  H_greedy: float = 0.3,
  no_improve_patience: int = 50,
  random_seed: Optional[int] = 42,
  metric_fn: Optional[Callable[[Optional[float]], float]] = None,
  emit_cb: Optional[Callable[[str, dict], Awaitable[None]]] = None,
)
```

* `run(context)` — returns `context` with:

  * `final_solution`: `{ id, plan, code, metric, output, summary, parent_id, ... }`
  * `search_tree_size`: number of nodes generated

Helper:

```python
get_tree() -> {
  "nodes": [SolutionNode...],
  "best_id": str|None,
  "visits": {node_id: int},
  "value": {node_id: float},
}
```

---

## FAQ

**Q: Where do I put the agent code?**
`stephanie/agents/agentic_tree_search.py` (already done). Export it in `stephanie/agents/__init__.py`.

**Q: How do I add task‑specific tricks?**
Put them in `context["knowledge"]` — they’re embedded into prompts.

**Q: My metric isn’t in stdout.**
Add a pattern to `OutputVerifier._METRIC_PATTERNS` or override `extract_metric()`.

**Q: Can I stop early when score is above X?**
Yes — wrap `run()` or subclass and break in `_should_stop()` when `best_metric>=X`.
