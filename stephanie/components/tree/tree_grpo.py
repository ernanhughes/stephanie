# stephanie/components/tree/tree_grpo.py
"""
# TreeGRPOAdapter — Module Guide

> A thin, drop-in wrapper around your `AgenticTreeSearch` that upgrades it to **Tree-GRPO** (tree-based group relative policy optimization) with **prefix-sharing rollouts**, **step-level (intra-tree) process credit**, **global (inter-tree) normalization**, and optional **value-scaled rewards**.

---

## What is it?

`TreeGRPOAdapter` orchestrates a *forest* of small search trees over your existing step objects (your `SolutionNode` = a full Thought-Action-Observation step).
Instead of collecting many independent (chain) rollouts, it:

* Builds **M** independent roots (prefixes you can reuse)
* Runs **L** expansion rounds; per round, per tree, samples **N** nodes to expand
* Scores branches with your existing metric
* Computes **intra-tree** (sibling) and **inter-tree** (global) advantages
* (Optionally) **value-scales** advantages (RLEV-style) using a single scalar in context
* Emits a **training batch** you can feed into your policy updater (GRPO/DPO/GILD/Q-MAX)

Visually:

```
M roots
  ├─ expand round 1: pick N nodes per tree → branch
  └─ expand round 2: pick N nodes per tree → branch ...
       ↓
 more samples under same budget (prefix sharing) + step-level credit signals
```

---

## Why use it?

* **More samples per budget**: Shared prefixes let you gather ~1.5× more rollouts under the same token/tool limits (typical).
* **Process supervision for free**: Intra-tree sibling comparisons convert sparse outcome rewards into **step-level** signals.
* **Stable updates**: Inter-tree normalization (global z-score) gives a consistent baseline across trees.
* **Value awareness (optional)**: Multiply advantages by a clipped scale derived from a *value* label (e.g., knowledge importance).

Use this when you want to *train* or *tune* the agent’s decision policy—not just search for one good answer.

---

## Key ideas (1-minute refresher)

* **Intra-tree advantage**: Within a parent, a child’s advantage is `reward(child) − mean(reward(siblings))` (or sibling z-score).
* **Inter-tree advantage**: Across all branches in the forest, compute a **global z-score** to stabilize updates.
* **Total advantage**: `A_tree = A_intra + A_inter`.
* **Value scaling (RLEV)**: `scale = 1 + min(alpha * v, 1)`, with `v∈[0,1]`. Final `A = A_tree * scale`. Set `alpha=0` to disable.

---

## What it depends on

* Your **existing** `AgenticTreeSearch` (or equivalent) exposing:

  * `plan_generator` with `draft_plan/improve_plan/debug_plan`
  * Internal helpers: `_process_plan`, `_add_node`, `_backprop`, `_update_best`
  * `metric_fn` and `metric_policy ("minimize" or "maximize")`
  * `nodes_by_id` map and `SolutionNode` that tracks `id`, `parent_id`, `root_id`, `depth`, `plan`, `summary`, `is_buggy`, `metric`, `node_type`, `task_description`
* (Optional) `agent.get_logprob(text, context)` to populate log-probs in the output batch

No changes are required to your search logic aside from those already present in your codebase.

---

## Public surface

### Classes

#### `TreeGRPOConfig`

Configuration for the adapter.

* `M: int = 2` — number of root trees
* `N: int = 2` — nodes sampled **per tree per round**
* `L: int = 1` — number of expansion rounds
* `scorer_name: str = "sicql"` — label for downstream consumers
* `dimensions: List[str] = ["alignment"]` — labels carried into `training_batch.meta`
* `use_zscore_intra: bool = False` — if `True`, sibling z-score (else mean-center)
* `use_zscore_inter: bool = True` — global z-score across all nodes
* `value_alpha: float = 0.0` — RLEV scaling strength; `0.0` disables
* `prefer_non_buggy: bool = True` — skip buggy nodes when sampling

#### `TreeGRPOAdapter(base: AgenticTreeSearch, cfg: TreeGRPOConfig | None = None)`

* Wraps your `AgenticTreeSearch`.
* Maintains internal lists of root ids and per-tree node ids.

**Primary method**

* `await rollout_forest(context: dict) -> dict`

  * **Input**:

    * `context["goal"]["goal_text"]` (required): string task description
    * `context["value"]` (optional): float in `[0,1]` for value scaling
    * Any other fields needed by your agent/planner
  * **Output** (keys):

    * `root_ids: List[str]`
    * `nodes: List[ {id,parent_id,root_id,depth,sibling_index,metric,type} ]`
    * `rewards: Dict[node_id, float]`
    * `adv_intra: Dict[node_id, float]`
    * `adv_inter: Dict[node_id, float]`
    * `advantages: Dict[node_id, float]`  (combined)
    * `training_batch: { "items": [...], "meta": {...} }`

      * Each `item` includes: `node_id,parent_id,root_id,depth,type,plan,reward,advantage,logprob?`

---

## How it works (under the hood)

1. **Rooting (`M`)**
   Creates `M` independent root nodes by `draft_plan(...)` + `_process_plan(..., parent=None, node_type="draft")`.

2. **Expansion (`L × N`)**
   For each round:

   * For each tree:

     * Collect *candidate* nodes (non-buggy by default).
     * Sample up to `N` nodes using simple *depth-spread* (covers shallow & deep prefixes).
     * For each sampled node:

       * If buggy → `debug_plan`, else → `improve_plan`.
       * `_process_plan(...)` to create the child, then `_add_node`, `_backprop`, `_update_best`.

3. **Scoring**
   Reward = `metric_fn(node.metric)` with negation if `metric_policy == "minimize"`.

4. **Advantages**

   * **Intra:** within each parent’s children, either mean-centered or sibling z-score.
   * **Inter:** global z-score across all nodes in the forest.
   * Sum to get `A_tree`.

5. **Value scaling** (optional)
   Reads `context["value"]` → `scale = 1 + min(alpha * value, 1)` → `A = A_tree * scale`.

6. **Batch**
   Emits a training batch of items that include `plan`, `reward`, `advantage`, and optional `logprob`.

---

## Typical usage

```python
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig

ats = AgenticTreeSearch(agent, max_iterations=0)  # we drive expansions via adapter
cfg = TreeGRPOConfig(M=3, N=2, L=2, use_zscore_intra=False, use_zscore_inter=True, value_alpha=0.5)

adapter = TreeGRPOAdapter(ats, cfg)

context = {
    "goal": {"goal_text": "Synthesize Tree-GRPO integration notes for Stephanie."},
    "value": 0.8,  # optional value-weighting
    # ... any other fields your planner/agent needs
}

result = await adapter.rollout_forest(context)
batch = result["training_batch"]

# pass `batch` to your updater:
#   - GRPO-style: use `advantage` as weights on logprobs
#   - Step-DPO-style: pair sibling winners/losers using `reward` or `advantage`
```

---

## Config guidance

* **Low budget** → smaller `M`, higher `N`/`L` to maximize branch count (more prefix sharing).
* **Breadth vs depth**

  * Larger `M` explores more distinct roots (breadth).
  * Larger `L` and `N` dig deeper and widen within each root (depth + diversity).
* **Stability**

  * Keep `use_zscore_inter=True` for a global baseline.
  * `use_zscore_intra=False` (mean-center) is often stable; enable z-score if sibling counts vary widely.
* **Value scaling**

  * Start with `value_alpha = 0.25 ~ 0.5`.
  * Ensure your `context["value"]` is normalized to `[0,1]`.

---

## Integration tips

* **Rewards**: If your metric is “lower is better”, set `metric_policy="minimize"` on the base; adapter honors that.
* **Log-probs**: Implement `agent.get_logprob(plan, context)` to unlock classic GRPO/DPO losses. If not available, you can still run **preference-gradient** style updates using reward/advantage ranks.
* **Visualization**:

  * Depth → exploration length
  * Advantage → color intensity
    This pairs nicely with your VPM overlays for quick rollout diagnostics.
* **Scheduling**: Run tree-GRPO expansions on lower cost models; periodically refresh with higher-quality scorers (HRM/SICQL/MARS) to reduce drift.

---

## Troubleshooting

* **“Context must contain goal.goal_text”**
  Provide a non-empty `context["goal"]["goal_text"]`.
* **Few or no candidates**
  If most nodes are marked buggy, set `prefer_non_buggy=False` in config while stabilizing.
* **Advantages all near zero**
  Check reward variance; ensure `metric_fn` returns meaningful spread. Consider enabling sibling z-score.
* **Instability**
  Keep inter-tree z-score on; reduce `value_alpha`; verify scorer consistency.

---

## Extending it (optional)

* **Smarter sampling**: Replace `_sample_nodes` with UCB or entropy-aware scores.
* **Custom value**: Replace `_value_scale_for_nodes` to call your `KnowledgeValueEvaluator` (novelty × relevance × impact).
* **Per-depth curricula**: Early rounds favor shallow nodes, later rounds favor deeper nodes.
* **Tool-aware groups**: Compute intra-tree advantages per tool type to avoid mixing incomparable branches.

---

## FAQ

**Q: Is this only for RL?**
A: No. The emitted batch works for **DPO** (step-level via sibling wins/losses), **GRPO/GSPO** style objectives, or **GILD/Q-MAX** hybrids.

**Q: Do I need a reward model?**
A: Not necessarily. You can use your existing **goal-conditioned metric** (e.g., MARS/HRM/SICQL composite) as the reward.

**Q: How big should M, N, L be?**
A: Start with `M=2, N=2, L=1` (very cheap). If budget allows, increase `L` first, then `N`, finally `M`.

---

## Versioning notes

* **Adapter is non-breaking**: It calls existing private helpers (`_process_plan`, `_add_node`, `_backprop`, `_update_best`) and doesn’t alter your core search behavior.
* **Safe defaults**: If you skip `value_alpha`, behavior reduces to pure Tree-GRPO (no value weighting).

---

## Minimal checklist

* [ ] Provide `goal.goal_text` in `context`
* [ ] Confirm `metric_fn` is wired and returns floats
* [ ] (Optional) Implement `agent.get_logprob`
* [ ] Choose `M,N,L` within your budget
* [ ] Decide on `value_alpha` (0 to disable)

"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import (Any, Callable, DefaultDict, Dict, List, Optional)

import numpy as np

from stephanie.components.tree.core import AgenticTreeSearch, SolutionNode
from stephanie.components.tree.plan_generator import PlanGenerator


@dataclass
class TreeGRPOConfig:
    M: int = 2          # number of root trees (independent rollouts)
    N: int = 2          # nodes sampled per tree per round
    L: int = 1          # expansion rounds
    scorer_name: str = "sicql"
    dimensions: List[str] = None
    # Advantage / normalization
    use_zscore_intra: bool = False    # if False, use (r - sibling_mean)
    use_zscore_inter: bool = True     # global z-score across all branches
    # Value scaling (RLEV-style): a in [0, +], clipped to +1 gain
    value_alpha: float = 0.0          # 0 disables value scaling
    # Sampling
    prefer_non_buggy: bool = True

class TreeGRPOAdapter:
    """
    Tree-GRPO style wrapper around your AgenticTreeSearch that:
      - builds M independent root trees
      - performs L rounds; per-round samples up to N nodes per tree for expansion
      - computes intra-tree and inter-tree advantages from node rewards
      - (optionally) applies value scaling (RLEV)
      - exports per-node training tuples for downstream GRPO/DPO-style updates
    """

    def __init__(self, base: AgenticTreeSearch, cfg: Optional[TreeGRPOConfig] = None):
        self.base = base
        self.cfg = cfg or TreeGRPOConfig()
        if self.cfg.dimensions is None:
            self.cfg.dimensions = ["alignment"]

        # convenience
        self.plan_gen: PlanGenerator = self.base.plan_generator

        # storage
        self._roots: List[str] = []            # root node ids
        self._trees: Dict[str, List[str]] = {} # root_id -> list of node ids (flat index into base.tree)

    # ----------------------------- PUBLIC API ------------------------------

    async def rollout_forest(self, context: dict) -> Dict[str, Any]:
        """
        Build a small 'forest' of search trees and expand them Tree-GRPO style.
        Returns a dict with: nodes, per-node rewards, advantages, and a training batch.
        """
        # 1) Create M roots by calling the base search's initial drafting once per root
        goal = context.get("goal", {})
        task_description = goal.get("goal_text", "").strip()
        if not task_description:
            raise ValueError("Context must contain 'goal.goal_text'")

        # Generate M root drafts (prefix sharing comes from reusing these as parents)
        for _ in range(self.cfg.M):
            plan = await self.plan_gen.draft_plan(task_description, context)
            node = await self.base._process_plan(plan, parent_node=None, node_type="draft",
                                                 task_description=task_description, context=context)
            self._register_node(node)

        # 2) Perform L expansion rounds; in each round, for each tree sample up to N nodes to expand
        for _round in range(self.cfg.L):
            try:
                self.base.events.on_progress({"phase": "grpo_round_start", "round": _round})
            except Exception:
                pass
            # snapshot candidate nodes per tree
            for rid in list(self._roots):
                candidates = self._candidate_nodes(rid)
                if not candidates:
                    continue
                sample = self._sample_nodes(candidates, self.cfg.N)
                # expand each sampled node once (improve or debug if buggy; else improve)
                expand_tasks = [self._expand_one(nid, context) for nid in sample]
                for new_node in await asyncio.gather(*expand_tasks):
                    if new_node is not None:
                        self._register_node(new_node)

        # 3) Compute rewards and advantages
        nodes = [self.base.nodes_by_id[nid] for rid in self._roots for nid in self._trees[rid]]
        rewards = {n.id: self._reward_of(n) for n in nodes}

        adv_intra = self._compute_intra_tree_advantages(rewards, nodes)
        adv_inter = self._compute_inter_tree_normalization(rewards, nodes) if self.cfg.use_zscore_inter else {nid:0.0 for nid in rewards}

        # 4) Combine
        advantages = {nid: adv_intra.get(nid, 0.0) + adv_inter.get(nid, 0.0) for nid in rewards}

        # 5) Optional value scaling (RLEV-like)
        if self.cfg.value_alpha > 0.0:
            value_scale = self._value_scale_for_nodes(nodes, context)  # returns in [1,2]
            for nid in advantages:
                advantages[nid] *= value_scale.get(nid, 1.0)

        # 6) Export a training batch sketch (IDs + advantages + metadata)
        batch = self._to_training_batch(nodes, advantages, rewards, context)

        return {
            "root_ids": list(self._roots),
            "nodes": [self._node_rec(n) for n in nodes],
            "rewards": rewards,
            "advantages": advantages,
            "adv_intra": adv_intra,
            "adv_inter": adv_inter,
            "training_batch": batch,
        }

    # ---------------------------- EXPANSION LOGIC --------------------------

    async def _expand_one(self, node_id: str, context: dict) -> Optional[SolutionNode]:
        parent = self.base.nodes_by_id.get(node_id)
        if not parent:
            return None
        task_description = parent.task_description or context.get("goal", {}).get("goal_text", "")
        # heuristic: if parent is buggy, try debug; else improve
        if parent.is_buggy:
            plan = await self.plan_gen.debug_plan(parent.plan, parent.summary or "error", context)
            node = await self.base._process_plan(plan, parent, "debug", task_description, context)
        else:
            fb = parent.summary or "refine quality"
            plan = await self.plan_gen.improve_plan(parent.plan, fb, context)
            node = await self.base._process_plan(plan, parent, "improve", task_description, context)
        self.base._add_node(node)
        self.base._backprop(node)
        self.base._update_best(node)
        return node

    def _candidate_nodes(self, root_id: str) -> List[str]:
        """All non-buggy nodes in this tree are eligible candidates by default."""
        ids = self._trees.get(root_id, [])
        if not ids: return []
        if self.cfg.prefer_non_buggy:
            return [nid for nid in ids if not self.base.nodes_by_id[nid].is_buggy]
        return ids

    def _sample_nodes(self, candidates: List[str], k: int) -> List[str]:
        if k <= 0 or not candidates:
            return []
        if len(candidates) <= k:
            return list(candidates)
        # Simple diversity: sample across depths (greedy spread)
        by_depth: DefaultDict[int, List[str]] = defaultdict(list)
        for nid in candidates:
            by_depth[self.base.nodes_by_id[nid].depth].append(nid)
        # round-robin sample from shallow to deep to cover prefixes
        order = sorted(by_depth.keys())
        out = []
        idx = 0
        while len(out) < k:
            bucket = by_depth[order[idx % len(order)]]
            out.append(bucket[idx % len(bucket)])
            idx += 1
        return list(dict.fromkeys(out))[:k]  # de-dup

    # ------------------------ ADVANTAGE COMPUTATION ------------------------

    def _reward_of(self, n: SolutionNode) -> float:
        # Use the base metric_fn with None-handling already done in core
        return float(self.base.metric_fn(n.metric))

    def _compute_intra_tree_advantages(self, rewards: Dict[str, float], nodes: List[SolutionNode]) -> Dict[str, float]:
        """
        For each parent, compare siblings; advantage = r - sibling_mean (or z-score if enabled).
        Leaves get compared within their sibling group; roots use their group mean (all roots).
        """
        by_parent: DefaultDict[Optional[str], List[str]] = defaultdict(list)
        for n in nodes:
            by_parent[n.parent_id].append(n.id)

        adv: Dict[str, float] = {}
        for parent_id, group in by_parent.items():
            r = np.array([rewards[g] for g in group], dtype=float)
            if len(r) == 1:
                adv[group[0]] = 0.0
                continue
            if self.cfg.use_zscore_intra:
                mu, sd = float(r.mean()), float(r.std(ddof=1) or 1.0)
                z = (r - mu) / sd
                for nid, zval in zip(group, z):
                    adv[nid] = float(zval)
            else:
                mu = float(r.mean())
                for nid in group:
                    adv[nid] = float(rewards[nid] - mu)
        return adv

    def _compute_inter_tree_normalization(self, rewards: Dict[str, float], nodes: List[SolutionNode]) -> Dict[str, float]:
        """
        Global baseline across ALL nodes from ALL trees (z-score).
        """
        arr = np.array([rewards[n.id] for n in nodes], dtype=float)
        mu, sd = float(arr.mean()), float(arr.std(ddof=1) or 1.0)
        return {n.id: float((rewards[n.id] - mu) / sd) for n in nodes}

    # ------------------------- VALUE SCALING (RLEV) ------------------------

    def _value_scale_for_nodes(self, nodes: List[SolutionNode], context: dict) -> Dict[str, float]:
        """
        Returns per-node multiplicative scale in [1, 2], where scale = 1 + min(alpha * v, 1).
        v ∈ [0,1] should reflect 'goal importance' or 'knowledge value'.
        For now: read from context.get("value", 1.0) (already normalized), or 0.0 default.
        You can replace this with your KnowledgeValueEvaluator.
        """
        alpha = float(self.cfg.value_alpha)
        v = float(context.get("value", 0.0))
        scale = 1.0 + min(alpha * max(0.0, min(1.0, v)), 1.0)
        return {n.id: scale for n in nodes}

    # -------------------------- TRAINING BATCH -----------------------------

    def _to_training_batch(
        self,
        nodes: List[SolutionNode],
        advantages: Dict[str, float],
        rewards: Dict[str, float],
        context: dict,
    ) -> Dict[str, Any]:
        """
        Build a lightweight batch for GRPO/DPO-style updates.
        We include: node_id, parent_id, depth, plan text, reward, advantage.
        Hook 'get_logprob' can be provided via base.agent to fetch token logprobs if available.
        """
        get_logprob: Optional[Callable[[str, dict], float]] = getattr(self.base.agent, "get_logprob", None)

        items = []
        for n in nodes:
            item = {
                "node_id": n.id,
                "parent_id": n.parent_id,
                "root_id": n.root_id or n.id,
                "depth": n.depth,
                "type": n.node_type,
                "plan": n.plan,
                "reward": rewards[n.id],
                "advantage": advantages[n.id],
                # optional fields:
                "logprob": None,
            }
            if callable(get_logprob):
                try:
                    item["logprob"] = float(get_logprob(n.plan, context))
                except Exception:
                    item["logprob"] = None
            items.append(item)
        return {
            "items": items,
            "meta": {
                "scorer": self.cfg.scorer_name,
                "dimensions": self.cfg.dimensions,
                "use_zscore_intra": self.cfg.use_zscore_intra,
                "use_zscore_inter": self.cfg.use_zscore_inter,
                "value_alpha": self.cfg.value_alpha,
            },
        }

    # --------------------------- INTERNAL UTILS ----------------------------

    def _register_node(self, node: SolutionNode) -> None:
        self.base._add_node(node)
        rid = node.root_id or node.id if node.parent_id is None else (self.base.nodes_by_id[node.parent_id].root_id or node.root_id)
        if node.parent_id is None:
            # new root
            node.root_id = node.id
            rid = node.id
            self._roots.append(rid)
        self._trees.setdefault(rid, []).append(node.id)

    def _node_rec(self, n: SolutionNode) -> Dict[str, Any]:
        return dict(
            id=n.id, parent_id=n.parent_id, root_id=n.root_id, depth=n.depth,
            sibling_index=n.sibling_index, metric=n.metric, type=n.node_type
        )
