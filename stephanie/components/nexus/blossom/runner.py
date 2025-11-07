# stephanie/components/nexus/blossom/runner.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import time
import logging

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.tree.core import AgenticTreeSearch
from stephanie.components.tree.tree_grpo import TreeGRPOAdapter, TreeGRPOConfig
from stephanie.memory.blossom_store import BlossomStore
from stephanie.scoring.scorable import Scorable
from stephanie.utils.emit_broadcaster import EmitBroadcaster

log = logging.getLogger(__name__)


@dataclass
class BlossomRunConfig:
    # Tree-GRPO forest params
    M: int = 2
    N: int = 2
    L: int = 1
    use_zscore_inter: bool = True
    use_zscore_intra: bool = False
    value_alpha: float = 0.0
    prefer_non_buggy: bool = True

    # Post-processing
    return_top_k: int = 1         # how many winning leaves to report
    sharpen_top_k: int = 0        # 0 = disabled

    # Prompt “jitter” (VPM) hinting
    enable_vpm_hint: bool = False
    vpm_op: str = "zoom_max"      # e.g., "zoom_max" (if ZeroModel supports it)


def maybe_vpm_mutate_goal(container, goal_text: str, *, op: str = "zoom_max") -> str:
    """
    Optional novelty injector: derive a tiny VPM “jitter” and append a hint to the goal.
    If ZeroModel isn't wired, return the original goal.
    """
    try:
        if not goal_text:
            return goal_text
        zm = container.get("zero_model")
        if not zm:
            return goal_text

        vpm_u8, _ = zm.vpm_from_scorable(
            Scorable(id=None, text=goal_text, target_type="document"),
            img_size=96,
        )
        used_op = op
        if hasattr(zm, "apply_visual_thought"):
            _, used_op = zm.apply_visual_thought(vpm_u8, op=op)

        hint = (
            f"\n[VisualJitterHint: applied {used_op}; prefer crisper structure, "
            f"explicit claims, and stronger evidential links.]\n"
        )
        return goal_text + hint
    except Exception:
        return goal_text


class BlossomRunnerAgent(BaseAgent):
    """
    Agentic Tree Search → Blossom graph writer.
    - Runs Tree-GRPO
    - Writes nodes/edges into SQL via BlossomStore
    - Marks winners (tags) + merges run stats into BlossomORM.stats
    - Optional sharpening with scorer-backed re-eval
    - Emits telemetry via EmitBroadcaster sinks
    """
    name = "blossom_runner"

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = BlossomRunConfig(
            M=int(cfg.get("M", 2)),
            N=int(cfg.get("N", 2)),
            L=int(cfg.get("L", 1)),
            use_zscore_inter=bool(cfg.get("use_zscore_inter", True)),
            use_zscore_intra=bool(cfg.get("use_zscore_intra", False)),
            value_alpha=float(cfg.get("value_alpha", 0.0)),
            prefer_non_buggy=bool(cfg.get("prefer_non_buggy", True)),
            return_top_k=int(cfg.get("return_top_k", 1)),
            sharpen_top_k=int(cfg.get("sharpen_top_k", 0)),
            enable_vpm_hint=bool(cfg.get("enable_vpm_hint", False)),
            vpm_op=str(cfg.get("vpm_op", "zoom_max")),
        )

        # ATS knobs (prompt-oriented search defaults)
        self.ats_cfg = {
            "N_init": cfg.get("ats_N_init", 4),
            "max_iterations": cfg.get("ats_max_iter", 80),
            "time_limit": cfg.get("ats_time_limit", 120),
            "no_improve_patience": cfg.get("ats_patience", 25),
            "H_debug": 0.0,
            "H_greedy": cfg.get("ats_H_greedy", 0.5),
            "C_ucb": cfg.get("ats_C_ucb", 1.2),
            "random_seed": cfg.get("random_seed", 42),
        }

        # Stores/services
        self.blossoms: BlossomStore = self.memory.blossoms
        self.scoring = container.get("scoring")

        # Telemetry
        self.msubj_req = "blossom.prompt.request"
        self.ats_report = "blossom.prompt.responses.available"

        # Runtime
        self.run_id: Optional[str] = None
        self._goal_text: str = ""

        # Agentic search with dual sinks (log + bus/timeline)
        self.base_search: AgenticTreeSearch = AgenticTreeSearch(
            agent=self,
            N_init=self.ats_cfg["N_init"],
            max_iterations=self.ats_cfg["max_iterations"],
            time_limit=self.ats_cfg["time_limit"],
            no_improve_patience=self.ats_cfg["no_improve_patience"],
            H_debug=self.ats_cfg["H_debug"],
            H_greedy=self.ats_cfg["H_greedy"],
            C_ucb=self.ats_cfg["C_ucb"],
            metric_fn=lambda m: 0.0 if m is None else float(m),
            emit_cb=EmitBroadcaster(self._emit_tolog, self._timeline_sink),
            random_seed=self.ats_cfg["random_seed"],
        )

        tcfg = TreeGRPOConfig(
            M=self.cfg.M,
            N=self.cfg.N,
            L=self.cfg.L,
            use_zscore_intra=self.cfg.use_zscore_intra,
            use_zscore_inter=self.cfg.use_zscore_inter,
            value_alpha=self.cfg.value_alpha,
            prefer_non_buggy=self.cfg.prefer_non_buggy,
        )
        self.adapter = TreeGRPOAdapter(self.base_search, tcfg)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal", {}) or {}
        goal_text = (goal.get("goal_text") or "").strip()
        self.run_id = context.get("pipeline_run_id")
        self._goal_text = goal_text

        # VPM “jitter” hint (optional)
        if self.cfg.enable_vpm_hint and goal_text:
            goal_text = maybe_vpm_mutate_goal(self.container, goal_text, op=self.cfg.vpm_op)
            context = {**context, "goal": {**goal, "goal_text": goal_text}}
            self._goal_text = goal_text

        seed_meta = context.get("seed", {}) or {}

        # --- create blossom episode (status = running) ---
        blossom = self.blossoms.create_blossom({
            "agent_name": self.name,
            "strategy": "tree_grpo",
            "seed_type": seed_meta.get("seed_type"),
            "seed_id": seed_meta.get("seed_id"),
            "goal_id": goal.get("id"),
            "pipeline_run_id": self.run_id,
            "status": "running",
            "params": {
                "runner": asdict(self.cfg),
                "ats": self.ats_cfg,
            },
            "extra_data": {"goal_text": goal_text},
        })

        # Seed plans if provided
        seed_plans: List[str] = []
        if "plan_text" in seed_meta:
            seed_plans = [str(seed_meta["plan_text"])]
        elif isinstance(seed_meta.get("plans"), list):
            seed_plans = [str(p) for p in seed_meta["plans"]]

        # --- roll out forest ---
        forest_out = await self.adapter.rollout_forest({**context, "seed_plans": seed_plans or None})

        # --- persist forest (two-pass to resolve parent DB IDs) ---
        ext2db: Dict[str, int] = {}
        try:
            self._persist_nodes_first_pass(blossom.id, forest_out, ext2db)
            self._persist_edges_and_parents_second_pass(blossom.id, forest_out, ext2db)
            # set root if unique root exists
            roots = [eid for eid, rec in self._by_id(forest_out).items() if not rec.get("parent_id")]
            if len(roots) == 1 and roots[0] in ext2db:
                self.blossoms.set_root(blossom.id, ext2db[roots[0]])
        except Exception as e:
            self.logger.log("BlossomPersistForestError", {"blossom_id": blossom.id, "error": str(e)})

        # --- winners ---
        top_leaves = self._topk_leaves(forest_out, k=self.cfg.return_top_k)
        winners: List[Dict[str, Any]] = []

        # optional sharpen
        sharpened: List[Optional[Dict[str, Any]]] = []
        if self.cfg.sharpen_top_k > 0 and top_leaves:
            top_paths = [self._path_for_leaf(forest_out, leaf_id) for (leaf_id, _) in top_leaves]
            sharpened = await self._sharpen_top_leaves(top_paths, context)

        for i, (leaf_id, reward) in enumerate(top_leaves):
            path_ext = self._path_for_leaf(forest_out, leaf_id)
            path_db = [ext2db.get(x) for x in path_ext if x in ext2db]
            sh = sharpened[i] if i < len(sharpened) else None

            winners.append({
                "leaf_ext_id": leaf_id,
                "leaf_db_id": ext2db.get(leaf_id),
                "reward": float(reward),
                "path_ext": path_ext,
                "path_db": path_db,
                "sharpened": sh,
            })

            # tag DB leaf as winner
            try:
                db_leaf_id = ext2db.get(leaf_id)
                if db_leaf_id:
                    node = self.blossoms.update_node(db_leaf_id, {
                        "tags": ["winner"] if not sh else ["winner", "sharpened"],
                        "scores": {"reward": float(reward)},
                        "sharpen_passes": (1 if sh else 0),
                        "sharpen_gain": (float(sh["score"]) - float(reward)) if (sh and "score" in sh) else None,
                        "sharpen_meta": sh or None,
                    })
                    # also mark PATH edges as 'select'
                    for a, b in zip(path_db, path_db[1:]):
                        if a and b:
                            self.blossoms.add_edge({
                                "blossom_id": blossom.id,
                                "src_node_id": a,
                                "dst_node_id": b,
                                "relation": "select",
                                "score": None,
                                "rationale": None,
                                "extra_data": {"selected": True},
                            })
            except Exception as e:
                self.logger.log("BlossomWinnerTagError", {"blossom_id": blossom.id, "leaf_ext": leaf_id, "error": str(e)})

        # --- finalize episode ---
        stats = {
            "num_nodes": len(forest_out.get("nodes", []) or []),
            "num_edges": len(forest_out.get("nodes", []) or []),  # approx: edges==nodes-#roots (cheap placeholder)
            "top_reward": float(top_leaves[0][1]) if top_leaves else None,
            "winners": winners,
        }
        try:
            self.blossoms.update_status(blossom.id, status="completed", stats=stats)
        except Exception as e:
            self.logger.log("BlossomCloseEpisodeError", {"blossom_id": blossom.id, "error": str(e)})

        # return compact result
        context["blossom_result"] = {
            "episode_id": blossom.id,
            "winners": winners,
            "training_batch": forest_out.get("training_batch"),  # for GRPO/DPO
        }
        return context

    # ------------------------------------------------------------------ #
    # Telemetry sinks
    # ------------------------------------------------------------------ #
    async def _timeline_sink(self, event: str, payload: Dict[str, Any]) -> None:
        """Emit to bus; never raise."""
        if not self.run_id:
            return
        try:
            if event == "node":
                node = payload.get("node", payload)
                await self.memory.bus.publish(
                    subject=self.msubj_req,
                    payload={
                        "run_id": self.run_id,
                        "node_id": node.get("id"),
                        "parent_id": node.get("parent_id"),
                        "action_type": node.get("type", "draft"),
                        "goal_text": self._goal_text,
                        "prompt_text": payload.get("prompt_text", node.get("plan")),
                        "best_metric": node.get("metric"),
                        "bug": bool(node.get("bug", False)),
                        "ts_enqueued": time.time(),
                    },
                )
            elif event == "report":
                await self.memory.bus.publish(
                    subject=self.ats_report,
                    payload={"run_id": self.run_id},
                )
        except Exception as e:
            log.warning("Blossom timeline sink failed: %s", e)

    def _emit_tolog(self, event: str, payload: Dict[str, Any]) -> None:
        """Sync logger sink; never raises."""
        self.logger.log(f"Blossom::{event}", payload)

    # ------------------------------------------------------------------ #
    # Forest persistence (two pass)
    # ------------------------------------------------------------------ #
    def _persist_nodes_first_pass(self, blossom_id: int, forest_out: Dict[str, Any], ext2db: Dict[str, int]) -> None:
        """
        Insert all nodes with no parent first; keep mapping from external (ATS) IDs to DB IDs.
        """
        nodes: List[Dict[str, Any]] = forest_out.get("nodes", []) or []
        by_id = {n["id"]: n for n in nodes}

        for rec in nodes:
            ext_id = rec.get("id")
            # fetch plan text from ATS if available
            plan_text = ""
            try:
                node_obj = self.base_search.nodes_by_id.get(ext_id)
                plan_text = getattr(node_obj, "plan", "") or ""
            except Exception:
                pass

            obj = self.blossoms.add_node({
                "blossom_id": blossom_id,
                "parent_id": None,  # set in pass 2
                "depth": int(rec.get("depth") or 0),
                "order_index": int(rec.get("sibling_index") or 0),
                "state_text": plan_text,
                "accepted": True,
                "rationale": None,
                "scores": ({"reward": float(rec["metric"])} if rec.get("metric") is not None else None),
                "features": None,
                "tags": [str(rec.get("type") or "draft")],
                "extra_data": {"ext_id": ext_id, "root_id": rec.get("root_id")},
            })
            ext2db[ext_id] = obj.id

    def _persist_edges_and_parents_second_pass(self, blossom_id: int, forest_out: Dict[str, Any], ext2db: Dict[str, int]) -> None:
        nodes: List[Dict[str, Any]] = forest_out.get("nodes", []) or []
        for rec in nodes:
            ext_id = rec.get("id")
            db_id = ext2db.get(ext_id)
            parent_ext = rec.get("parent_id")
            if parent_ext:
                parent_db = ext2db.get(parent_ext)
                if parent_db:
                    # set DB parent on node
                    self.blossoms.update_node(db_id, {"parent_id": parent_db})
                    # add edge
                    self.blossoms.add_edge({
                        "blossom_id": blossom_id,
                        "src_node_id": parent_db,
                        "dst_node_id": db_id,
                        "relation": str(rec.get("type") or "child"),
                        "score": None,
                        "rationale": None,
                        "extra_data": None,
                    })

    # ------------------------------------------------------------------ #
    # Forest utilities
    # ------------------------------------------------------------------ #
    def _by_id(self, forest_out: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = forest_out.get("nodes", []) or []
        return {n["id"]: n for n in nodes}

    def _topk_leaves(self, forest_out: Dict[str, Any], k: int) -> List[Tuple[str, float]]:
        nodes: List[Dict[str, Any]] = forest_out.get("nodes", []) or []
        rewards: Dict[str, float] = {
            nid: float(r) for nid, r in (forest_out.get("rewards") or {}).items()
        }
        # leaf = node id not referenced as a parent
        node_ids = [n["id"] for n in nodes]
        parents = {n["parent_id"] for n in nodes if n.get("parent_id")}
        leaves = [nid for nid in node_ids if nid not in parents]
        def score(nid: str) -> float:
            if nid in rewards:
                return rewards[nid]
            # fallback to node.metric if present
            rec = next((n for n in nodes if n["id"] == nid), None)
            return float(rec.get("metric") or 0.0) if rec else 0.0
        scored = [(nid, score(nid)) for nid in leaves]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: max(1, int(k))]

    def _path_for_leaf(self, forest_out: Dict[str, Any], leaf_id: str) -> List[str]:
        by_id = self._by_id(forest_out)
        path: List[str] = []
        cur = leaf_id
        while cur and cur in by_id:
            path.append(cur)
            cur = by_id[cur].get("parent_id")
        return list(reversed(path))

    # ------------------------------------------------------------------ #
    # Sharpen loop (optional, scorer-backed)
    # ------------------------------------------------------------------ #
    async def _sharpen_top_leaves(self, top_paths: List[List[str]], context: Dict[str, Any]):
        out = []
        for path in top_paths:
            leaf = self.base_search.nodes_by_id.get(path[-1])
            plan_text = getattr(leaf, "plan", "") or ""

            improved = self.call_llm(
                self._build_sharpen_prompt(context.get("goal", {}).get("goal_text", ""), plan_text),
                context=context
            ).strip()

            # Re-score improved text
            try:
                sc = Scorable(id=None, text=improved, target_type="document")
                if hasattr(self.scoring, "evaluate_state"):
                    res = self.scoring.evaluate_state(
                        scorable=sc,
                        context=context,
                        scorers=context.get("scorers") or ["sicql", "mrq", "hrm"],
                        dimensions=context.get("dimensions") or ["alignment", "faithfulness", "coverage", "clarity", "coherence"],
                        scorer_weights=context.get("scorer_weights") or {},
                        dimension_weights=context.get("dimension_weights") or {},
                        include_llm_heuristic=bool(context.get("use_llm_heuristic", True)),
                        fuse_mode="weighted_mean",
                        clamp_01=True,
                    )
                    score_val = float(res.get("overall", 0.0))
                else:
                    # simple fallback
                    bundle = self.scoring.score(
                        scorer_name="sicql", scorable=sc, context=context, dimensions=["alignment"]
                    )
                    score_val = float(bundle.aggregate())
            except Exception:
                score_val = 0.0

            out.append({"original": plan_text, "sharpened": improved, "score": score_val})
        return out

    def _build_sharpen_prompt(self, goal: str, plan: str) -> str:
        return f"""Improve the following plan for the goal.
Keep the same steps, but make it clearer, safer, and more goal-aligned.

GOAL:
{goal}

PLAN:
\"\"\"{plan}\"\"\"


Rewrite the plan now:"""
