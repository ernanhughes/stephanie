from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent
# These ORMs/stores were defined earlier in our Blossom schema work
from stephanie.models.blossom import BlossomNodeORM, BlossomORM
from stephanie.models.blossom_output import (BlossomOutputFeaturesORM,
                                             BlossomOutputORM)
from stephanie.scoring.scorable import ScorableType

# ------------------------------- CONFIG ----------------------------------

@dataclass
class BlossomToScorableConfig:
    # What to blossom (default: chat turns). You can also pass items via context["items"].
    source_type: str = "chat_turn"          # 'chat_turn' | 'document_section' | 'dynamic' | 'raw'
    scorable_out_type: str = "dynamic"      # what scorable we materialize as
    max_candidates_per_node: int = 3        # keep top-k candidates in addition to winner
    keep_roles: Tuple[str, ...] = ("winner", "candidate")
    attach_features: bool = True            # compute features if available
    emit_training_events: bool = True       # add pointwise/pairwise events
    add_baseline_row: bool = True           # store baseline in blossom_outputs if present
    default_dimension: str = "alignment"    # training events dimension tag
    scorer_name: str = "sicql"              # metadata only
    min_reward_delta: float = 0.0           # only emit pairwise if winner >= baseline + delta
    # Optional sharpening/refine step for each candidate before scoring
    refine_with_llm: bool = False


# --------------------------- RUNNER INTERFACE ----------------------------

class BlossomRunnerIFace:
    """
    Expected minimal interface for a blossom runner injected via container:
      run_episode(source_text: str, goal: dict, context: dict) -> dict
    Returns dict with shape:
      {
        "episode_id": int,
        "nodes": [
            {"bn_id": int, "parent_bn_id": Optional[int], "plan_text": str,
             "reward": float, "metrics": dict, "role": "candidate"|"winner"|...}
        ],
        "winners": [
            {"bn_id": int, "plan_text": str, "reward": float, "metrics": dict}
        ],
        "baseline": {"text": str, "metrics": dict, "reward": float}  # optional
      }
    """
    async def run_episode(self, source_text: str, goal: dict, context: dict) -> dict:  # pragma: no cover
        raise NotImplementedError


# ----------------------------- THE AGENT ---------------------------------

class BlossomToScorableAgent(BaseAgent):
    """
    Orchestrates: source item -> Blossom episode -> Scorables (+features/metrics) -> links.

    Input (context):
      - items: Optional[List[dict]]    # each item must include 'id' and 'text' if source_type='raw'
      - goal: Optional[dict]           # used by runner; falls back to cfg or pipeline goal

    Output (context):
      - blossom_scorables: Dict[bn_id, List[{role, scorable_type, scorable_id, reward}]]
      - blossom_episode_id: int
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = BlossomToScorableConfig(**{
            **BlossomToScorableConfig().__dict__,
            **{k: v for k, v in (cfg or {}).items()}
        })
        # Resolve runner & helpers
        self.runner: BlossomRunnerIFace = container.get("blossom_runner")
        self.scoring = container.get("scoring")                 # optional
        self.feature_builder = container.get("feature_builder") # optional

        # Stores
        self._out_store = getattr(self.memory, "blossom_outputs", None)
        if self._out_store is None:
            raise RuntimeError("memory.blossom_outputs store is required.")
        self._dyn = getattr(self.memory, "dynamic_scorables", None)
        if self._dyn is None and self.cfg.scorable_out_type == "dynamic":
            raise RuntimeError("memory.dynamic_scorables store is required for dynamic outputs.")

    # ----------------------------- PUBLIC --------------------------------

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = self._resolve_goal(context)
        items = self._load_source_items(context)  # [{id, text, ...}]
        if not items:
            self.report({"event": "no_items"})
            return context

        all_node_links: Dict[int, List[Dict[str, Any]]] = {}
        episode_id_last: Optional[int] = None

        for item in items:
            src_text = item["text"]
            # 1) Run Blossom on this item
            episode = await self.runner.run_episode(src_text, goal, context)
            episode_id = int(episode.get("episode_id"))
            episode_id_last = episode_id
            winners = episode.get("winners", []) or []
            nodes = episode.get("nodes", []) or []
            baseline = episode.get("baseline")

            # 2) Optionally capture baseline as a scorable + output row
            baseline_row = None
            if self.cfg.add_baseline_row and baseline and baseline.get("text"):
                baseline_row = self._persist_baseline(episode_id, baseline, item)

            # 3) Materialize winners/candidates into scorables, link to nodes
            #    Keep top-K by reward for candidates (besides winners)
            #    We’ll build per-node scorable lists as we go.
            node_links: Dict[int, List[Dict[str, Any]]] = {}

            # winners first
            for w in winners:
                link = self._materialize_and_link(
                    episode_id=episode_id,
                    bn_id=int(w["bn_id"]),
                    role="winner",
                    text=w["plan_text"],
                    reward=float(w.get("reward", 0.0)),
                    metrics=dict(w.get("metrics") or {}),
                    source_item=item,
                    baseline_row=baseline_row,
                )
                node_links.setdefault(int(w["bn_id"]), []).append(link)

            # then top-K candidates by reward, excluding winners
            win_ids = {int(w["bn_id"]) for w in winners}
            cands = [n for n in nodes if n.get("role") == "candidate" and int(n["bn_id"]) not in win_ids]
            cands.sort(key=lambda r: float(r.get("reward", 0.0)), reverse=True)
            for c in cands[: max(0, self.cfg.max_candidates_per_node)]:
                link = self._materialize_and_link(
                    episode_id=episode_id,
                    bn_id=int(c["bn_id"]),
                    role="candidate",
                    text=c["plan_text"],
                    reward=float(c.get("reward", 0.0)),
                    metrics=dict(c.get("metrics") or {}),
                    source_item=item,
                    baseline_row=baseline_row,
                )
                node_links.setdefault(int(c["bn_id"]), []).append(link)

            # 4) Persist per-node links into BlossomNode.extra_data for quick lookup
            self._attach_links_to_nodes(node_links)

            # 5) Accumulate in output
            for bn_id, links in node_links.items():
                all_node_links.setdefault(bn_id, []).extend(links)

        # 6) return mapping (bn_id -> scorable links)
        context["blossom_scorables"] = all_node_links
        if episode_id_last is not None:
            context["blossom_episode_id"] = episode_id_last
        return context

    # ---------------------------- HELPERS --------------------------------

    def _resolve_goal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get("goal") or {}
        if "goal_text" not in goal:
            # fallback to pipeline goal if present
            pg = context.get("pipeline_goal") or {}
            if "goal_text" in pg:
                goal = pg
            else:
                goal = {"goal_text": "Improve and diversify scorables via Blossom."}
        return goal

    def _load_source_items(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Return a list of items: [{id, text, ...}]
        If context["items"] provided, we trust it. Otherwise, try store by source_type.
        """
        if context.get("items"):
            # normalize
            out = []
            for it in context["items"]:
                if "text" in it and "id" in it:
                    out.append(it)
            return out

        st = self.cfg.source_type
        if st == "chat_turn":
            store = getattr(self.memory, "chat_turns", None)
            if not store:
                self.logger.warn("No chat_turns store; provide items via context.")
                return []
            rows = store.recent(limit=50)  # choose your loader
            return [{"id": r.id, "text": r.text, "meta": r.to_dict()} for r in rows]
        elif st == "document_section":
            ds = getattr(self.memory, "document_sections", None)
            if not ds:
                self.logger.warn("No document_sections store; provide items via context.")
                return []
            # Load any recent sections; caller can filter via context in practice
            rows = ds.get_by_document(context.get("document_id", -1)) if context.get("document_id") else []
            return [{"id": r.id, "text": (r.section_text or ""), "meta": r.to_dict()} for r in rows]
        elif st == "dynamic":
            dyn = getattr(self.memory, "dynamic_scorables", None)
            if not dyn:
                self.logger.warn("No dynamic_scorables store; provide items via context.")
                return []
            rows = dyn.recent(limit=50)
            return [{"id": r.id, "text": (r.text or ""), "meta": r.to_dict()} for r in rows]
        else:
            self.logger.warn(f"Unknown source_type={st}; provide items in context.")
            return []

    def _persist_baseline(self, episode_id: int, baseline: Dict[str, Any], item: Dict[str, Any]) -> Optional[BlossomOutputORM]:
        """
        Optionally store the baseline version (pre-blossom text) into blossom_outputs,
        and (optionally) also as a dynamic scorable for uniform comparisons.
        """
        try:
            base_text = baseline.get("text") or ""
            if not base_text.strip():
                return None

            # Create a scorable row for the baseline if we want strict A/B symmetry
            scorable_id = None
            if self.cfg.scorable_out_type == "dynamic":
                rec = self._dyn.add(
                    pipeline_run_id=None,
                    scorable_type=ScorableType.DYNAMIC,
                    source="blossom",
                    text=base_text,
                    source_scorable_type=self.cfg.source_type,
                    source_scorable_id=int(item["id"]),
                    meta={"episode_id": episode_id, "role": "baseline"},
                )
                scorable_id = int(rec.id)

            row = self._out_store.upsert_output_unique({
                "episode_id": episode_id,
                "source_bn_id": None,
                "role": "baseline",
                "scorable_type": self.cfg.scorable_out_type,
                "scorable_id": scorable_id if scorable_id is not None else int(item["id"]),
                "source_type": self.cfg.source_type,
                "source_id": int(item["id"]),
                "reward": float(baseline.get("reward", 0.0)),
                "metrics": dict(baseline.get("metrics") or {}),
                "notes": "baseline capture",
            })
            return row
        except Exception as e:
            self.logger.log("BlossomBaselinePersistError", {"episode_id": episode_id, "error": str(e)})
            return None

    def _materialize_and_link(
        self,
        *,
        episode_id: int,
        bn_id: int,
        role: str,
        text: str,
        reward: float,
        metrics: Dict[str, Any],
        source_item: Dict[str, Any],
        baseline_row: Optional[BlossomOutputORM],
    ) -> Dict[str, Any]:
        """
        Create (dynamic) scorable, persist blossom_outputs (+features), optionally emit training events.
        Returns a lightweight link record for attaching to the node.
        """
        # Optional refinement with LLM (sharpening-style)
        out_text = text
        if self.cfg.refine_with_llm:
            out_text = self._refine_text(text=text, goal_text=self._resolve_goal({"goal":{}})["goal_text"])

        # 1) Persist scorable
        if self.cfg.scorable_out_type != "dynamic":
            raise NotImplementedError("Only dynamic scorable materialization is implemented here.")
        scorable = self._dyn.add(
            pipeline_run_id=None,
            scorable_type=ScorableType.DYNAMIC,
            source="blossom",
            text=out_text,
            source_scorable_type=self.cfg.source_type,
            source_scorable_id=int(source_item["id"]),
            meta={
                "episode_id": episode_id,
                "bn_id": bn_id,
                "role": role,
                "origin": "blossom_to_scorable",
                "reward": reward,
                "metrics": metrics,
            },
        )
        scorable_id = int(scorable.id)

        # 2) Blossom outputs row
        out_row = self._out_store.upsert_output_unique({
            "episode_id": episode_id,
            "source_bn_id": bn_id,
            "role": role,
            "scorable_type": self.cfg.scorable_out_type,
            "scorable_id": scorable_id,
            "source_type": self.cfg.source_type,
            "source_id": int(source_item["id"]),
            "reward": float(reward),
            "metrics": dict(metrics or {}),
            "notes": None,
        })

        # 3) Optional features
        if self.cfg.attach_features:
            try:
                feats = None
                if callable(self.feature_builder):
                    feats = self.feature_builder(out_text)  # expected dict
                if feats:
                    self._out_store.add_features(out_row.id, feats)
            except Exception as e:
                self.logger.log("BlossomOutputFeatureError", {"output_id": out_row.id, "error": str(e)})

        # 4) Optional training events: pairwise baseline vs winner/candidate
        if self.cfg.emit_training_events and baseline_row and role in ("winner",):
            try:
                self._emit_training(baseline_row, out_row, source_item, reward_delta=self.cfg.min_reward_delta)
            except Exception as e:
                self.logger.log("BlossomTrainingEventError", {"episode_id": episode_id, "error": str(e)})

        return {
            "role": role,
            "scorable_type": self.cfg.scorable_out_type,
            "scorable_id": scorable_id,
            "reward": float(reward),
        }

    def _emit_training(self, baseline_row: BlossomOutputORM, improved_row: BlossomOutputORM,
                       source_item: Dict[str, Any], reward_delta: float) -> None:
        """
        Emit pointwise + pairwise events so repeated blossoms can lift policy quality.
        Uses stored metrics/reward when available; otherwise falls back to simple values.
        """
        if not hasattr(self.memory, "training_events"):
            return

        # Retrieve texts (best effort)
        base_text = self._fetch_text_for_output(baseline_row)
        imp_text = self._fetch_text_for_output(improved_row)
        title = (source_item.get("meta") or {}).get("title") or f"{self.cfg.source_type}:{source_item['id']}"

        # Simple overall value proxies
        base_val = float(baseline_row.reward or 0.0)
        imp_val = float(improved_row.reward or 0.0)

        if imp_val < base_val + reward_delta:
            return  # skip weak wins

        w = max(0.1, min(1.0, (imp_val - base_val) + 0.3))

        # Pointwise (improved)
        self.memory.training_events.add_pointwise(
            model_key="retriever.mrq.v1",
            dimension=self.cfg.default_dimension,
            query_text=title,
            cand_text=imp_text,
            label=1,
            weight=imp_val,
            trust=max(0.1, min(1.0, imp_val)),
            goal_id=None,
            pipeline_run_id=None,
            agent_name=self.name,
            source="blossom",
            meta={"episode_id": improved_row.episode_id, "bn_id": improved_row.source_bn_id},
        )
        # Pairwise (improved vs baseline)
        self.memory.training_events.insert_pairwise(
            model_key="ranker.sicql.v1",
            dimension=self.cfg.default_dimension,
            query_text=title,
            pos_text=imp_text,
            neg_text=base_text,
            weight=w,
            trust=w * 0.6,
            goal_id=None,
            pipeline_run_id=None,
            agent_name=self.name,
            source="blossom",
            meta={
                "episode_id": improved_row.episode_id,
                "bn_id": improved_row.source_bn_id,
                "improved_reward": imp_val,
                "baseline_reward": base_val,
                "gain": imp_val - base_val,
            },
        )

    def _fetch_text_for_output(self, out_row: BlossomOutputORM) -> str:
        """Resolve text for a blossom_outputs record (dynamic scorable path)."""
        if out_row.scorable_type == "dynamic":
            try:
                rec = self._dyn.get(out_row.scorable_id)
                return getattr(rec, "text", "") or ""
            except Exception:
                return ""
        return ""

    def _attach_links_to_nodes(self, node_links: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Persist scorable links into BlossomNode.extra_data["scorables"] for quick lookup.
        """
        s = self.memory.session
        for bn_id, links in node_links.items():
            node: Optional[BlossomNodeORM] = s.query(BlossomNodeORM).get(bn_id)
            if not node:
                continue
            extra = dict(node.extra_data or {})
            lst = list(extra.get("scorables", []))
            lst.extend(links)
            extra["scorables"] = dedup_links(lst)
            node.extra_data = extra
        s.commit()

    def _refine_text(self, *, text: str, goal_text: str) -> str:
        """
        Lightweight built-in sharpening step; replace with your Track-B loop if desired.
        """
        prompt = f"""You are a precise editor improving a candidate answer.

Goal: {goal_text}

Improve this text for clarity, faithfulness, and usefulness. Keep length similar.

TEXT:
{text}

Rewrite (single block, no bullets):"""
        try:
            return self.call_llm(prompt).strip()
        except Exception:
            return text


# ------------------------------ UTIL -------------------------------------

def dedup_links(links: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for l in links:
        key = (l.get("role"), l.get("scorable_type"), int(l.get("scorable_id", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out
