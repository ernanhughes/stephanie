# stephanie/agents/scorable_loader.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import math
import random

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.utils.embed_utils import as_list_floats, cos_safe, dist, farthest_point_sample


class ScorableLoaderAgent(BaseAgent, ProgressMixin):
    """
    Loads chat-turn scorables. When ab_select.enabled=True, it splits into:
      - scorables_targeted: high similarity to goal
      - scorables_baseline: low similarity to goal, diverse among themselves
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", True)
        self.target_type = cfg.get("target_type", ScorableType.CONVERSATION_TURN)
        self.include_ner = cfg.get("include_ner", False)
        self.save_pipeline_refs = cfg.get("save_pipeline_refs", False)
        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])
        self.scoring: ScoringService = self.container.get("scoring")
        self.limit = int(cfg.get("limit", 100))
        self.batch_size = int(cfg.get("batch_size", 100))

        # A/B split knobs
        ab = cfg.get("ab_select", {}) or {}
        self.ab_enabled = bool(ab.get("enabled", False))
        self.ab_seed = int(ab.get("seed", 0))
        self.ab_n_targeted = int(ab.get("n_targeted", self.limit // 2 or 50))
        self.ab_n_baseline = int(ab.get("n_baseline", self.limit // 2 or 50))
        self.ab_targeted_min_sim = float(ab.get("targeted_min_sim", 0.60))
        self.ab_baseline_max_sim = float(ab.get("baseline_max_sim", 0.20))
        self.ab_ensure_diversity = bool(ab.get("ensure_diversity", True))
        self.ab_goal_source = str(ab.get("goal_source", "goal_text")).lower()  # "goal_text" | "user_text_title"
        self.ab_relax_steps = int(ab.get("relax_steps", 5))  # auto-relax thresholds if too few matches

    async def run(self, context: dict) -> dict:
        scorables: List[Dict[str, Any]] = []
        task = f"ScorableLoad:{context.get('pipeline_run_id', 'na')}"
        self._init_progress(self.container, self.logger)

        if self.target_type == ScorableType.CONVERSATION_TURN:
            self.pstart(
                task=task,
                total=self.limit,
                meta={
                    "target_type": str(self.target_type),
                    "limit": self.limit,
                    "batch_size": self.batch_size,
                },
            )

            produced = 0
            for batch in self.memory.chats.iter_turns_with_texts(
                total_limit=self.limit,
                batch_size=self.batch_size,
                include_texts=self.include_full_text,
                include_goal=True,
                require_assistant_text=True,
                require_nonempty_ner=self.include_ner,
                min_assistant_len=1,
                order_desc=True,
            ):
                for row in batch:
                    item = TextItem.from_chat_turn(row)
                    scorables.append(asdict(item))
                produced += len(batch)
                self.ptick(task=task, done=produced, total=self.limit)
                if produced >= self.limit:
                    break

            self.pdone(task=task)

        # ---------------------------------------
        # A/B selection (optional; on by config)
        # ---------------------------------------
        if self.ab_enabled:
            goal_text = self._resolve_goal_text(context, scorables)
            goal_vec = as_list_floats(self.memory.embedding.get_or_create(goal_text))

            # build embeddings + similarity to goal once
            vecs: List[List[float]] = []
            sims: List[float] = []
            n_items = len(scorables)
            task2 = f"Embeddings:{context.get('pipeline_run_id','na')}"
            # start a dedicated progress task for this slow phase
            self.pstart(
                task=task2,
                total=n_items,
                meta={
                    "phase": "embed+sim",
                    "target_type": str(self.target_type),
                    "items": n_items,
                },
            )

            # tick every ~2% of progress (at least every 1 item)
            tick_every = max(1, n_items // 50)
            done = 0


            for i, s in enumerate(scorables, 1):
                txt = (s.get("text") or s.get("user_text") or "").strip()
                # NOTE: keep as a single call; DB conn likely not thread-safe, so avoid threading here.
                v = self.memory.embedding.get_or_create(txt[:4096] if txt else " ")
                s.setdefault("embeddings", {})["global"] = v

                sim = cos_safe(v, goal_vec)
                s["ab_sim_goal"] = float(sim)

                vecs.append(v)
                sims.append(sim)

                done = i
                if (i % tick_every) == 0 or i == n_items:
                    self.ptick(task=task2, done=done, total=n_items)

            self.pdone(task=task2)

            # pick TARGETED (high sim)
            tgt_idxs = [i for i, sim in enumerate(sims) if sim >= self.ab_targeted_min_sim]
            # relax if not enough
            if len(tgt_idxs) < self.ab_n_targeted:
                step = max(0.02, (self.ab_targeted_min_sim - 0.20) / max(1, self.ab_relax_steps))
                thr = self.ab_targeted_min_sim
                for _ in range(self.ab_relax_steps):
                    thr = max(0.20, thr - step)
                    tgt_idxs = [i for i, sim in enumerate(sims) if sim >= thr]
                    if len(tgt_idxs) >= self.ab_n_targeted:
                        break
                # final trim
            tgt_idxs = sorted(tgt_idxs, key=lambda i: sims[i], reverse=True)[: self.ab_n_targeted]

            # pick BASELINE (low sim + diverse)
            base_pool = [i for i, sim in enumerate(sims) if sim <= self.ab_baseline_max_sim]
            # relax if not enough
            if len(base_pool) < self.ab_n_baseline:
                step = max(0.02, (0.80 - self.ab_baseline_max_sim) / max(1, self.ab_relax_steps))
                thr = self.ab_baseline_max_sim
                for _ in range(self.ab_relax_steps):
                    thr = min(0.80, thr + step)
                    base_pool = [i for i, sim in enumerate(sims) if sim <= thr]
                    if len(base_pool) >= self.ab_n_baseline:
                        break

            if self.ab_ensure_diversity and base_pool:
                task = "FPS-SelectBaseline"
                k_baseline = min(self.ab_n_baseline, len(base_pool))  # or: min(len(tgt_idxs), len(base_pool))
                self.pstart(task=task, total=k_baseline)

                def _tick(step, total):
                    self.ptick(task=task, done=step, total=total)

                base_idxs = farthest_point_sample(
                    base_pool,            # indexes into vecs
                    vecs,                 # full vec list
                    k_baseline,
                    seed=self.ab_seed,
                    progress=_tick
                )
                self.pdone(task=task)
            else:
                rng = random.Random(self.ab_seed)
                rng.shuffle(base_pool)
                base_idxs = base_pool[: self.ab_n_baseline]

            scorables_targeted = [scorables[i] for i in tgt_idxs]
            scorables_baseline = [scorables[i] for i in base_idxs]

            # expose both sets; keep the union in 'scorables' for backward-compat
            context["scorables_targeted"] = scorables_targeted
            context["scorables_baseline"] = scorables_baseline
            context[self.output_key] = scorables_targeted + scorables_baseline
            context["retrieved_ids"] = [d.get("scorable_id") for d in (scorables_targeted + scorables_baseline)]

            self.logger.log("ABSelectCompleted", {
                "n_total": len(scorables),
                "goal_text": (goal_text or "")[:120],
                "targeted_min_sim": self.ab_targeted_min_sim,
                "baseline_max_sim": self.ab_baseline_max_sim,
                "n_targeted": len(scorables_targeted),
                "n_baseline": len(scorables_baseline),
                "relaxed": (len(tgt_idxs) < self.ab_n_targeted) or (len(base_idxs) < self.ab_n_baseline),
            })

            if self.save_pipeline_refs:
                for d in scorables_targeted:
                    self.memory.pipeline_references.insert({
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "scorable_type": d["scorable_type"],
                        "scorable_id": d["scorable_id"],
                        "relation_type": "retrieved_targeted",
                        "source": self.name,
                    })
                for d in scorables_baseline:
                    self.memory.pipeline_references.insert({
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "scorable_type": d["scorable_type"],
                        "scorable_id": d["scorable_id"],
                        "relation_type": "retrieved_baseline",
                        "source": self.name,
                    })

            return context

        # non-AB fallback: previous behavior
        context[self.output_key] = scorables
        context["retrieved_ids"] = [d.get("scorable_id") for d in scorables]

        if self.save_pipeline_refs:
            for d in scorables:
                self.memory.pipeline_references.insert(
                    {
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "scorable_type": d["scorable_type"],
                        "scorable_id": d["scorable_id"],
                        "relation_type": "retrieved",
                        "source": self.name,
                    }
                )

        self.logger.log(
            "KnowledgeDBLoaded",
            {
                "count": len(scorables),
                "target_type": str(self.target_type),
                "limit": self.limit,
                "batch_size": self.batch_size,
            },
        )
        return context

    # ------------- internals -------------
    def _resolve_goal_text(self, context: dict, scorables: List[Dict[str, Any]]) -> str:
        # 1) explicit goal
        gx = context.get("goal") or {}
        gtxt = (gx.get("goal_text") or "").strip()
        if gtxt:
            return gtxt
        # 2) conversation title of newest turn (your from_chat_turn puts it in meta.conversation_title)
        if self.ab_goal_source == "user_text_title":
            for s in scorables:
                t = (s.get("meta") or {}).get("conversation_title") or ""
                if t.strip():
                    return t.strip()
        # 3) fallback: first user_text with content
        for s in scorables:
            ut = (s.get("user_text") or "").strip()
            if ut:
                return ut
        return "default target"


class GoalKind(str, Enum):
    CONTEXT = "context"
    TURN_USER = "turn_user"
    SECTION = "section"
    CUSTOM = "custom"


@dataclass
class GoalRef:
    text: str = ""
    kind: GoalKind = GoalKind.CONTEXT


@dataclass
class TextItem:
    scorable_type: ScorableType = ScorableType.CONVERSATION_TURN
    scorable_id: Optional[int] = None
    conversation_id: Optional[int] = None
    external_id: Optional[str] = None

    order_index: int = 0
    created_at: Optional[datetime] = None

    text: str = ""
    user_text: str = ""
    goal_ref: GoalRef = field(default_factory=GoalRef)

    ner: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)
    domains: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)

    star: Optional[int] = None
    ai_score: Optional[float] = None
    ai_rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chat_turn(cls, row: Dict[str, Any]) -> "TextItem":
        goal_text_user = (row.get("user_text") or "").strip()
        goal_text_title = (row.get("goal_text") or "").strip()
        return cls(
            scorable_type=ScorableType.CONVERSATION_TURN,
            scorable_id=int(row.get("id")) if row.get("id") is not None else None,
            conversation_id=row.get("conversation_id"),
            order_index=int(row.get("order_index") or 0),
            created_at=None,
            text=(row.get("assistant_text") or "").strip(),
            user_text=(row.get("user_text") or "").strip(),
            goal_ref=GoalRef(
                text=goal_text_user if goal_text_user else goal_text_title,
                kind=GoalKind.TURN_USER if goal_text_user else GoalKind.CONTEXT,
            ),
            ner=row.get("ner") or [],
            domains=row.get("domains") or [],
            star=(int(row["star"]) if row.get("star") is not None else None),
            ai_score=(float(row["ai_score"]) if row.get("ai_score") is not None else None),
            ai_rationale=row.get("ai_rationale") or "",
            meta={"conversation_title": goal_text_title},
        )
