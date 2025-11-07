# stephanie/agents/scorable_loader.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import random

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.utils.progress_mixin import ProgressMixin
from stephanie.utils.embed_utils import as_list_floats, cos_safe, farthest_point_sample
from stephanie.data.text_item import TextItem

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

        # --- Preload knobs (oversample then filter) ---
        pre = cfg.get("preload", {}) or {}
        self.preload_enabled = bool(pre.get("enabled", self.ab_enabled))
        self.preload_total = int(pre.get("total_limit", max(self.limit * 10, 1000)))
        self.preload_include_texts = bool(pre.get("include_texts", True))
        self.preload_require_ner = bool(pre.get("require_nonempty_ner", False))
        self.preload_require_assistant = bool(pre.get("require_assistant_text", False))
        self.preload_min_len = int(pre.get("min_assistant_len", 0))
        self.preload_order_desc = bool(pre.get("order_desc", True))

    async def run(self, context: dict) -> dict:
        scorables: List[Dict[str, Any]] = []
        task = f"ScorableLoad:{context.get('pipeline_run_id', 'na')}"
        self._init_progress(self.container, self.logger)

        if self.target_type == ScorableType.CONVERSATION_TURN:
            # Choose fetch profile: oversample if enabled, otherwise use the tight/defaults
            fetch_total = self.preload_total if self.preload_enabled else self.limit
            fetch_include_texts = self.preload_include_texts if self.preload_enabled else self.include_full_text
            fetch_require_nonempty_ner = self.preload_require_ner if self.preload_enabled else self.include_ner
            fetch_require_assistant = self.preload_require_assistant if self.preload_enabled else True
            fetch_min_len = self.preload_min_len if self.preload_enabled else 1
            fetch_order_desc = self.preload_order_desc if self.preload_enabled else True

            self.pstart(
                task=task,
                total=fetch_total,
                meta={
                    "target_type": str(self.target_type),
                    "fetch_total": fetch_total,
                    "batch_size": self.batch_size,
                    "preload_enabled": self.preload_enabled,
                    "filters": {
                        "require_assistant_text": fetch_require_assistant,
                        "require_nonempty_ner": fetch_require_nonempty_ner,
                        "min_assistant_len": fetch_min_len,
                    },
                },
            )

            produced = 0
            for batch in self.memory.chats.iter_turns_with_texts(
                total_limit=fetch_total,
                batch_size=self.batch_size,
                include_texts=fetch_include_texts,
                include_goal=True,
                require_assistant_text=fetch_require_assistant,
                require_nonempty_ner=fetch_require_nonempty_ner,
                min_assistant_len=fetch_min_len,
                order_desc=fetch_order_desc,
            ):
                for row in batch:
                    scorables.append(asdict(TextItem.from_chat_turn(row)))
                produced += len(batch)
                self.ptick(task=task, done=produced, total=fetch_total)
                if produced >= fetch_total:
                    break

            self.pdone(task=task)

            # If still empty, exit gracefully (prevents ZeroModel finalize crash)
            if not scorables:
                self.logger.log("ScorableLoadEmpty", {
                    "preload_enabled": self.preload_enabled,
                    "fetch_total": fetch_total,
                    "hint": "DB may be empty or filters too strict. Try preload.require_* = false."
                })
                context[self.output_key] = []
                context["retrieved_ids"] = []
                return context

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

            # TARGETED (high sim)
            # --- A/B selection with disjoint guarantee + robust fallback ---

            N = len(scorables)
            idx_all = list(range(N))

            # TARGETED (high sim)
            tgt_thr_init = float(self.ab_targeted_min_sim)
            tgt_idxs = [i for i, s in enumerate(sims) if s >= tgt_thr_init]
            tgt_thr_final = tgt_thr_init
            if len(tgt_idxs) < self.ab_n_targeted:
                step = max(0.02, (tgt_thr_init - 0.20) / max(1, self.ab_relax_steps))
                thr = tgt_thr_init
                for _ in range(self.ab_relax_steps):
                    thr = max(0.20, thr - step)
                    tgt_idxs = [i for i, s in enumerate(sims) if s >= thr]
                    if len(tgt_idxs) >= self.ab_n_targeted:
                        tgt_thr_final = thr
                        break
            tgt_idxs = sorted(tgt_idxs, key=lambda i: sims[i], reverse=True)[: self.ab_n_targeted]
            tgt_set = set(tgt_idxs)

            # BASELINE (low sim + diverse) — exclude targeted up-front
            base_thr_init = float(self.ab_baseline_max_sim)
            base_pool = [i for i, s in enumerate(sims) if s <= base_thr_init and i not in tgt_set]
            base_thr_final = base_thr_init
            if len(base_pool) < self.ab_n_baseline:
                step = max(0.02, (0.80 - base_thr_init) / max(1, self.ab_relax_steps))
                thr = base_thr_init
                for _ in range(self.ab_relax_steps):
                    thr = min(0.80, thr + step)
                    base_pool = [i for i, s in enumerate(sims) if s <= thr and i not in tgt_set]
                    if len(base_pool) >= self.ab_n_baseline:
                        base_thr_final = thr
                        break

            # Fallbacks if still too small:
            if len(base_pool) < self.ab_n_baseline:
                # use the globally lowest-sim items (excluding targeted)
                remain = [i for i in idx_all if i not in tgt_set]
                remain_sorted = sorted(remain, key=lambda i: sims[i])
                base_pool = remain_sorted[: max(self.ab_n_baseline, len(remain_sorted))]

            # Select baseline with diversity if enabled
            if self.ab_ensure_diversity and base_pool:
                task = "FPS-SelectBaseline"
                k_baseline = min(self.ab_n_baseline, len(base_pool))
                self.pstart(task=task, total=k_baseline, meta={"candidates": len(base_pool)})

                def _tick(step, total):
                    self.ptick(task=task, done=step, total=total)

                base_idxs = farthest_point_sample(base_pool, vecs, k_baseline, seed=self.ab_seed, progress=_tick)
                self.pdone(task=task)
            else:
                rng = random.Random(self.ab_seed)
                pool = list(base_pool)
                rng.shuffle(pool)
                base_idxs = pool[: self.ab_n_baseline]

            # Paranoid disjointness guard + final top-up
            base_idxs = [i for i in base_idxs if i not in tgt_set]
            if len(base_idxs) < self.ab_n_baseline:
                extra = [i for i in idx_all if i not in tgt_set and i not in set(base_idxs)]
                base_idxs.extend(extra[: (self.ab_n_baseline - len(base_idxs))])

            # Diagnostics
            overlap = set(tgt_idxs) & set(base_idxs)
            def _stats(idxs):
                if not idxs: return {"n":0,"mean":0.0,"p50":0.0,"p90":0.0}
                vals = sorted(float(sims[i]) for i in idxs)
                n = len(vals); p50 = vals[n//2]; p90 = vals[min(n-1, int(round(0.90*(n-1))))]
                return {"n": n, "mean": sum(vals)/n, "p50": p50, "p90": p90}
            st_t, st_b = _stats(tgt_idxs), _stats(base_idxs)
            jaccard = (len(overlap) / max(1, len(set(tgt_idxs) | set(base_idxs)))) if (tgt_idxs or base_idxs) else 0.0

            self.logger.log("ABSplitSummary", {
                "targeted_thr_init": tgt_thr_init, "targeted_thr_final": tgt_thr_final,
                "baseline_thr_init": base_thr_init, "baseline_thr_final": base_thr_final,
                "targeted_stats": st_t, "baseline_stats": st_b,
                "overlap_count": len(overlap), "jaccard": jaccard,
            })

            # materialize sets
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

