# stephanie/agents/dspy/mcts_reasoning_agent.py
"""
MCTSReasoningAgent (enhanced, patched)
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from collections import OrderedDict, defaultdict

import dspy
from dspy import InputField, OutputField, Predict, Signature

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.utils.llm_response_parser import parse_scored_block

_logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# DSPy Signatures
# -------------------------------------------------------------------------
class TraceStep(Signature):
    state = InputField(desc="Current problem state")
    trace = InputField(desc="History of thoughts/actions so far")
    next_step = OutputField(desc="Next reasoning step")


class ValueEstimator(Signature):
    state = InputField(desc="Current problem state")
    trace = InputField(desc="Reasoning steps taken")
    goal = InputField(desc="Goal text")
    score = OutputField(desc="Normalized score (0–1)")
    rationale = OutputField(desc="Why this path is promising")


# -------------------------------------------------------------------------
# LM wrapper (optional prompt logging)
# -------------------------------------------------------------------------
class LoggingLM(dspy.LM):
    def __init__(self, *args, debug_prompts: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._debug_prompts = debug_prompts

    def __call__(self, *args, **kwargs):
        if self._debug_prompts:
            prompt = kwargs.get("prompt")
            messages = kwargs.get("messages")
            if prompt:
                _logger.debug(
                    "=== DSPy PROMPT ===\n%s\n====================", prompt
                )
            if messages:
                _logger.debug(
                    "=== DSPy MESSAGES ===\n%s\n====================", messages
                )
        result = super().__call__(*args, **kwargs)
        if self._debug_prompts:
            _logger.debug(
                "=== DSPy RESPONSE ===\n%s\n====================", result
            )
        return result


# -------------------------------------------------------------------------
# Node
# -------------------------------------------------------------------------
class MCTSReasoningNode:
    def __init__(self, state, trace, parent=None):
        self.id = uuid.uuid4().hex[:6]
        self.state = state
        self.trace = trace
        self.parent = parent
        self.visits = 0
        self.reward = 0.0
        self.children = []
        self.score = 0.0
        self.dimension_scores = {}

    def uct_value(self, parent_visits, ucb_weight):
        if self.visits == 0:
            return float("inf")
        exploitation = self.reward / max(1, self.visits)
        exploration = ucb_weight * math.sqrt(
            max(1e-9, math.log(max(1, parent_visits)) / self.visits)
        )
        return exploitation + exploration


# -------------------------------------------------------------------------
# Program shell (holds Predict modules; no budgets here)
# -------------------------------------------------------------------------
class MCTSReasoningProgram(dspy.Module):
    def __init__(self, cfg):
        super().__init__()
        self.generator = Predict(TraceStep)
        self.value_estimator = Predict(ValueEstimator)
        self.max_depth = int(cfg.get("max_depth", 3))


# -------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------
class MCTSReasoningAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container=container, logger=logger)

        # search / scoring knobs
        self.max_depth = int(cfg.get("max_depth", 4))
        self.branching_factor = int(cfg.get("branching_factor", 2))
        self.num_simulations = int(cfg.get("num_simulations", 20))
        self.ucb_weight = float(cfg.get("ucb_weight", 1.41))

        self.scorer_name = cfg.get("scorer_name", "sicql")
        self.scorable_type = cfg.get("scorable_type", ScorableType.HYPOTHESIS)
        self.dimensions = cfg.get(
            "dimensions",
            [
                "alignment",
                "clarity",
                "implementability",
                "novelty",
                "relevance",
            ],
        )

        # evaluation schedule
        self.eval_at = str(cfg.get("eval_at", "leaf"))  # "leaf" | "every_k"
        self.eval_stride = int(cfg.get("eval_stride", 2))
        ve_cfg = cfg.get("value_estimator", {}) or {}
        self.value_estimator_enabled = bool(ve_cfg.get("enabled", False))
        self.value_estimator_at_leaf_only = bool(
            ve_cfg.get("at_leaf_only", True)
        )

        # runtime containers
        self.nodes = []
        self.children = defaultdict(list)
        self.score_cache = {}

        # program & LM
        self.lats_program = MCTSReasoningProgram(cfg)
        model_cfg = cfg.get("model", {}) or {}
        lm = LoggingLM(
            model_cfg.get("name", "ollama_chat/qwen3"),
            api_base=model_cfg.get("api_base", "http://localhost:11434"),
            api_key=model_cfg.get("api_key", ""),
            debug_prompts=bool(cfg.get("debug_prompts", False)),
        )
        dspy.configure(lm=lm)

        # expansion controls
        self.expand_mode = str(
            cfg.get("expand_mode", "single_step")
        )  # "single_step" | "rollout"
        self.samples_per_expand = int(
            cfg.get(
                "samples_per_expand", max(1, cfg.get("branching_factor", 2))
            )
        )

        # budgets / caching (kept on the agent)
        self.max_lm_calls = int(cfg.get("max_lm_calls", 12))
        self.calls_used = 0
        cache_cfg = cfg.get("caching", {}) or {}
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        self.cache_size = int(cache_cfg.get("lru_size", 2048))
        self._cache = OrderedDict()
        self._cache_tail_len = int(cache_cfg.get("tail_len", 6))

        # early stopping if we hit an acceptable score
        t = cfg.get("early_stop_threshold", None)  # e.g., 0.75 or 75
        self.early_stop_threshold = None if t is None else float(t)
        if (
            self.early_stop_threshold is not None
            and self.early_stop_threshold > 1.0
        ):
            self.early_stop_threshold /= 100.0  # accept 75 as 0.75

        self._best_so_far = (float("-inf"), None)  # (score, node)
        self.top_k = int(cfg.get("top_k_leaves", 3))

        self.search_strategy_name = cfg.get("strategy_name", "mcts_v1")
        self.domain = cfg.get("domain", "general")

        self.light_cfg = None
        self._first_emit_done = False
        self._best_emitted = None   # (score, fragment_id)
        self._t0 = None             # start time per run


        _logger.debug(
            "MCTSInitialized depth=%s bf=%s sims=%s ucb=%s dims=%s max_lm_calls=%s eval_at=%s stride=%s",
            self.max_depth,
            self.branching_factor,
            self.num_simulations,
            self.ucb_weight,
            self.dimensions,
            self.max_lm_calls,
            self.eval_at,
            self.eval_stride,
        )

    # ---------------- cache helpers ----------------
    def _cache_key(self, state: str, trace: list[str]) -> tuple:
        return (hash(state), tuple(trace[-self._cache_tail_len :]))

    def _cache_get(self, key):
        if not self.cache_enabled:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _cache_put(self, key, value: str):
        if not self.cache_enabled:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _ms_since_start(self):
        import time
        return int((time.time() - (self._t0 or time.time())) * 1000)

    def _should_emit_now(self, sim_idx: int) -> bool:
        lc = self.light_cfg
        if not lc or not lc.enabled:
            return False
        # first emit condition
        if not self._first_emit_done and sim_idx + 1 >= lc.first_emit_after_sims:
            return True
        # cadence
        if (sim_idx + 1) % lc.emit_every_sims == 0:
            return True
        # hard deadline: if deadline approaching, force emit once
        return (not self._first_emit_done) and (self._ms_since_start() >= lc.hard_deadline_ms)

    # ---------------- LM step (budget + cache) ----------------
    def _predict_next(self, state: str, trace: list[str]) -> str:
        if self.calls_used >= self.max_lm_calls:
            return ""
        key = self._cache_key(state, trace)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        # 👉 call the program's generator
        pred = self.lats_program.generator(state=state, trace=trace)
        self.calls_used += 1
        nxt = (
            pred.next_step.strip()
            if (pred and getattr(pred, "next_step", None))
            else ""
        )
        self._cache_put(key, nxt)
        return nxt

    def next_step_once(self, state: str, trace: list[str]) -> str:
        return self._predict_next(state, trace)

    # ---------------- main entry ----------------
    async def run(self, context: dict) -> dict:

        self._t0 = time.time()
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        root = self._create_node(state=goal_text, trace=[], parent=None)
        self.logger.log("MCTSReasoningAgentStart", {"goal": goal_text})

        # best_so_far: (score, node). Ensure initialized each run.
        self._best_so_far = (float("-inf"), None)

        # MCTS loop
        for sim in range(self.num_simulations):
            node = self._select(root)
            if not self._is_terminal(node):
                node = await self._expand(node, context)

            reward = self._evaluate(node, context)
            # after scoring/backprop
            if self._should_emit_now(sim):
                # pick current best leaf
                leaves = self._collect_leaves(root)
                if leaves:
                    best_leaf = max(leaves, key=lambda n: (n.score or 0.0))
                    self._emit_lightning(context, best_leaf)

            # make robust if _evaluate returns None
            reward = 0.0 if reward is None else float(reward)

            self._backpropagate(node, reward)

            # update best-so-far (based on the just-evaluated node)
            if reward > self._best_so_far[0]:
                self._best_so_far = (reward, node)

            if (sim + 1) % 5 == 0:
                self._log_progress(sim, root)

            # early stopping if we hit an acceptable score
            if self.early_stop_threshold is not None and self._best_so_far[0] >= float(self.early_stop_threshold):
                self.logger.log(
                    "MCTSEarlyStop",
                    {
                        "after_sim": sim + 1,
                        "score": round(self._best_so_far[0], 3),
                        "threshold": float(self.early_stop_threshold),
                    },
                )
                break

            if (self.light_cfg and self.light_cfg.enabled and self._first_emit_done
                and self.calls_used >= self.light_cfg.soft_budget_calls):
                break


        # choose best node if loop ended without updating best
        best_node = self._best_so_far[1] or (self._best_child(root, 0) if self.children[root.id] else root)

        best_score = float(best_node.score or 0.0)
        best_text  = "\n".join(best_node.trace)

        promoted = False
        if self._best_emitted:
            emitted_score, emitted_id = self._best_emitted
            margin = float(self.light_cfg.promote_margin if self.light_cfg else 0.0)
            if best_score >= emitted_score + margin:
                # write refined fragment
                rewards_vec = getattr(best_node, "dimension_scores", {}) or {"overall_norm": best_score}
                attrs = {
                    "role": "refined",
                    "strategy": self.search_strategy_name,
                    "run_id": context.get("run_id"),
                    "latency_ms": self._ms_since_start(),
                    "lm_calls_used": int(self.calls_used),
                    "score_overall": best_score,
                    "improves_on_fragment_id": emitted_id,
                    "improvement": best_score - emitted_score,
                }
                self.container["trajectory_store"].add_fragment(
                    case_id=context.get("case_id"),
                    source_type="paper",
                    section=context.get("section") or "unknown",
                    text=best_text,
                    attrs=attrs,
                    scores=rewards_vec,
                    uncertainty=None,
                )
                promoted = True

        # Log final
        self.logger.log("LightningSummary", {
            "first_emit_done": bool(self._first_emit_done),
            "best_emitted": self._best_emitted,
            "final_score": round(best_score, 3),
            "promoted": promoted
        })


        # Optional LM value estimator (for best node only; multi-candidate handled below)
        ve_score = None
        ve_rationale = None
        best_text = "\n".join(best_node.trace)
        if self.value_estimator_enabled and (not self.value_estimator_at_leaf_only or self._is_terminal(best_node)):
            try:
                ve = self.lats_program.value_estimator(state=best_node.state, trace=best_text, goal=goal_text)
                ve_score = float(getattr(ve, "score", 0.0) or 0.0)
                ve_rationale = getattr(ve, "rationale", None)
            except Exception as e:
                self.logger.log("MCTSValueEstimatorError", {"error": str(e)})

        # ---- MULTI-CANDIDATE EMISSION (top-K leaves) ----

        leaves = self._collect_leaves(root)
        for n in leaves:
            self._ensure_scored(n, context)

        leaves_sorted = sorted(leaves, key=lambda n: (n.score or 0.0), reverse=True)
        picked = leaves_sorted[: max(1, self.top_k)]

        scorables_out = []
        for rank, n in enumerate(picked, start=1):
            leaf_text = "\n".join(n.trace)
            # parse block (safe fallback to full text)
            try:
                parsed = parse_scored_block(leaf_text) or {}
            except Exception:
                parsed = {}
            content = parsed.get("content") or leaf_text

            sc = Scorable(
                id=n.id,
                text=content,
                target_type=self.scorable_type,
                meta={
                    "mcts": {
                        "rank": rank,
                        "best_node_score": float(n.score or 0.0),
                        "depth": int(len(n.trace)),
                        "nodes_explored": int(len(self.nodes)),
                        "lm_calls_used": int(getattr(self, "calls_used", 0)),
                    },
                    "llm_self_report": {
                        "score": parsed.get("llm_score"),
                        "rationale": parsed.get("rationale"),
                        "bullets": parsed.get("bullets"),
                    },
                    # only attach VE details to rank 1 to keep payloads small
                    "value_estimator": {
                        "enabled": self.value_estimator_enabled if rank == 1 else False,
                        "score": ve_score if rank == 1 else None,
                        "rationale": ve_rationale if rank == 1 else None,
                    },
                    "raw_block": leaf_text,
                },
            )
            scorables_out.append(sc.to_dict())

        context.setdefault(self.output_key, []).extend(scorables_out)

        # compact summary
        self.logger.log(
            "MCTSReasoningAgentComplete",
            {
                "goal": goal_text,
                "produced": len(scorables_out),
                "top_score": round(picked[0].score, 3) if picked else None,
                "top_depth": len(picked[0].trace) if picked else None,
                "nodes_explored": len(self.nodes),
                "lm_calls_used": int(getattr(self, "calls_used", 0)),
                "early_stop_hit": bool(
                    getattr(self, "early_stop_threshold", None) is not None
                    and self._best_so_far[0] >= float(self.early_stop_threshold)
                ),
            },
        )
        return context

    # ---------------- core MCTS ----------------
    def _create_node(self, state, trace, parent):
        node = MCTSReasoningNode(state, trace, parent)
        self.nodes.append(node)
        return node

    def _select(self, node):
        while self.children[node.id]:
            unvisited = [c for c in self.children[node.id] if c.visits == 0]
            if unvisited:
                return unvisited[0]
            node = self._best_child(node)
        return node

    async def _expand(self, node, context):
        children = []
        completions = []

        if self.expand_mode == "single_step":
            # one-step sampling, K children
            for _ in range(self.samples_per_expand):
                comp = self.next_step_once(node.state, node.trace)
                if not comp:
                    break
                completions.append(comp)
                child = self._create_node(
                    state=f"{node.state}\n{comp}",
                    trace=node.trace + [comp],
                    parent=node,
                )
                self.children[node.id].append(child)
                children.append(child)
        else:
            # light rollout: walk forward up to remaining depth, then take last B steps as siblings
            cur_trace = list(node.trace)
            rollout_steps = []
            remaining = max(0, self.max_depth - len(cur_trace))
            for _ in range(remaining):
                step = self._predict_next(node.state, cur_trace)
                if not step:
                    break
                rollout_steps.append(step)
                cur_trace.append(step)
            completions = (
                rollout_steps[-self.branching_factor :]
                if rollout_steps
                else []
            )
            for comp in completions:
                child = self._create_node(
                    state=f"{node.state}\n{comp}",
                    trace=node.trace + [comp],
                    parent=node,
                )
                self.children[node.id].append(child)
                children.append(child)

        self.logger.log(
            "MCTSExpand",
            {
                "node_id": node.id,
                "new_children": [c.id for c in children],
                "completions": completions,
            },
        )



        return children[0] if children else node

    def _should_eval(self, node: MCTSReasoningNode) -> bool:
        if self.eval_at == "leaf":
            return self._is_terminal(node)
        k = max(1, self.eval_stride)
        return (len(node.trace) % k) == 0

    def _evaluate(self, node, context):
        # skip eval if schedule says so
        if not self._should_eval(node):
            return 0.0

        text = "\n".join(node.trace)
        rewards_vec = {}
        try:
            if text in self.score_cache:
                score = float(self.score_cache[text] or 0.0)
            else:
                scorable = Scorable(text=text)
                bundle = self.container.get("scoring").score(
                    self.scorer_name,
                    scorable=scorable,
                    context=context,
                    dimensions=self.dimensions,
                )
                # guard aggregate() being None
                agg = bundle.aggregate()
                score = float(agg if agg is not None else 0.0) / 100.0
                self.score_cache[text] = score
                rewards_vec = bundle.to_rewards_vector(prefix="sicql_")
        except Exception as e:
            self.logger.log("MCTSEvaluateError", {"error": str(e)})
            score = 0.0

        air = 0.0
        children = self.children.get(node.id, [])
        if children:
            air = sum((c.score or 0.0) for c in children) / len(children)
        else:
            air = node.score or 0.0

        # Transition: evaluation at this node
        self.memory.trajectory.emit_transition(
                run_id=context.get("pipeline_run_id"),
                step_idx=len(self.nodes),
                agent=self.name,
                state={
                    "goal_preview": node.state[:200],
                    "trace_len": len(node.trace),
                    "strategy": self.search_strategy_name,
                    "node_id": node.id,
                    "event": "evaluate",
                },
                action={
                    "type": "score",
                    "name": self.scorer_name,
                    "strategy": self.search_strategy_name,
                    "output": {"text_len": len(text)},
                },
                reward_air=0.0,
                rewards_vec=rewards_vec,
            )
        node.score = score
        node.reward += score
        self.logger.log(
            "MCTSEvaluate",
            {
                "node_id": node.id,
                "trace_len": len(node.trace),
                "score": round(score, 3),
            },
        )
        return score

    def _backpropagate(self, node, reward):
        try:
            r = float(reward)
            if r != r:  # NaN check
                r = 0.0
        except Exception:
            r = 0.0

        while node:
            node.visits += 1
            node.reward += r
            node = node.parent

    def _best_child(self, node, ucb_weight=None):
        return max(
            self.children[node.id],
            key=lambda c: c.uct_value(
                node.visits, ucb_weight or self.ucb_weight
            ),
        )

    def _is_terminal(self, node):
        return len(node.trace) >= self.max_depth

    def _log_progress(self, sim, root):
        best = self._best_child(root, 0) if self.children[root.id] else root
        self.logger.log(
            "MCTSReasoningProgress",
            {
                "simulation": sim + 1,
                "percent_complete": f"{(sim + 1) / self.num_simulations * 100:.1f}%",
                "nodes_explored": len(self.nodes),
                "best_score": round(best.score, 3),
                "trace_preview": best.trace[-3:],
                "lm_calls_used": self.calls_used,
            },
        )

    def _collect_leaves(self, root) -> list:
        """DFS to collect nodes with no children (frontier leaves)."""
        leaves, stack = [], [root]
        while stack:
            n = stack.pop()
            kids = self.children.get(n.id, [])
            if not kids:
                leaves.append(n)
            else:
                stack.extend(kids)
        return leaves

    def _ensure_scored(self, node, context) -> float:
        """Guarantee node.score is a float. If missing, evaluate on-demand."""
        if node.score is not None and abs(node.score) > 1e-9:
            return float(node.score)
        text = "\n".join(node.trace)
        if text in self.score_cache:
            score = self.score_cache[text]
        else:
            sc = Scorable(text=text, target_type=self.scorable_type)
            bundle = self.container.get("scoring").score(
                self.scorer_name,
                scorable=sc,
                context=context,
                dimensions=self.dimensions,
            )
            score = float(bundle.aggregate()) / 100.0
            self.score_cache[text] = score
        node.score = score
        return score

    def _emit_lightning(self, context, best_node):
        text = "\n".join(best_node.trace)
        # score is already on node; recompute rewards_vec from bundle if you prefer
        rewards_vec = getattr(best_node, "dimension_scores", {}) or {"overall_norm": float(best_node.score or 0.0)}

        # guard by min score
        if self.light_cfg and rewards_vec.get("overall_norm", 0.0) < float(self.light_cfg.min_score):
            return None

        attrs = {
            "role": "lightning",
            "strategy": self.search_strategy_name,
            "run_id": context.get("run_id"),
            "latency_ms": self._ms_since_start(),
            "lm_calls_used": int(self.calls_used),
            "score_overall": float(rewards_vec.get("overall_norm", best_node.score or 0.0)),
        }
        frag = self.memory.trajectory.add_fragment(
            case_id=context.get("case_id"),
            source_type="paper",
            section=context.get("section") or "unknown",
            text=text,
            attrs=attrs,
            scores=rewards_vec,
            uncertainty=None,
        )

        # transition
        self._step_idx += 1
        self.container["trajectory_store"].emit_transition(
            run_id=context.get("run_id"),
            step_idx=self._step_idx,
            agent=self.__class__.__name__,
            state={"event": "emit_lightning", "node_id": best_node.id, "trace_len": len(best_node.trace)},
            action={"type": "emit", "name": "lightning", "strategy": self.search_strategy_name, "output": {"fragment_id": frag.id}},
            reward_air=0.0,
            rewards_vec=rewards_vec,
        )

        # track best emitted
        score = float(rewards_vec.get("overall_norm", 0.0))
        if not self._first_emit_done:
            self._first_emit_done = True
            self._best_emitted = (score, frag.id)
        elif self._best_emitted and score > self._best_emitted[0]:
            self._best_emitted = (score, frag.id)
        return frag.id
