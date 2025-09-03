# stephanie/agents/dspy/mcts_reasoning_agent.py
"""
MCTSReasoningAgent (enhanced, patched)
"""

import math
import uuid
import logging
from collections import defaultdict, OrderedDict

import dspy
from dspy import Predict, Signature, InputField, OutputField

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.llm_response_parser import parse_scored_block

logger = logging.getLogger(__name__)


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
    goal  = InputField(desc="Goal text")
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
                logger.debug("=== DSPy PROMPT ===\n%s\n====================", prompt)
            if messages:
                logger.debug("=== DSPy MESSAGES ===\n%s\n====================", messages)
        result = super().__call__(*args, **kwargs)
        if self._debug_prompts:
            logger.debug("=== DSPy RESPONSE ===\n%s\n====================", result)
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
        exploration = ucb_weight * math.sqrt(max(1e-9, math.log(max(1, parent_visits)) / self.visits))
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
    def __init__(self, cfg, memory, log):
        super().__init__(cfg, memory, log)

        # search / scoring knobs
        self.max_depth = int(cfg.get("max_depth", 4))
        self.branching_factor = int(cfg.get("branching_factor", 2))
        self.num_simulations = int(cfg.get("num_simulations", 20))
        self.ucb_weight = float(cfg.get("ucb_weight", 1.41))

        self.scorer_name = cfg.get("scorer_name", "sicql")
        self.scorable_type = cfg.get("scorable_type", TargetType.HYPOTHESIS)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "implementability", "novelty", "relevance"])

        # evaluation schedule
        self.eval_at = str(cfg.get("eval_at", "leaf"))   # "leaf" | "every_k"
        self.eval_stride = int(cfg.get("eval_stride", 2))
        ve_cfg = cfg.get("value_estimator", {}) or {}
        self.value_estimator_enabled = bool(ve_cfg.get("enabled", False))
        self.value_estimator_at_leaf_only = bool(ve_cfg.get("at_leaf_only", True))

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
        self.expand_mode = str(cfg.get("expand_mode", "single_step"))  # "single_step" | "rollout"
        self.samples_per_expand = int(cfg.get("samples_per_expand", max(1, cfg.get("branching_factor", 2))))

        # budgets / caching (kept on the agent)
        self.max_lm_calls = int(cfg.get("max_lm_calls", 12))
        self.calls_used = 0
        cache_cfg = cfg.get("caching", {}) or {}
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        self.cache_size = int(cache_cfg.get("lru_size", 2048))
        self._cache = OrderedDict()
        self._cache_tail_len = int(cache_cfg.get("tail_len", 6))

        # early stopping if we hit an acceptable score
        t = cfg.get("early_stop_threshold", None)                # e.g., 0.75 or 75
        self.early_stop_threshold = None if t is None else float(t)
        if self.early_stop_threshold is not None and self.early_stop_threshold > 1.0:
            self.early_stop_threshold /= 100.0                  # accept 75 as 0.75

        self._best_so_far = (float("-inf"), None)               # (score, node)


        logger.debug(
            "MCTSInitialized depth=%s bf=%s sims=%s ucb=%s dims=%s max_lm_calls=%s eval_at=%s stride=%s",
            self.max_depth, self.branching_factor, self.num_simulations, self.ucb_weight,
            self.dimensions, self.max_lm_calls, self.eval_at, self.eval_stride
        )

    # ---------------- cache helpers ----------------
    def _cache_key(self, state: str, trace: list[str]) -> tuple:
        return (hash(state), tuple(trace[-self._cache_tail_len:]))

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
        nxt = pred.next_step.strip() if (pred and getattr(pred, "next_step", None)) else ""
        self._cache_put(key, nxt)
        return nxt

    def next_step_once(self, state: str, trace: list[str]) -> str:
        return self._predict_next(state, trace)

    # ---------------- main entry ----------------
    async def run(self, context: dict) -> dict:
        goal_text = (context.get("goal") or {}).get("goal_text", "") or ""
        root = self._create_node(state=goal_text, trace=[], parent=None)
        self.logger.log("MCTSReasoningAgentStart", {"goal": goal_text})

        for sim in range(self.num_simulations):
            node = self._select(root)
            if not self._is_terminal(node):
                node = await self._expand(node, context)
            reward = self._evaluate(node, context)
            self._backpropagate(node, reward)
            if (sim + 1) % 5 == 0:
                self._log_progress(sim, root)
            # early stopping if we hit an acceptable score
            if (
                self.early_stop_threshold is not None
                and self._best_so_far[0] >= self.early_stop_threshold
            ):
                self.logger.log("MCTSEarlyStop", {
                    "after_sim": sim + 1,
                    "score": round(self._best_so_far[0], 3),
                    "threshold": self.early_stop_threshold,
                })
                break


        best_node = (
            self._best_so_far[1]
            or (self._best_child(root, 0) if self.children[root.id] else root)
        )
        best_text = "\n".join(best_node.trace)

        # optional LM value estimator
        ve_score = None
        ve_rationale = None
        if self.value_estimator_enabled and (not self.value_estimator_at_leaf_only or self._is_terminal(best)):
            try:
                ve = self.lats_program.value_estimator(state=best.state, trace=best_text, goal=goal_text)
                ve_score = float(getattr(ve, "score", 0.0) or 0.0)
                ve_rationale = getattr(ve, "rationale", None)
            except Exception as e:
                self.logger.log("MCTSValueEstimatorError", {"error": str(e)})

        # parse block (safe fallback to full text)
        try:
            parsed = parse_scored_block(best_text) or {}
        except Exception:
            parsed = {}
        parsed_content = parsed.get("content") or best_text

        result = Scorable(
            id=best.id,
            text=parsed_content,
            target_type=self.scorable_type,
            metadata={
                "mcts": {
                    "best_node_score": float(best.score),
                    "depth": int(len(best.trace)),
                    "nodes_explored": int(len(self.nodes)),
                    "lm_calls_used": int(self.calls_used),
                },
                "llm_self_report": {
                    "score": parsed.get("llm_score"),
                    "rationale": parsed.get("rationale"),
                    "bullets": parsed.get("bullets"),
                },
                "value_estimator": {
                    "enabled": self.value_estimator_enabled,
                    "score": ve_score,
                    "rationale": ve_rationale,
                },
                "raw_block": best_text,
            },
        )

        context.setdefault(self.output_key, []).append(result.to_dict())
        self.logger.log("MCTSReasoningAgentComplete", {
            "goal": goal_text,
            "best_score": round(best.score, 3),
            "trace_length": len(best.trace),
            "nodes_explored": len(self.nodes),
            "lm_calls_used": self.calls_used,
        })
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
            completions = rollout_steps[-self.branching_factor:] if rollout_steps else []
            for comp in completions:
                child = self._create_node(
                    state=f"{node.state}\n{comp}",
                    trace=node.trace + [comp],
                    parent=node,
                )
                self.children[node.id].append(child)
                children.append(child)

        self.logger.log("MCTSExpand", {
            "node_id": node.id,
            "new_children": [c.id for c in children],
            "completions": completions,
        })
        return children[0] if children else node

    def _should_eval(self, node: MCTSReasoningNode) -> bool:
        if self.eval_at == "leaf":
            return self._is_terminal(node)
        k = max(1, self.eval_stride)
        return (len(node.trace) % k) == 0

    def _evaluate(self, node, context):
        if not self._should_eval(node):
            return 0.0

        text = "\n".join(node.trace)
        if text in self.score_cache:
            score = self.score_cache[text]
        else:
            scorable = Scorable(text=text)
            bundle = self.scoring.score(
                self.scorer_name,
                scorable=scorable,
                context=context,
                dimensions=self.dimensions,
            )
            score = float(bundle.aggregate()) / 100.0  # normalize to 0..1
            self.score_cache[text] = score

            node.score = score
            node.reward += score

            # track global best
            if score > self._best_so_far[0]:
                self._best_so_far = (score, node)

            self.logger.log("MCTSEvaluate", {
                "node_id": node.id,
                "trace_len": len(node.trace),
                "score": round(score, 3),
            })
            return score

    def _backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _best_child(self, node, ucb_weight=None):
        return max(self.children[node.id], key=lambda c: c.uct_value(node.visits, ucb_weight or self.ucb_weight))

    def _is_terminal(self, node):
        return len(node.trace) >= self.max_depth

    def _log_progress(self, sim, root):
        best = self._best_child(root, 0) if self.children[root.id] else root
        self.logger.log("MCTSReasoningProgress", {
            "simulation": sim + 1,
            "percent_complete": f"{(sim + 1) / self.num_simulations * 100:.1f}%",
            "nodes_explored": len(self.nodes),
            "best_score": round(best.score, 3),
            "trace_preview": best.trace[-3:],
            "lm_calls_used": self.calls_used,
        })
