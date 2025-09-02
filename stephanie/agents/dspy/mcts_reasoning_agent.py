# stephanie/agents/dspy/mcts_reasoning_agent.py
"""
MCTSReasoningAgent

This agent implements a hybrid **Monte Carlo Tree Search (MCTS)** reasoning system
guided by **LLM expansions** (via DSPy) and evaluated using Stephanie's scoring stack.

- Search backbone: Monte Carlo Tree Search (UCT selection, expansion, simulation, backpropagation)
- Expansion: DSPy LLM calls generate next reasoning steps
- Evaluation: Stephanie scorers (e.g., SICQL) score partial reasoning traces
- Output: The best reasoning path is returned as a hypothesis

This is a *foundation agent* you can extend with:
- CBR (Case-Based Reasoning) hooks
- Reflection / revision cycles
- Cross-agent comparisons (Memento paper setup)

Author: Ernan + ChatGPT
"""

import math
import re
import uuid
import asyncio
import time
from collections import defaultdict
import logging

import dspy
from dspy import Predict, Signature, InputField, OutputField

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# DSPy Signatures
# -------------------------------------------------------------------------
class TraceStep(Signature):
    """Defines the prediction of the *next reasoning step* given state + trace"""
    state = InputField(desc="Current problem state")
    trace = InputField(desc="History of thoughts/actions so far")
    next_step = OutputField(desc="Next reasoning step")


class ValueEstimator(Signature):
    """Estimates value of a reasoning path (LM-powered heuristic score)"""
    state = InputField(desc="Current problem state")
    trace = InputField(desc="Reasoning steps taken")
    goal = InputField(desc="Goal text")
    score = OutputField(desc="Normalized score (0–1)")
    rationale = OutputField(desc="Why this path is promising")


class LoggingLM(dspy.LM):
    """
    Wrapper around DSPy LM that logs requests and responses.
    Helpful for debugging model behavior inside the search loop.
    """
    def __call__(self, *args, **kwargs):
        logger.debug("📡 Sending request to LLM...")
        result = super().__call__(*args, **kwargs)
        logger.debug(f"✅ Received response: {result}")
        return result

# -------------------------------------------------------------------------
# Node + Program
# -------------------------------------------------------------------------
class MCTSReasoningNode:
    """
    Represents a node in the reasoning tree.
    Stores state, trace, statistics, children, and scoring info.
    """
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
        """UCT formula: exploitation + exploration"""
        if self.visits == 0:
            return float("inf")
        exploitation = self.reward / self.visits
        exploration = ucb_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class MCTSReasoningProgram(dspy.Module):
    """
    DSPy program wrapper used for *step generation* (TraceStep)
    and optional value estimation (ValueEstimator).
    """
    def __init__(self, cfg):
        super().__init__()
        self.generator = Predict(TraceStep)
        self.value_estimator = Predict(ValueEstimator)
        self.max_depth = cfg.get("max_depth", 3)

    def __call__(self, state, trace, depth=0):
        """
        Recursive forward generation until max depth reached.
        Returns a trace of reasoning steps.
        """
        if depth >= self.max_depth:
            return trace, 0.5
        prediction = self.generator(state=state, trace=trace)
        if not prediction or not prediction.next_step:
            return trace, 0.0
        next_step = prediction.next_step.strip()
        new_trace = trace + [next_step]
        return self.__call__(state, new_trace, depth + 1)

# -------------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------------
class MCTSReasoningAgent(BaseAgent):
    """
    Monte Carlo Tree Search Reasoning Agent.

    Combines:
    - MCTS for structured exploration of reasoning paths
    - DSPy LLM expansions for step generation
    - Stephanie scorers for reward evaluation

    Outputs:
    - Best-scoring hypothesis (reasoning trace)
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.max_depth = cfg.get("max_depth", 4)
        self.branching_factor = cfg.get("branching_factor", 2)
        self.num_simulations = cfg.get("num_simulations", 20)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.scorer_name = cfg.get("scorer_name", "sicql")
        self.dimensions = cfg.get(
            "dimensions",
            ["alignment", "clarity", "implementability", "novelty", "relevance"],
        )

        self.nodes = []
        self.children = defaultdict(list)
        self.score_cache = {}
        self.lats_program = MCTSReasoningProgram(cfg)

        # Configure DSPy LM
        model_config = cfg.get("model", {})
        model_name = model_config.get("name", "ollama_chat/qwen3")
        api_base = model_config.get("api_base", "http://localhost:11434")
        api_key = model_config.get("api_key", "")

        lm = LoggingLM(model_name, api_base=api_base, api_key=api_key)
        dspy.configure(lm=lm)

    # ---------------------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------------------
    async def run(self, context: dict) -> dict:
        """
        Main entry point. Runs MCTS simulations to explore reasoning paths.
        Returns best-scoring hypothesis added to context.
        """
        goal = context["goal"]["goal_text"]
        root = self._create_node(state=goal, trace=[], parent=None)

        self.logger.info("MCTSReasoningAgentStart", {"goal": goal})

        for sim in range(self.num_simulations):
            node = self._select(root)
            if not self._is_terminal(node):
                node = await self._expand(node, context)
            reward = self._evaluate(node, context)
            self._backpropagate(node, reward)

            if sim % 5 == 0:
                self._log_progress(sim, root)

        best = self._best_child(root, 0)
        best_hypothesis = "\n".join(best.trace)
        result = {"text": best_hypothesis, "score": best.score}

        # Update context
        context.setdefault("hypotheses", []).append(result)

        self.logger.info("MCTSReasoningAgentComplete", {
            "goal": goal,
            "best_score": round(best.score, 3),
            "trace_length": len(best.trace),
            "nodes_explored": len(self.nodes)
        })

        return context

    # ---------------------------------------------------------------------
    # MCTS Core Methods
    # ---------------------------------------------------------------------
    def _create_node(self, state, trace, parent):
        node = MCTSReasoningNode(state, trace, parent)
        self.nodes.append(node)
        return node

    def _select(self, node):
        """Selection step: choose node with best UCT value"""
        while self.children[node.id]:
            unvisited = [c for c in self.children[node.id] if c.visits == 0]
            if unvisited:
                return unvisited[0]
            node = self._best_child(node)
        return node

    async def _expand(self, node, context):
        """Expansion step: use DSPy to generate completions and add children"""
        trace, _ = self.lats_program(node.state, node.trace)
        completions = trace[-self.branching_factor :]
        children = []
        for comp in completions:
            child = self._create_node(
                state=f"{node.state}\n{comp}", trace=node.trace + [comp], parent=node
            )
            self.children[node.id].append(child)
            children.append(child)

        self.logger.debug("MCTSExpand", {
            "node_id": node.id,
            "new_children": [c.id for c in children],
            "completions": completions
        })

        return children[0] if children else node

    def _evaluate(self, node, context):
        """Evaluation step: score reasoning trace using Stephanie scorers"""
        text = "\n".join(node.trace)
        if text in self.score_cache:
            score = self.score_cache[text]
        else:
            scorable = Scorable(text=text)
            score_result = self.scoring.score(
                self.scorer_name, scorable=scorable, context=context, dimensions=self.dimensions
            )
            score = score_result.aggregate() / 100
            self.score_cache[text] = score
        node.score = score
        node.reward += score

        self.logger.debug("MCTSEvaluate", {
            "node_id": node.id,
            "trace_preview": node.trace[-2:],
            "score": round(score, 3)
        })

        return score

    def _backpropagate(self, node, reward):
        """Backpropagation step: update node statistics up the tree"""
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _best_child(self, node, ucb_weight=None):
        """Choose best child using UCT (exploration vs exploitation)"""
        return max(
            self.children[node.id],
            key=lambda c: c.uct_value(node.visits, ucb_weight or self.ucb_weight)
        )

    def _is_terminal(self, node):
        """Terminal if max depth reached"""
        return len(node.trace) >= self.max_depth

    def _log_progress(self, sim, root):
        """Periodic logging of progress"""
        best = self._best_child(root, 0) if self.children[root.id] else root
        msg = {
            "simulation": sim + 1,
            "percent_complete": f"{(sim+1)/self.num_simulations*100:.1f}%",
            "nodes_explored": len(self.nodes),
            "best_score": round(best.score, 3),
            "trace_preview": best.trace[-3:],
        }
        self.logger.info("MCTSReasoningProgress", msg)
