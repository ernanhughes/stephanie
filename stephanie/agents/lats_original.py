# stephanie/agents/lats_original.py
import math
import re
from collections import defaultdict

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.agents.proximity import ProximityAgent
from stephanie.agents.rule_tuner import RuleTunerAgent
from stephanie.agents.unified_mrq import UnifiedMRQAgent
from stephanie.analysis.symbolic_impact_analyzer import SymbolicImpactAnalyzer
from stephanie.constants import GOAL
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.graph_tools import (build_mermaid_graph,
                                         save_mermaid_to_file)
from stephanie.scoring.scorable_factory import ScorableFactory

class LATSAgent(ScoringMixin, BaseAgent):
    """
    Enhanced LATS agent with:
    - Tree search (MCTS + UCT)
    - Multi-dimensional scoring
    - Proximity-based reuse
    - Reflection/refinement
    - Rule tuning
    - Mermaid visualization
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        # Configuration
        self.max_depth = cfg.get("max_depth", 5)
        self.branching_factor = cfg.get("branching_factor", 3)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)
        self.lambda_weight = cfg.get("lambda", 0.5)
        self.min_score_threshold = cfg.get("min_score_threshold", 0.7)
        self.prune_threshold = cfg.get("prune_threshold", 0.4)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)

        # Node tracking
        self.nodes = []
        self.N = defaultdict(int)  # visit count
        self.W = defaultdict(float)  # total reward
        self.children = dict()  # node -> children

        # Initialize sub-agents
        self.proximity_agent = ProximityAgent(
            cfg.get("proximity", {}), memory=memory, logger=logger
        )
        self.rule_tuner = RuleTunerAgent(
            cfg.get("rule_tuner", {}), memory=memory, logger=logger
        )
        self.mrq_agent = UnifiedMRQAgent(
            cfg.get("mrq", {}), memory=memory, logger=logger
        )
        self.impact_analyzer = SymbolicImpactAnalyzer(self._get_score)

    async def run(self, context: dict) -> dict:
        """
        Main LATS search loop
        """
        goal = context[GOAL]
        root_state = {
            "goal": goal["goal_text"],
            "current": goal["goal_text"],
            "trace": [],
        }

        # Initialize root node
        root = self.create_node(state=root_state, trace=[], parent=None)

        # Run MCTS simulations
        best_scores = []
        for sim_num in range(self.num_simulations):
            # Selection
            node = self.select(root)

            # Expansion
            if not self.is_terminal(node):
                node = await self.expand(node, context)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

            # Track progress
            current_best = self._get_best_score(root)
            best_scores.append(current_best)

            # ✅ Log Mermaid graph after each simulation
            if sim_num % 5 == 0:  # Every 5 simulations
                percent_complete = (sim_num + 1) / self.num_simulations * 100
                self.logger.log(
                    "Progress",
                    {
                        "simulation": sim_num + 1,
                        "percent_complete": f"{percent_complete:.1f}%",
                        "best_score": self._get_best_score(root),
                    },
                )
                mermaid_lines = build_mermaid_graph(root, max_depth=3)
                mermaid_diagram = "\n".join(mermaid_lines)
                self.logger.log("SearchTree", {"diagram": mermaid_diagram})

            # Optional: Periodic refinement
            if sim_num % 10 == 0:
                await self._refine_system(context)

        # 3. Get best path
        best_child = self.best_uct(node=root, ucb_weight=0)  # Greedy selection
        best_trace = best_child["trace"]

        # ✅ Final Mermaid visualization
        mermaid_lines = build_mermaid_graph(best_child, max_depth=5)
        mermaid_diagram = "\n".join(mermaid_lines)
        self.logger.log("FinalSearchTree", {"diagram": mermaid_diagram})
        save_mermaid_to_file(mermaid_diagram, "final_search_tree.mmd")

        # Reconstruct merged context for prompt
        merged_for_prompt = {
            "state": best_child["state"]["current"]
            if isinstance(best_child["state"], dict)
            else best_child["state"],
            "trace": best_trace,
            "mode": "reason",  # Or use context.get("mode", "reason")
        }
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged_for_prompt)
        prompt_id = self.memory.prompt.get_id_from_response(prompt_text)

        # Safely extract scores
        dimension_scores = best_child.get("dimension_scores", {})

        # 4. Create final hypothesis using save_hypothesis
        hypothesis = self.save_hypothesis(
            {
                "prompt_id": prompt_id,
                "text": "\n".join(best_trace),
                "metadata": {
                    "trace": best_trace,
                    "path": [n["id"] for n in best_trace],
                    "scores": {
                        dim: data["score"] for dim, data in dimension_scores.items()
                    },
                    "score": best_child.get("score", 0.0),
                },
            },
            context=context,
        )
        context.setdefault("lats_result", []).append(hypothesis.to_dict())
        context.setdefault("hypotheses", []).append(hypothesis.to_dict())
        # Refine system using analysis
        await self._refine_system(context)

        return context

    def create_node(self, state, trace, parent=None):
        """Create a new node in the search tree"""

        # Ensure trace is always a list
        if isinstance(trace, str):
            print("Trace is not right")
            trace = trace.split("\n")  # Convert string to list
        elif not isinstance(trace, list):
            print("Trace is not right")
            trace = [str(trace)]  # Fallback

        node = {
            "id": len(self.nodes) + 1,
            "state": state,
            "trace": trace,
            "parent": parent,
            "visits": 0,
            "reward": 0.0,
            "children": [],
            "is_terminal": False,
            "dimension_scores": {},
            "final_score": 0.0,
        }
        self.nodes.append(node)
        return node

    def select(self, node):
        self._log_node(node, level="debug")
        """Select node for expansion using UCT"""
        while self.children.get(id(node)) and self.children[id(node)]:
            unvisited = [c for c in self.children[id(node)] if c["visits"] == 0]
            if unvisited:
                return unvisited[0]
            node = self.best_uct(node)
        return node

    def best_uct(self, node, ucb_weight=None):
        """Select best child using UCT formula"""
        ucb_weight = ucb_weight or self.ucb_weight

        def uct(child):
            if child["visits"] == 0:
                return float("inf")
            return (child["reward"] / child["visits"]) + ucb_weight * math.sqrt(
                math.log(node["visits"]) / child["visits"]
            )

        return max(self.children[id(node)], key=uct)

    async def expand(self, node, context: dict):
        """
        Generate new children nodes from current node
        """
        # Build prompt with context
        merged = {
            **context,
            "state": node["state"]["current"],
            "trace": node["trace"],
            "mode": "reason",
        }

        # Get similar hypotheses
        proximity_context = await self._run_proximity(context)
        merged["similar_hypotheses"] = proximity_context.get("most_similar", "")

        # Load prompt
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt_text, context=merged)

        prompt = self.memory.prompt.get_from_text(prompt_text)
        if not prompt:
            goal = context.get("goal")
            prompt_id = self.memory.prompt.save(
                goal, self.name, self.name, prompt_text, response
            )
        else:
            prompt_id = prompt.id
        # Parse completions
        completions = self._parse_completions(response)
        if not completions:
            self.logger.log(
                "NoCompletions", {"prompt": prompt_text, "response": response}
            )
            return node

        # Create child nodes
        children = []
        for comp in completions:
            new_state = self._update_state(node["state"], comp)
            new_trace = node["trace"] + [comp]

            # Score hypothesis
            hyp = self.save_hypothesis(
                {
                    "prompt_id": prompt_id,
                    "strategy": "lats",
                    "text": comp,
                },
                context=context,
            )

            # Add mode to context
            score_context = {**context, "mode": "reason"}
            context.setdefault("hypotheses", []).append(hyp.to_dict())
            # Score using dimensional scorers
            scorable = ScorableFactory.from_dict(hyp, TargetType.HYPOTHESIS)
            score_result = self.score_item(scorable, score_context, metrics="lats_node")

            # Create child node with metadata
            child = self.create_node(new_state, new_trace, parent=node)
            child["score"] = score_result.aggregate()
            child["dimension_scores"] = score_result.to_dict()
            child["action"] = comp

            children.append(child)

        # Store children
        self.children[id(node)] = children
        return children[0]  # Return first child

    async def _run_proximity(self, context):
        """
        Run proximity agent to find similar hypotheses
        """
        try:
            return await self.proximity_agent.run(context)
        except Exception as e:
            self.logger.log("ProximityAgentFailed", {"error": str(e)})
            return {}

    def _parse_completions(self, response: str) -> list:
        """
        Safely parse completions from LLM response
        """
        thought_pattern = r"(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s*(.*?)(?=\n(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        if not matches:
            return [response.strip()]

        completions = [match.strip() for match in matches if match.strip()]
        return completions[: self.branching_factor]

    def _update_state(self, state_dict, action: str) -> dict:
        """
        Updates state dictionary with new action
        """
        new_state = state_dict.copy()
        new_state["current"] = state_dict["current"] + "\n" + action
        new_state["trace"] = state_dict["trace"] + [action]
        return new_state

    def is_terminal(self, node):
        """
        Check if node is terminal state
        Works with both string and dict state formats
        """
        state = node["state"]

        # Handle structured state dict
        if isinstance(state, dict):
            current_text = state.get("current", "")
            return (
                "success" in current_text.lower()
                or len(node["trace"]) >= self.max_depth
            )

        # Fallback for string-based state
        return "success" in state.lower() or len(node["trace"]) >= self.max_depth

    def simulate(self, node):
        """
        Simulate until terminal state
        """
        current = node
        while not self.is_terminal(current) and len(current["trace"]) < self.max_depth:
            prompt = self._build_prompt(current, mode="simulate", context=current)
            response = self.call_llm(prompt, context=current)
            completions = self._parse_completions(response)

            if not completions:
                break

            action = completions[0]  # Take first completion
            new_state = self._update_state(current["state"], action)
            current = self.create_node(
                new_state, current["trace"] + [action], parent=current
            )

        # Evaluate final node
        return self._get_value(current)

    def _build_prompt(self, node, mode="reason", context=None):
        """
        Build prompt for node evaluation
        """
        merged = {
            **context,
            "state": node["state"]["current"],
            "trace": node["trace"],
            "mode": mode,
        }
        prompt_text = self.prompt_loader.load_prompt(self.cfg, merged)
        return prompt_text

    async def _run_proximity(self, context):
        """Run proximity agent to find similar hypotheses"""
        try:
            return await self.proximity_agent.run(context)
        except Exception as e:
            self.logger.log("ProximityAgentFailed", {"error": str(e)})
            return {}

    def _apply_proximity_guidance(self, comp, proximity_data):
        """Enhance completion using proximity feedback"""
        if not proximity_data.get("most_similar"):
            return comp

        # Use LLM to refine action with proximity info
        prompt = self.prompt_loader.load_prompt(
            "proximity_guidance",
            {
                "current_action": comp,
                "similar_hypotheses": proximity_data["most_similar"],
            },
        )

        response = self.call_llm(prompt, {})
        return response.strip()

    def _get_value(self, node):
        """
        Hybrid value function: LLM score + self-consistency
        """
        # Use LLM-powered scorer
        score_result = self._evaluate_node(node)

        # Self-consistency check
        sc_score = self._self_consistency(node)

        return (
            self.lambda_weight * score_result["score"]
            + (1 - self.lambda_weight) * sc_score
        )

    def _evaluate_node(self, node):
        """
        Evaluate node using dimensional scorers
        """
        scorable = Scorable(text="\n".join(node["trace"]))
        return self.score_item(
            scorable,
            {"goal": {"goal_text": node["state"]["goal"]}},
            metrics="lats_reflection",
        )

    def _self_consistency(self, node):
        """
        Calculate self-consistency score
        """
        prompt = self.prompt_loader.load_prompt(
            "self_consistency",
            {"trace": node["trace"], "state": node["state"]["current"]},
        )
        response = self.call_llm(prompt, {})

        # Parse numerical score
        score_match = re.search(r"(\d+)", response)
        return int(score_match.group(1)) / 100 if score_match else 0.5

    def backpropagate(self, node, reward):
        """
        Update node statistics up the tree
        """
        while node:
            node["visits"] += 1
            node["reward"] += reward

            # Store trace data for analysis
            if "history" not in node:
                node["history"] = []
            node["history"].append({"visits": node["visits"], "reward": reward})

            node = node["parent"]

    async def _refine_system(self, context):
        if len(self.nodes) > 1:
            # Pass full node dicts to analyzer
            analysis = self.impact_analyzer.analyze(
                self.nodes[: len(self.nodes) // 2],  # First half of nodes
                self.nodes[len(self.nodes) // 2 :],  # Second half
            )
            context["graph_analysis"] = analysis

        # Train MR.Q on high-quality traces
        high_scoring = [n for n in self.nodes if n.get("score", 0) > 0.8]
        if high_scoring:
            await self.mrq_agent.run({"traces": high_scoring})

        # Tune rules based on analysis
        if context.get("graph_analysis"):
            await self.rule_tuner.run(context)

        return context

    def _get_score(self, node, source="graph1"):
        """
        Score node using hypothesis-based evaluator
        """
        trace = node.get("trace", [])
        if isinstance(trace, str):
            trace = trace.split("\n")
        elif not isinstance(trace, list):
            trace = []

        hyp = {
            "text": "\n".join(trace),
            "goal_id": node["state"]["goal_id"]
            if isinstance(node["state"], dict)
            else "unknown",
        }

        goal_text = (
            node["state"]["goal"] if isinstance(node["state"], dict) else node["state"]
        )
        scorable = Scorable(text=hyp["text"])
        score_result = self.score_item(
            scorable,
            {"goal": {"goal_text": goal_text}},  # Always a dict
            metrics="lats_reflection",
        )

        return score_result.aggregate() / 100  # Normalize

    def _log_progress(self, sim_num, best_score, best_trace):
        percent_complete = (sim_num + 1) / self.num_simulations * 100
        self.logger.log(
            "LATSProgress",
            {
                "simulation": sim_num + 1,
                "percent_complete": f"{percent_complete:.1f}%",
                "best_score": best_score,
                "best_trace": [step[:20] + "..." for step in best_trace],
                # "node_id": best_node.get("id", "unknown")
            },
        )

    def _should_stop_early(self, score_history):
        if len(score_history) < 5:
            return False

        # Check for score plateau
        recent_scores = score_history[-5:]
        score_variance = max(recent_scores) - min(recent_scores)
        if score_variance < 0.01:
            return True

        # Check for high score
        if score_history[-1] >= self.cfg.get("threshold", 0.95):
            return True

        return False

    def _log_node(self, node, level="debug"):
        """
        Logs a structured representation of a node for debugging
        """
        # Safely extract parent ID
        parent_info = node.get("parent")
        parent_id = "none"
        if parent_info is not None:
            parent_id = (
                getattr(parent_info, "id", parent_info.get("id", "none"))
                if isinstance(parent_info, (dict, object))
                else "none"
            )

        # Safely extract state string
        state_data = node.get("state", {})
        if isinstance(state_data, dict):
            state_str = state_data.get("current", "")
        else:
            state_str = str(state_data)

        # Safely extract trace
        trace = node.get("trace", [])

        # Build node info for logging
        node_info = {
            "id": node.get("id", "unknown"),
            "visits": node.get("visits", 0),
            "reward": round(float(node.get("reward", 0.0)), 2),
            "depth": len(trace),
            "is_terminal": node.get("is_terminal", False),
            "trace_preview": trace[-3:] if trace else [],
            "state_preview": state_str[:200],  # Now safe for both string and dict
            "parent_id": parent_id,
            "child_count": len(node.get("children", [])),
            "score": round(float(node.get("score", 0.0)), 2),
            "dimension_scores": {
                dim: {
                    "score": round(float(score.get("score", 0)), 2),
                    "rationale": score.get("rationale", "")[:100],
                }
                for dim, score in node.get("dimension_scores", {}).items()
            }
            if "dimension_scores" in node
            else {},
        }

        # Log based on level
        if level == "debug":
            self.logger.log(
                "NodeDebug", {"node_id": node.get("id"), "node_info": node_info}
            )
        elif level == "info":
            self.logger.log(
                "NodeSummary",
                {
                    "node_id": node.get("id"),
                    "summary": self._format_node_summary(node_info),
                },
            )
        return node_info

    def _apply_reflection_to_prompt(self, prompt, reflection):
        """
        Inject reflection into prompt for future steps
        """
        if not reflection:
            return prompt

        reflection_prompt = self.prompt_loader.load_prompt(
            "reflection_injection", {"prompt": prompt, "reflection": reflection}
        )

        return self.call_llm(reflection_prompt, {})

    def _train_on_traces(self, traces):
        """
        Train agent on high-quality traces
        """
        examples = [
            {
                "state": trace["state"],
                "trace": trace["trace"],
                "next_step": trace["last_action"],
            }
            for trace in traces
        ]

        # Use dimensional scores as weights
        weighted_examples = [
            example for example in examples if self._is_high_quality(example)
        ]

        if not weighted_examples:
            return

        # Re-train on high-quality traces
        self._retrain_on_examples(weighted_examples)

    def _is_high_quality(self, trace):
        """
        Check if trace meets quality threshold
        """
        return trace["score"] > self.min_score_threshold

    def _retrain_on_examples(self, examples):
        """
        Retrain agent on high-quality examples
        """
        # Implement your own training logic here
        pass
