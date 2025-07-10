import math
import re
from collections import defaultdict

import dspy
from dspy import (BootstrapFewShot, Example, InputField, OutputField, Predict,
                  Signature)

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.agents.proximity import ProximityAgent
from stephanie.agents.rule_tuner import RuleTunerAgent
from stephanie.agents.unified_mrq import UnifiedMRQAgent
from stephanie.constants import GOAL
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.scoring.svm_scorer import SVMScorer
from stephanie.utils.graph_tools import (build_mermaid_graph, compare_graphs,
                                         save_mermaid_to_file)


class TraceStep(Signature):
    """
    A reasoning step in the LATS framework.

    Inputs:
        - state: Current problem state (e.g., goal + history)
        - trace: Sequence of previous thoughts/actions

    Outputs:
        - next_step: Next thought/action to explore
    """

    state = InputField(desc="Current problem state")
    trace = InputField(desc="History of thoughts/actions taken so far")
    next_step = OutputField(desc="Next reasoning step (thought or action)")


class ReflectionPrompt(Signature):
    """
    Self-reflection module to analyze failed reasoning paths.

    Inputs:
        - state: Final state after failed attempt
        - trace: Full reasoning path
        - goal: Original goal text

    Outputs:
        - rationale: Explanation of failure
        - improvement_plan: Suggested improvements
    """

    state = InputField(desc="Final state after failed attempt")
    trace = InputField(desc="Full reasoning path")
    goal = InputField(desc="Original goal text")

    rationale = OutputField(desc="Why the attempt failed")
    improvement_plan = OutputField(desc="Concrete steps to improve")


class ValueEstimator(Signature):
    """
    Evaluates a reasoning path using a hybrid value function.

    Inputs:
        - state: Current problem state
        - trace: Reasoning steps taken
        - goal: Goal text

    Outputs:
        - score: Normalized score (0–1)
        - rationale: Explanation of the score
    """

    state = InputField(desc="Current problem state")
    trace = InputField(desc="Sequence of thoughts/actions")
    goal = InputField(desc="Goal text")

    score = OutputField(desc="Hybrid score (LM + self-consistency)")
    rationale = OutputField(desc="Explanation of score")


class SharpeningPrompt(Signature):
    """
    Sharpens hypotheses using dimensional feedback.

    Inputs:
        - hypothesis: Original hypothesis text
        - feedback: Dimensional scores and rationales
        - goal: Original goal

    Outputs:
        - refined_hypothesis: Improved version
        - changes: Summary of changes made
    """

    hypothesis = InputField(desc="Original hypothesis")
    feedback = InputField(desc="Dimensional scores and rationales")
    goal = InputField(desc="Goal text")

    refined_hypothesis = OutputField(desc="Improved hypothesis")
    changes = OutputField(desc="Summary of changes made")


class LATSProgram(dspy.Module):
    def __init__(self, cfg, agent):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.generator = Predict(TraceStep)
        self.value_estimator = Predict(ValueEstimator)
        self.reflector = Predict(ReflectionPrompt)
        self.sharpener = Predict(SharpeningPrompt)
        self.max_depth = cfg.get("max_depth", 3)

    def _estimate_value(self, state, trace):
        """Estimate value using LM-powered scorer"""
        result = self.value_estimator(state=state, trace=trace, goal=state)
        try:
            score = float(result.score)
        except:
            score = 0.5
        return score, result.rationale

    def forward(self, state, trace, depth=0):
        if depth >= self.max_depth:
            return trace, self._estimate_value(state, trace)[0]

        prediction = self.generator(state=state, trace=trace)
        if not prediction or not prediction.next_step:
            return trace, 0.0

        next_step = prediction.next_step.strip()
        new_state = self.agent._update_state(state, next_step)
        new_trace = trace + [next_step]

        child_trace, child_score = self.forward(new_state, new_trace, depth + 1)

        if child_score < self.cfg.get("threshold", 0.7):
            reflection = self.reflector(state=new_state, trace=child_trace, goal=state)
            sharpened = self.sharpener(
                hypothesis=next_step, feedback=reflection.rationale, goal=state
            )
            child_trace[-1] = sharpened.refined_hypothesis
            new_state = self.agent._update_state(state, child_trace[-1])
            score, _ = self._estimate_value(new_state, child_trace)
            return child_trace, score

        return child_trace, child_score


class SymbolicImpactAnalyzer:
    """
    Analyzes structural overlap and divergence between two graph representations (e.g., symbolic vs. LATS)
    and attributes score delta to divergent paths.
    """

    def __init__(self, score_lookup_fn):
        self.score_lookup_fn = (
            score_lookup_fn  # Function to get scores for a given node or trace
        )

    def analyze(self, graph1, graph2):
        matches, only_1, only_2 = compare_graphs(graph1, graph2)
        results = []

        for node in matches:
            score_1 = self.score_lookup_fn(node, source="graph1")
            score_2 = self.score_lookup_fn(node, source="graph2")
            results.append(
                {"node": node, "type": "converged", "delta": score_2 - score_1}
            )

        for node in only_1 + only_2:
            score = self.score_lookup_fn(node, source="graph1")
            results.append({"node": node, "type": "diverged", "score": score})

        return results


class LATSDSPyAgent(ScoringMixin, BaseAgent):
    """
    Enhanced LATS agent with:
    - Tree search (MCTS + UCT)
    - Multi-dimensional scoring
    - Proximity-based reuse
    - Reflection/refinement
    - Rule tuning
    - DSPy optimization
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.max_depth = cfg.get("max_depth", 5)
        self.branching_factor = cfg.get("branching_factor", 3)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)
        self.lambda_weight = cfg.get("lambda", 0.5)

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

        # Setup DSPy
        lm = dspy.LM(
            "ollama_chat/qwen3",
            api_base="http://localhost:11434",
            api_key="",
        )
        dspy.configure(lm=lm)

        # Initialize DSPy program
        self.lats_program = LATSProgram(cfg, self)

        # Symbolic impact analyzer
        self.impact_analyzer = SymbolicImpactAnalyzer(self._get_score)
        self.score_map = {}
        self.completed_nodes = 0
        self.total_estimated_nodes = 1  # Start with 1 to avoid division by zero
        self.dimensions = self.get_dimensions("lats_reflection")
        self.scorer = MRQScorer(
            self.cfg, memory=self.memory, logger=self.logger, dimensions=self.dimensions
        )
        # self.scorer.train_from_database(cfg=self.cfg)
        # self.scorer.save_models()
        self.scorer.load_models()

    async def run(self, context: dict) -> dict:
        """Main LATS search loop"""
        goal = context[GOAL]
        root_state = root_state = {
            "goal": goal["goal_text"],
            "goal_id": goal["id"],
            "current": goal["goal_text"],
            "trace": [],
        }

        # 1. Initialize root node
        root = self.create_node(state=root_state, trace=[], parent=None)

        # 2. Run MCTS simulations
        for sim_num in range(self.num_simulations):
            # Selection
            node = self.select(root)

            # Expansion
            if not self.is_terminal(node):
                node = await self.expand(node, context)

            # Simulation & Evaluation
            reward, trace_data = self.simulate_and_evaluate(node, context)

            # Backpropagation
            self.backpropagate(node, reward, trace_data)

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

        # 4. Create final hypothesis
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
        """Generate new children nodes from current node"""
        self._log_node(node, level="debug")

        # Build prompt with context
        merged = {
            **context,
            "state": node["state"]["current"],  # Only pass the current reasoning state
            "trace": node["trace"],
            "mode": "reason",
        }

        # 1. Get similar hypotheses
        proximity_context = await self._run_proximity(context)
        self.logger.log(
            "ProximityContext",
            {
                "most_similar": proximity_context.get("most_similar"),
                "all": proximity_context,
            },
        )
        merged["similar_hypotheses"] = proximity_context.get("most_similar", "")

        # 2. Generate completions with DSPy
        completions, steps = self.lats_program.forward(
            state=node["state"], trace=node["trace"], depth=0
        )

        # 3. Apply proximity-based refinement
        refined_completions = []
        for comp in completions:
            refined = self._apply_proximity_guidance(comp, proximity_context)
            refined_completions.append(refined)

        # 4. Score and build children
        children = []
        for comp in refined_completions:
            new_state = self._update_state(node["state"], comp)
            new_trace = node["trace"] + [comp]

            # Ensure scoring context includes mode
            scoring_context = {
                **context,
                "mode": "reason",  # Required for CoR templates
            }

            # Score using dimensional scorers
            scorable = Scorable(text=comp)

            score_result = self.score_item(
                scorable, scoring_context, metrics="lats_node", scorer=self.scorer
            )
            node_path = self.build_node_path(node)
            aggregated_result = score_result.aggregate()
            print(f"Scoring result for {node_path}: {aggregated_result}")
            self.score_map[node_path] = aggregated_result

            # Create child node with metadata
            child = self.create_node(state=new_state, trace=new_trace, parent=node)
            child["score"] = score_result.aggregate()
            child["dimension_scores"] = score_result.to_dict()
            child["action"] = comp

            children.append(child)
            self._log_node(child, level="debug")
            self.completed_nodes += 1
            self.total_estimated_nodes = max(
                self.total_estimated_nodes,
                self.completed_nodes + len(refined_completions),
            )  # or open_nodes
            self._log_progress()

        # Store children
        self.children[id(node)] = children

        # Log all children at once
        for child in children:
            self._log_node(child, level="info")  # ← Add here

        return children[0] if children else node

    def _log_progress(self):
        pct = (self.completed_nodes / self.total_estimated_nodes) * 100
        print(
            f"🔁 Progress: {self.completed_nodes}/{self.total_estimated_nodes} nodes completed ({pct:.2f}%)"
        )

    def build_node_path(self, node):
        path = []
        while node:
            path.append(
                str(node["id"])
            )  # or node.name, or whatever identifies the node
            node = node.get("parent")  # assumes each node has a 'parent' reference
        return " → ".join(reversed(path))

    def simulate_and_evaluate(self, node, context):
        """Simulate until terminal state and return final reward"""
        current = node
        while not self.is_terminal(current) and len(current["trace"]) < self.max_depth:
            # Build prompt
            merged = {
                **context,
                "state": current["state"],
                "trace": current["trace"],
                "mode": "simulate",
            }
            prompt = self.prompt_loader.load_prompt(self.cfg, merged)
            response = self.call_llm(prompt, context=merged)

            # Parse completions
            completions = self._parse_completions(response)
            if not completions:
                break

            action = completions[0]  # Take first completion
            new_state = self._update_state(current["state"], action)
            new_trace = current["trace"] + [action]

            # Create new node
            current = self.create_node(state=new_state, trace=new_trace, parent=current)

        # Evaluate final node
        reward, trace_data = self.evaluate(current, context)
        return reward, trace_data

    def evaluate(self, node, context):
        """Evaluate node using hybrid LM + self-consistency scoring"""
        if self.cfg.get("use_environment", False):
            obs = self.env.step(node["state"])
            return obs["reward"], {"trace": node["trace"], "environment": obs}

        # Fallback: dimensional scoring
        print(node)
        scorable = Scorable(text="\n".join(node["trace"]))
        score_result = self.score_item(
            scorable, context, metrics="lats_reflection", scorer=self.scorer
        )
        return score_result.aggregate() / 100, score_result

    def backpropagate(self, node, reward, trace_data=None):
        """Update node statistics up the tree"""
        while node:
            self._log_node(node, level="debug")
            node["visits"] += 1
            node["reward"] += reward

            # Store trace data for analysis
            if trace_data:
                node.setdefault("history", []).append(
                    {
                        "visits": node["visits"],
                        "reward": reward,
                        "trace_data": trace_data,
                    }
                )

            node = node["parent"]

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

    def _update_state(self, state_dict, action):
        """
        Updates structured state dictionary with new action
        """
        if isinstance(state_dict, dict):
            new_state = state_dict.copy()
            new_state["current"] = f"{state_dict['current']}\n{action}"
            # Ensure trace stays as list
            new_state["trace"] = state_dict.get("trace", []) + [action]
            return new_state
        # Fallback for string state
        return {
            "current": f"{state_dict}\n{action}",
            "trace": [action],  # Start with action as list
        }

    def _parse_completions(self, response: str) -> list:
        """Parse multiple thoughts/actions from response"""
        thought_pattern = r"([Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s*(.*?)(?=\n(?:[Tt]hought\s*\d+|[Aa]ction\s*\d+|[-•])\s|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        if not matches:
            return [response.strip()]

        completions = [match[-1].strip() for match in matches if match[-1].strip()]
        return completions[: self.branching_factor]

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

    def _apply_reflection_to_prompt(self, prompt, reflection):
        """Inject reflection into prompt for future steps"""
        if not reflection:
            return prompt

        reflection_prompt = self.prompt_loader.load_prompt(
            "reflection_injection", {"prompt": prompt, "reflection": reflection}
        )

        return self.call_llm(reflection_prompt, {})

    def _get_score(self, node, source="graph1"):
        """Get score for node in impact analysis"""
        resolved = self.resolve_node(node)
        if not resolved:
            return 0.0

        # Safely extract trace (always a list)
        trace = resolved.get("trace", [])
        if isinstance(trace, str):
            trace = trace.split("\n")  # Fallback if trace is string
        elif not isinstance(trace, list):
            trace = []

        if not len(trace):
            return 0.0

        # Safely extract state (ensure dict)
        state = resolved.get("state", {})  # ✅ Fallback to empty dict
        goal_text = state.get("goal", "Unknown goal")

        # Build hypothesis for scoring
        scorable = Scorable(text="\n".join(trace))

        # Score using dimensional scorers
        score_result = self.score_item(
            scorable,
            {"goal": {"goal_text": goal_text}},  # Always a dict
            metrics="lats_reflection",
            scorer=self.scorer,
        )

        return score_result.aggregate() / 100  # Normalize

    def resolve_node(self, node):
        """Converts node ID or string to full node dict"""
        if isinstance(node, dict):
            return node

        # Try match by ID
        matches = [n for n in self.nodes if str(n["id"]) == str(node)]
        if matches:
            return matches[0]

        # Last resort: node is a trace string
        if isinstance(node, str):
            return {"trace": node.split("\n")}

        # Default
        return {"trace": []}

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

    def _get_value(self, node):
        """Calculate value using hybrid LM + self-consistency"""
        lm_score = node.get("score", 0.5)
        sc_score = self._self_consistency(node)
        return self.lambda_weight * lm_score + (1 - self.lambda_weight) * sc_score

    def _self_consistency(self, node):
        """Calculate self-consistency score for node"""
        if not node["trace"]:
            return 0.0

        # Use LLM to evaluate consistency
        prompt = self.prompt_loader.load_prompt(
            "self_consistency", {"trace": node["trace"], "state": node["state"]}
        )
        response = self.call_llm(prompt, {})

        # Parse numerical score
        score_match = re.search(r"(\d+)", response)
        return int(score_match.group(1)) / 100 if score_match else 0.5

    def _get_dimension_score(self, trace):
        """Get dimensional scores for trace"""
        # Build hypothesis
        scorable = Scorable(text="\n".join(trace))

        # Score across dimensions
        score_result = self.score_item(
            scorable,
            {"goal": {"goal_text": trace["goal"]}},
            metrics="lats_reflection",
            scorer=self.scorer,
        )
        return score_result.aggregate() / 100  # Normalize

    def _train_on_traces(self, traces):
        """Train DSPy module on high-quality traces"""
        # Convert traces to examples
        examples = [
            Example(
                state=trace["state"],
                trace=trace["trace"],
                next_step=trace["last_action"],
            )
            for trace in traces
        ]

        # Use dimensional scores as weights
        weighted_examples = [
            example.with_score(self._get_dimension_score(example.trace))
            for example in examples
        ]

        # Compile with BootstrapFewShot
        tuner = BootstrapFewShot(metric=self._dimension_aware_metric)
        self.lats_program.generator = tuner.compile(
            student=Predict(TraceStep), trainset=weighted_examples
        )

    def _dimension_aware_metric(self, example, pred):
        """Use dimensional scores for training metric"""
        scores = self._get_dimension_scores(pred.trace)
        return scores.aggregate()

    def _get_dimension_scores(self, trace) -> ScoreBundle:
        """Get scores across all dimensions"""
        hyp = {"text": "\n".join(trace)}
        return self.score_item(hyp, {}, metrics="lats_node", scorer=self.scorer)

    def _generate_reflection(self, node):
        """Generate reflection for failed trajectory"""
        prompt = self.prompt_loader.load_prompt(
            "reflection",
            {
                "trace": node["trace"],
                "state": node["state"],
                "goal": node["state"],  # Use state as goal proxy
            },
        )
        response = self.call_llm(prompt, {})
        return response.strip()

    def _build_prompt(self, node):
        """Build prompt for node evaluation"""
        merged = {"state": node["state"], "trace": node["trace"], "mode": "evaluate"}
        return self.prompt_loader.load_prompt(self.cfg, merged)

    def _choose_action(self, response):
        """Choose best action from response"""
        completions = self._parse_completions(response)
        return completions[0] if completions else ""

    def _self_consistency_check(self, node):
        """Validate consistency of reasoning path"""
        prompt = self.prompt_loader.load_prompt(
            "self_consistency", {"trace": node["trace"], "state": node["state"]}
        )
        response = self.call_llm(prompt, {})

        # Parse consistency score
        score_match = re.search(r"(\d+)", response)
        return int(score_match.group(1)) / 100 if score_match else 0.5

    def _should_prune(self, node):
        """Determine if node should be pruned"""
        return node.get("score", 0) < self.cfg.get("prune_threshold", 0.4)

    def _get_node_path(self, node):
        """Get full path from root to node"""
        path = []
        while node:
            path.append(node)
            node = node["parent"]
        return path[::-1]  # Reverse to get root-first

    def _log_simulation(self, sim_num, node, reward):
        """Log simulation results for analysis"""
        self.logger.log(
            "LATSIteration",
            {
                "simulation": sim_num,
                "node_id": node["id"],
                "reward": reward,
                "trace": node["trace"][-3:] if node["trace"] else [],
                "depth": len(node["trace"]),
            },
        )

    def _get_best_score(self, root):
        """Get best score from a graph representation"""
        best_node_id = max(self.score_map, key=lambda k: self.score_map[k])
        best_score = self.score_map[best_node_id]
        return {best_node_id, best_score}

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

    def _format_node_summary(self, node_info):
        """Formats node info into a single-line summary"""
        return (
            f"ID:{node_info['id']} "
            f"V:{node_info['visits']} "
            f"R:{node_info['reward']} "
            f"S:{node_info['score']} "
            f"D:{node_info['depth']} "
            f"P→{node_info['parent_id']} "
            f"Childs:{node_info['child_count']} "
            f"State:'{node_info['state_preview'][:30]}...'"
            f"Dimensions:{self._format_dimension_scores(node_info['dimension_scores'])}"
        )

    def _format_dimension_scores(self, dimension_scores):
        return " ".join(
            f"{dim}={data['score']}" for dim, data in dimension_scores.items()
        )
