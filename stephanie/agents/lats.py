# stephanie/agents/lats.py
import math
import re
import uuid

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.constants import GOAL
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType
from stephanie.utils.timing import time_function


class LATSNode:
    """Unified node structure for MCTS"""
    def __init__(self, state, trace, parent=None):
        self.id = str(uuid.uuid4())
        self.state = state
        self.trace = trace
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.score = None
        self.dimension_scores = {}
        self.is_terminal = False

    def is_leaf(self):
        return len(self.children) == 0


class LATSAgent(ScoringMixin, BaseAgent): 
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.root = None
        self.nodes = []
        self.max_depth = cfg.get("max_depth", 5)
        self.branching_factor = cfg.get("branching_factor", 3)
        self.ucb_weight = cfg.get("ucb_weight", 1.41)
        self.num_simulations = cfg.get("num_simulations", 50)
        self.prune_threshold = cfg.get("prune_threshold", 0.4)
        self.children = {}

        self.max_steps = self.cfg.get("max_steps", 10)
        self.branching_factor = self.cfg.get("branching_factor", 3)

    async def run(self, context: dict):
        """Main MCTS loop"""
        # Initialize root node
        goal = context["goal"]
        self.root = self.create_node(goal["goal_text"], [])
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self._select(self.root)
            if not self._is_terminal(node):
                self.expand(node, context)
            reward = self._simulate(node, context)
            self._backpropagate(node, reward)
        
        # Return best path
        best_child = self._best_uct(self.root)
        final_score, best_node, best_trace = self._get_best_score(self.root)

        context["best_trace"] = best_trace
        context["best_score"] = final_score

        # Log final result
        self.logger.log("FinalResult", {
            "trace": best_trace,
            "score": final_score,
            "dimension_scores": best_node.dimension_scores
        })

        # Get timing logs
        timing_logs = self.logger.get_logs_by_type("FunctionTiming")
        for log in timing_logs:
            print(f"{log['timestamp']} - {log['data']['function']}: {log['data']['duration_ms']}ms")
        return {
            "trace": best_child.trace,
            "score": best_child.score,
            "dimension_scores": best_child.dimension_scores
        }

    def expand(self, node, context: dict):
        completions = self._generate_completions(node, context)
        if not completions:
            self.logger.log("NoCompletions", {"context": context})
            return node

        children = []
        for i, comp in enumerate(completions):
            new_state = self._update_state(node.state, comp["action"])
            new_trace = node.trace + [comp["step"]]
            
            hyp = {"text": comp["action"], "goal_id": context[GOAL]["id"]}

            print(f"Node State:=====\n{node.state}\n===")

            if isinstance(node.state, dict):
                goal_text = node.state.get("goal", "Unknown goal")
            else:
                goal_text = str(node.state)

            score_result = self._score_hypothesis(hyp, {"goal": {"goal_text": goal_text}}, metrics="lats_node")
            
            child = self.create_node(new_state, new_trace, parent=node)
            child.score = score_result.aggregate()
            child.dimension_scores = score_result.to_dict()
            child.action = comp["action"]
            children.append(child)
        
        self.children[id(node)] = children
        return children[0] if children else node

    async def _run_proximity(self, context:dict):
        from stephanie.agents.proximity import ProximityAgent
        proximity_agent = ProximityAgent(
            self.cfg.get("proximity", {}), memory=self.memory, logger=self.logger
        )
        return await proximity_agent.run(context)

    # Set up LATSComponent
    @time_function(logger=None)
    def _generate_completions(self, node: LATSNode, context: dict):
        merged = {
            **context,
            "state": node.state["current"] if isinstance(node.state, dict) else node.state,
            "trace": node.trace,
            "mode": "reason",
            "branching_factor": self.branching_factor
        }
        
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        response = self.call_llm(prompt, context=merged)
        
        # First try structured parsing
        completions = self._parse_thought_sections(response)
        
        # Fallback to basic parsing
        if not completions:
            completions = self._fallback_parsing(response)
        
        return completions

    def _parse_thought_sections(self, response: str):
        """
        Extract structured thoughts from LLM response
        Format: ### Thought N\n**Rationale**: ...\n**Action**: ...
        """
        thought_pattern = r"###\s*Thought\s*\d+[\s\S]*?\n\*\*Rationale\*\*:\s*(.*?)(?:\n\*\*Action\*\*:\s*(.*?))?(\n###|\Z)"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)
        
        completions = []
        for i, (rationale, action, _) in enumerate(matches):
            completions.append({
                "step": f"Thought {i+1}: {rationale[:50]}...",
                "rationale": rationale.strip(),
                "action": action.strip()
            })
        
        return completions[:self.branching_factor]

    def _fallback_parsing(self, response: str):
        """Fallback parser for raw text responses"""
        # Match Thought N: patterns
        thought_pattern = r"[Tt]hought\s*\d+:\s*(.*?)(?=\n(?:[Tt]hought\s*\d+:\s|\Z))"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)

        # Clean and extract
        completions = [{"step": match.strip()} for match in matches if match.strip()]

        # If no thoughts found, use entire response as one hypothesis
        if not completions and response.strip():
            completions = [{"step": response.strip()}]

        return completions[:self.branching_factor]

    @time_function(logger=None)
    def _fallback_parsing(self, response: str):
        """Fallback parser for raw text responses"""
        # Match Thought N: patterns
        thought_pattern = r"[Tt]hought\s*\d+:\s*(.*?)(?=\n(?:[Tt]hought\s*\d+:\s|\Z))"
        matches = re.findall(thought_pattern, response.strip(), re.DOTALL)
        
        completions = []
        for i, match in enumerate(matches[:self.branching_factor]):
            step = match.strip()
            completions.append({
                "step": f"Thought {i+1}: {step}",
                "rationale": "Fallback step due to parsing failure",
                "action": step  # Use step as action
            })
        
        # If no matches found, wrap the whole response
        if not completions and response.strip():
            completions.append({
                "step": "Fallback reasoning step",
                "rationale": "No structured thoughts found",
                "action": response.strip()
            })
        
        return completions

    @time_function(logger=None)
    def _update_state(self, state, action: str):
        """Update state with new action"""
        if isinstance(state, dict):
            new_state = state.copy()
            new_state["current"] = f"{state['current']}\n{action}"
            new_state["trace"] = state.get("trace", []) + [action]
            return new_state
        
        # Fallback: string-based state → wrap into dict
        return {
            "goal": state,
            "current": f"{state}\n{action}",
            "trace": [action]
        }

    @time_function(logger=None)
    def _score_hypothesis(self, hypothesis: dict, context: dict, metrics: str = "lats_node"):
        """Use dimensional scoring system"""
        scorable = ScorableFactory.from_dict(hypothesis, TargetType.HYPOTHESIS)
        return super().score_hypothesis(scorable, context, metrics)

    @time_function(logger=None)
    def _simulate(self, node: LATSNode, context: dict):
        """Simulate until terminal state"""
        current = node
        
        while not self._is_terminal(current) and len(current.trace) < self.max_depth:
            prompt = self._build_prompt(current, context, mode="simulate")
            response = self.call_llm(prompt, context)
            completions = self._fallback_parsing(response)
            
            if completions:
                action = completions[0]
                new_state = self._update_state(current.state, action)
                current = self.create_node(new_state, current.trace + [action], parent=current)
        
        # Evaluate final node
        return self._get_value(current)

    @time_function(logger=None)
    def _get_value(self, node: LATSNode):
        """Hybrid value function using LM + self-consistency"""
        if self.cfg.get("use_environment", False):
            obs = self.env.step(node.state)
            return obs['reward']
        
        # Safely extract goal from state
        if isinstance(node.state, dict):
            goal_text = node.state.get("goal", "Unknown goal")
        else:
            goal_text = str(node.state)
        
        score_result = self._score_hypothesis(
            {"text": "\n".join(str(node.trace))},
            {"goal": {"goal_text": goal_text}},
            metrics="lats_reflection"
        )
        return score_result.aggregate() / 100  # Normalize

    @time_function(logger=None)
    def _build_prompt(self, node, context:dict, mode="reason"):
        """Build prompt from node state"""
        if isinstance(node.state, dict):
            state = node.state["current"]
        else:
            state = str(node.state)
        
        merged = {
            **context,
            "state": state,
            "trace": node.trace,
            "mode": mode,
        }   
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        print(f"Prompt for {mode}: {prompt[:100]}...")  # Debugging output
        return prompt
    
    def _extract_goal_text(self, state):
        """
        Safely extract goal text from state
        Handles both string and dict-based state
        """
        if isinstance(state, dict):
            return state.get("goal", state.get("current", "Unknown goal"))
        return str(state)
    
    def create_node(self, state, trace, parent=None):
        """Create a new node with proper structure"""
        node = LATSNode(state, trace, parent)
        self.nodes.append(node)
        return node

    def _is_terminal(self, node: LATSNode) -> bool:
        """Check if node is terminal state"""
        return node.is_terminal or len(node.trace) >= self.max_depth

    def _backpropagate(self, node: LATSNode, reward: float):
        """Update node statistics up the tree"""
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def _uct_score(self, parent_visits: int, child: LATSNode):
        """Calculate UCT score for node selection"""
        if child.visits == 0:
            return float('inf')
        return (child.reward / child.visits) + \
               self.ucb_weight * math.sqrt(math.log(parent_visits) / child.visits)

    def _best_uct(self, node: LATSNode):
        """Select best child using UCT formula"""
        return max(node.children, key=lambda c: self._uct_score(node.visits, c))

    def _select(self, node: LATSNode):
        """Select node for expansion using UCT"""
        while not self._is_terminal(node) and not node.is_leaf():
            node = self._best_uct(node)
        return node

    def _expand(self, node: LATSNode, context: dict):
        """Generate children nodes using agent-specific expansion"""
        completions = self._generate_completions(node, context)
        
        for comp in completions:
            new_state = self._update_state(node.state, comp)
            new_trace = node.trace + [comp]
            

                        # Add mode to context
            score_context = {
                "goal": {"goal_text": context["goal"]["goal_text"]}
            }

            # Generate dimension scores
            score_result = self._score_hypothesis(
                {"text": comp},
                score_context,
                metrics="lats_node"
            )
            
            child = self.create_node(new_state, new_trace, parent=node)
            child.score = score_result.aggregate()
            child.dimension_scores = score_result.
            child.reward = child.score  # Initialize reward
            node.children.append(child)

