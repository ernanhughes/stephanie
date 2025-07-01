# stephanie/agents/ats/agentic_tree_search.py

import random
import time
from typing import Dict, List, Optional, Tuple

from stephanie.agents.ats.solution_node import SolutionNode
from stephanie.agents.base_agent import BaseAgent


class AgenticTreeSearch:
    def __init__(
        self,
        agent: BaseAgent,
        max_iterations: int = 500,
        time_limit: int = 86400,  # 24 hours
        N_init: int = 5,
        H_debug: float = 0.8,
        H_greedy: float = 0.8,
    ):
        self.agent = agent
        self.tree: List[SolutionNode] = []
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.N_init = N_init
        self.H_debug = H_debug
        self.H_greedy = H_greedy
        self.iteration = 0
        self.start_time = time.time()
        self.best_node: Optional[SolutionNode] = None

    async def run(self, context: dict):
        task_description = context.get("goal", {}).get("goal_text")
        knowledge = context.get("knowledge", [])

        while not self._should_stop():
            action, parent_node = self.select_action()
            new_plan = await self.generate_plan(task_description, parent_node, action, knowledge)
            new_code = self.generate_code(new_plan)
            result = self.execute_code(new_code)

            verification = self.verify_output(result)
            new_node = SolutionNode(
                plan=new_plan,
                code=new_code,
                metric=verification["metric"],
                output=result.get("stdout"),
                summary=verification["summary"],
                parent_id=parent_node.id if parent_node else None,
                is_buggy=verification["is_bug"]
            )

            self.tree.append(new_node)
            self.update_best_node(new_node)
            self.iteration += 1

        # Final submission
        best_solution = self.get_best_solution()
        context["final_solution"] = best_solution.to_dict()
        return context

    def _should_stop(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True
        if time.time() - self.start_time > self.time_limit:
            return True
        return False

    def select_action(self) -> Tuple[str, Optional[SolutionNode]]:
        """
        Implements Algorithm 1 from AutoMind paper_score
        Returns (action_type, parent_node)
        """
        if len([n for n in self.tree if "draft" in n.plan]) < self.N_init:
            return "draft", None

        buggy_nodes = [n for n in self.tree if n.is_buggy]
        valid_nodes = [n for n in self.tree if not n.is_buggy and n.metric is not None]

        if random.random() < self.H_debug and buggy_nodes:
            return "debug", random.choice(buggy_nodes)

        if self.best_node and random.random() < self.H_greedy:
            return "improve", self.best_node

        if valid_nodes:
            return "improve", random.choice(valid_nodes)

        return "draft", None

    async def generate_plan(self, task_description, parent_node, action, knowledge):
        # Use prompt templates defined below
        pass

    def generate_code(self, plan):
        # Decide one-pass vs stepwise based on complexity score
        pass

    def execute_code(self, code):
        # Run in sandboxed environment
        pass

    def verify_output(self, output):
        # Check for bugs, overfitting, submission file
        pass

    def update_best_node(self, node: SolutionNode):
        if node.metric is not None:
            if self.best_node is None or node.metric > self.best_node.metric:
                self.best_node = node

    def get_best_solution(self) -> Optional[SolutionNode]:
        return self.best_node