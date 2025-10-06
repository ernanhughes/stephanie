# stephanie/agents/agentic_tree_search.py
from __future__ import annotations

import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from stephanie.agents.base_agent import BaseAgent


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    has_submission_file: bool = False


class SolutionNode:
    def __init__(
        self,
        plan: str,
        code: Optional[str] = None,
        metric: Optional[float] = None,
        output: Optional[str] = None,
        summary: Optional[str] = None,
        parent_id: Optional[str] = None,
        is_buggy: bool = False,
        node_type: str = "draft",  # 'draft', 'improve', 'debug'
        timestamp: Optional[float] = None,
    ):
        # Use UUID for stable, unique ID (hash(self) changes per instance!)
        self.id: str = str(uuid.uuid4())
        self.plan = plan
        self.code = code
        self.metric = metric
        self.output = output
        self.summary = summary
        self.parent_id = parent_id
        self.is_buggy = is_buggy
        self.node_type = node_type
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plan": self.plan,
            "code": self.code,
            "metric": self.metric,
            "output": self.output,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "is_buggy": self.is_buggy,
            "node_type": self.node_type,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SolutionNode":
        # Remove 'id' since it's regenerated
        data = data.copy()
        data.pop("id", None)
        return cls(**data)


class PlanGenerator:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def draft_plan(self, task_description: str, knowledge: list) -> str:
        prompt = f"""
You are an expert machine learning engineer.
Create a detailed solution plan for this task:
{task_description}

Some relevant tricks from past solutions:
{" ".join(knowledge) if knowledge else "None"}

Output only the plan as natural language text. Do not include code.
"""
        response = await self.agent.llm(prompt)
        return response.strip()

    async def improve_plan(
        self, previous_plan: str, feedback: str, knowledge: list
    ) -> str:
        prompt = f"""
Improve this ML solution plan:
{previous_plan}

Feedback: {feedback}
Additional knowledge: {" ".join(knowledge) if knowledge else "None"}

Output only the improved plan. Do not include code.
"""
        response = await self.agent.llm(prompt)
        return response.strip()

    async def debug_plan(self, previous_plan: str, error_log: str) -> str:
        prompt = f"""
Fix this buggy ML solution plan:
{previous_plan}

Error log: {error_log}
Output only the corrected plan. Do not include code.
"""
        response = await self.agent.llm(prompt)
        return response.strip()


class OutputVerifier:
    def verify(self, output: str, has_submission_file: bool) -> Dict[str, Any]:
        is_bug = "Error" in output or "Exception" in output or "Traceback" in output
        is_overfitting = "val_loss increasing" in output.lower()
        metric = self.extract_metric(output)
        summary = self.summarize(output)

        return {
            "is_bug": is_bug,
            "is_overfitting": is_overfitting,
            "has_csv_submission": has_submission_file,
            "metric": metric,
            "summary": summary,
        }

    def extract_metric(self, output: str) -> Optional[float]:
        # Try common patterns: "val_accuracy: 0.85", "score=0.92", etc.
        patterns = [
            r"val[_\s]?accuracy[:=]\s*([0-9]*\.?[0-9]+)",
            r"accuracy[:=]\s*([0-9]*\.?[0-9]+)",
            r"score[:=]\s*([0-9]*\.?[0-9]+)",
            r"metric[:=]\s*([0-9]*\.?[0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None  # Return None if no metric found

    def summarize(self, output: str) -> str:
        lines = output.strip().split("\n")[-5:]  # Last 5 lines
        return " ".join(lines) if lines else "No output."


class CodeExecutor:
    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def score_complexity(self, task_description: str, plan: str) -> float:
        prompt = f"""
Rate the complexity of this task and plan on a scale of 1–5 (1 = simple, 5 = very complex).
Task: {task_description}
Plan: {plan}

Respond with a single number between 1 and 5.
"""
        response = await self.agent.llm(prompt)
        try:
            score = float(response.strip())
            return max(1.0, min(5.0, score))
        except (ValueError, TypeError):
            return 3.0

    async def one_pass_codegen(self, plan: str) -> str:
        prompt = f"""
Generate complete, runnable Python code for the following machine learning plan.
Do not explain. Only output code.

Plan:
{plan}
"""
        response = await self.agent.llm(prompt)
        # Optional: post-process to remove markdown
        if response.startswith("```python"):
            response = response[9:]
        if response.endswith("```"):
            response = response[:-3]
        return response.strip()

    async def stepwise_codegen(self, plan: str) -> str:
        # For now, fallback to one-pass; stepwise is complex and LLMs rarely return structured steps reliably
        return await self.one_pass_codegen(plan)


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

        self.plan_generator = PlanGenerator(agent)
        self.code_executor = CodeExecutor(agent)
        self.verifier = OutputVerifier()

    async def run(self, context: dict) -> dict:
        task_description = context.get("goal", {}).get("goal_text", "")
        knowledge = context.get("knowledge", [])

        if not task_description:
            raise ValueError("Context must contain 'goal.goal_text'")

        # Initial drafts
        for _ in range(self.N_init):
            if self._should_stop():
                break
            plan = await self.plan_generator.draft_plan(task_description, knowledge)
            node = await self._process_plan(plan, None, "draft", task_description, knowledge)
            self.tree.append(node)
            self.update_best_node(node)
            self.iteration += 1

        # Main loop
        while not self._should_stop():
            action, parent_node = self.select_action()
            if action == "draft":
                plan = await self.plan_generator.draft_plan(task_description, knowledge)
                node = await self._process_plan(plan, None, "draft", task_description, knowledge)
            elif action == "improve":
                assert parent_node is not None
                feedback = parent_node.summary or "No specific feedback."
                plan = await self.plan_generator.improve_plan(parent_node.plan, feedback, knowledge)
                node = await self._process_plan(plan, parent_node, "improve", task_description, knowledge)
            elif action == "debug":
                assert parent_node is not None
                error_log = parent_node.output or "Unknown error."
                plan = await self.plan_generator.debug_plan(parent_node.plan, error_log)
                node = await self._process_plan(plan, parent_node, "debug", task_description, knowledge)
            else:
                raise ValueError(f"Unknown action: {action}")

            self.tree.append(node)
            self.update_best_node(node)
            self.iteration += 1

        # Final result
        best_solution = self.get_best_solution()
        if best_solution:
            context["final_solution"] = best_solution.to_dict()
        else:
            context["final_solution"] = None
        return context

    async def _process_plan(
        self,
        plan: str,
        parent_node: Optional[SolutionNode],
        node_type: str,
        task_description: str,
        knowledge: list,
    ) -> SolutionNode:
        # Generate code
        complexity = await self.code_executor.score_complexity(task_description, plan)
        if complexity > 3.5:
            code = await self.code_executor.stepwise_codegen(plan)
        else:
            code = await self.code_executor.one_pass_codegen(plan)

        # Execute code (⚠️ WARNING: This must be sandboxed in production!)
        result = await self.execute_code(code)

        # Verify output
        verification = self.verifier.verify(result.stdout, result.has_submission_file)

        return SolutionNode(
            plan=plan,
            code=code,
            metric=verification["metric"],
            output=result.stdout,
            summary=verification["summary"],
            parent_id=parent_node.id if parent_node else None,
            is_buggy=verification["is_bug"],
            node_type=node_type,
        )

    def _should_stop(self) -> bool:
        if self.iteration >= self.max_iterations:
            return True
        if time.time() - self.start_time > self.time_limit:
            return True
        return False

    def select_action(self) -> Tuple[str, Optional[SolutionNode]]:
        draft_count = sum(1 for n in self.tree if n.node_type == "draft")
        if draft_count < self.N_init:
            return "draft", None

        buggy_nodes = [n for n in self.tree if n.is_buggy]
        valid_nodes = [n for n in self.tree if not n.is_buggy and n.metric is not None]

        if buggy_nodes and random.random() < self.H_debug:
            return "debug", random.choice(buggy_nodes)

        if self.best_node and random.random() < self.H_greedy:
            return "improve", self.best_node

        if valid_nodes:
            return "improve", random.choice(valid_nodes)

        return "draft", None

    async def execute_code(self, code: str) -> ExecutionResult:
        """
        ⚠️ SECURITY WARNING:
        Executing arbitrary LLM-generated code is EXTREMELY DANGEROUS.
        In production, this must run in a sandboxed environment (e.g., Docker, gVisor, or restricted subprocess).
        For now, this is a placeholder that simulates execution.
        """
        # TODO: Replace with real sandboxed execution
        import os
        import subprocess
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                # ⚠️ NEVER do this in production without sandboxing!
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60s timeout
                    cwd=tempfile.gettempdir()
                )
                has_csv = any(fname.endswith('.csv') for fname in os.listdir(tempfile.gettempdir()))
                os.unlink(f.name)
                return ExecutionResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    has_submission_file=has_csv
                )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=1,
                has_submission_file=False
            )

    def update_best_node(self, node: SolutionNode):
        if node.metric is not None:
            if self.best_node is None or node.metric > self.best_node.metric:
                self.best_node = node

    def get_best_solution(self) -> Optional[SolutionNode]:
        return self.best_node