# File: stephanie/agents/task_generator_agent.py

import json
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.models.mrq_memory_entry import MRQMemoryEntryORM
from stephanie.models.mrq_preference_pair import MRQPreferencePairORM


class TaskGeneratorAgent(BaseAgent):
    """
    Generates synthetic problems for self-play loops.
    Uses LADDER-style recursive decomposition and prompt engineering.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.difficulty_levels = ["easy", "medium", "hard"]

    async def run(self, context: dict) -> dict:
        """Main loop: generate problem → solve → judge → log improvement"""
        goal = context.get("goal", {})
        task_type = goal.get("type", "math")  # e.g., math, logic, coding
        depth = context.get("decomposition_depth", 2)

        # Step 1: Generate problem
        problem = await self.generate_problem(task_type, depth)
        self.logger.log("ProblemGenerated", {"problem": problem[:100]})

        # Step 2: Solve it (via inner agent)
        solution = await self.solve_problem(problem, context)
        self.logger.log("SolutionGenerated", {"solution": solution[:100]})

        # Step 3: Judge quality
        scores = await self.judge_solution(problem, solution)
        self.logger.log("SolutionScored", {"scores": scores})

        # Step 4: Log to MRQ memory
        self.log_to_mrq(problem, solution, scores)

        # Update context with results
        context["generated_problem"] = problem
        context["generated_solution"] = solution
        context["scores"] = scores
        return context

    async def generate_problem(self, task_type: str, depth: int) -> str:
        """Generate problem using prompt templates and recursive decomposition"""
        if depth <= 0:
            return self._generate_base_problem(task_type)

        # Recursive decomposition
        subproblems = []
        for _ in range(random.randint(2, 3)):
            subproblem = await self.generate_problem(task_type, depth - 1)
            subproblems.append(subproblem)

        # Combine subproblems
        prompt = f"""
        Combine these subproblems into a composite {task_type} problem:
        {" ".join([f"Subproblem {i+1}: {p}" for i, p in enumerate(subproblems)])}
        Output only the final problem statement.
        """
        composite_problem = await self.llm(prompt)
        return composite_problem.strip()

    def _generate_base_problem(self, task_type: str) -> str:
        """Generate a base-level problem using templates"""
        templates = {
            "math": [
                "Solve for x: {eq}",
                "What is the value of ∫{expr} dx from {a} to {b}?",
                "Find the roots of {poly}"
            ],
            "logic": [
                "Given premises: {premises}, what conclusion follows?",
                "If A implies B and B implies C, does A imply C?"
            ],
            "coding": [
                "Write a Python function to {task}",
                "Debug this code: {code_snippet}"
            ]
        }

        template = random.choice(templates.get(task_type, templates["math"]))
        if "{eq}" in template:
            return template.format(
                eq=self._generate_math_expression(),
                expr=self._generate_math_expression(),
                a=random.randint(1, 10),
                b=random.randint(11, 20),
                poly=self._generate_polynomial(),
                task="reverse a linked list",
                code_snippet="def buggy_func(x): return x + '2'"
            )
        return template

    def _generate_math_expression(self) -> str:
        """Helper: generate random math expressions"""
        return random.choice([
            "sin(x)^2 + cos(x)^2",
            "e^(ix)",
            "lim_{x→0} (sin x)/x"
        ])

    def _generate_polynomial(self) -> str:
        """Helper: generate random polynomials"""
        return random.choice([
            "x^3 - 6x^2 + 11x - 6",
            "2x^2 + 3x + 1",
            "x^4 - 1"
        ])

    async def solve_problem(self, problem: str, context: dict) -> str:
        """Use inner agent to solve the problem"""
        solver_name = self.cfg.get("solver_agent", "ChainOfThoughtAgent")
        try:
            module_name, class_name = solver_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            solver_cls = getattr(module, class_name)
            solver = solver_cls(cfg=self.cfg, memory=self.memory, logger=self.logger)
            solution = await solver.run({"goal": problem})
            return solution.get("response", "")
        except Exception as e:
            self.logger.error("ProblemSolvingFailed", {"error": str(e)})
            return ""

    async def judge_solution(self, problem: str, solution: str) -> Dict[str, float]:
        """Use MR.Q and LLM judge to score solution"""
        # Use MR.Q for fast embedding alignment
        mrq_score = self._mrq_judge(problem, solution)

        # Use LLM judge for detailed rubric scoring
        llm_scores = await self._llm_judge(problem, solution)

        # Combine scores
        return {
            "mrq_similarity": mrq_score,
            **llm_scores
        }

    def _mrq_judge(self, problem: str, solution: str) -> float:
        """Use MR.Q to compute embedding similarity"""
        try:
            from stephanie.evaluator.mrq_self_evaluator import MRQSelfEvaluator
            evaluator = MRQSelfEvaluator(memory=self.memory, logger=self.logger)
            result = evaluator.score_single(problem, solution, context={})
            return result.get("overall", 0.0)
        except Exception as e:
            self.logger.error("MRQScoringFailed", {"error": str(e)})
            return 0.0

    async def _llm_judge(self, problem: str, solution: str) -> Dict[str, float]:
        """Use structured rubric-based LLM judge"""
        try:
            from stephanie.evaluator.pipeline_judge import PipelineJudgeAgent
            judge = PipelineJudgeAgent(cfg=self.cfg, memory=self.memory, logger=self.logger)
            _, scores = await judge.compare_outputs(problem, solution, solution)
            return scores
        except Exception as e:
            self.logger.error("LLMJudgingFailed", {"error": str(e)})
            return {}

    def log_to_mrq(self, problem: str, solution: str, scores: dict):
        """Log generated problem and solution to MRQ memory"""
        try:
            entry = MRQMemoryEntryORM(
                goal=problem,
                strategy="self_play",
                prompt=problem,
                response=solution,
                reward=scores.get("mrq_similarity", 0.0),
                metadata_=json.dumps({
                    "scores": scores,
                    "source": "task_generator"
                }),
                created_at=datetime.now(timezone.utc)
            )
            self.session.add(entry)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            self.logger.error("MRQLoggingFailed", {"error": str(e)})