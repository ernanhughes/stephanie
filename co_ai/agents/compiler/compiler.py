from co_ai.agents.compiler.reasoning_trace import ReasoningTree
from co_ai.agents.compiler.node_executor import NodeExecutor
from co_ai.agents.compiler.final_prompt_builder import FinalPromptBuilder
from co_ai.agents.compiler.step_selector import StepSelector
from co_ai.agents.compiler.symbol_mapper import SymbolMapper
from co_ai.agents.compiler.scorer import ReasoningNodeScorer
from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.pipeline.pipeline_runner import PipelineRunnerAgent
from co_ai.agents.mixins.scoring_mixin import ScoringMixin

class CompilerAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, full_cfg=None):
        super().__init__(cfg, memory, logger)
        self.tree = ReasoningTree()
        runner = PipelineRunnerAgent(cfg, memory=memory, logger=logger, full_cfg=full_cfg)
        self.executor = NodeExecutor(cfg, memory=memory, logger=logger, pipeline_runner=runner, tree=self.tree)
        self.scorer = ReasoningNodeScorer(cfg, memory=memory, logger=logger)
        self.mapper = SymbolMapper(cfg, memory=memory, logger=logger)
        self.selector = StepSelector()
        self.builder = FinalPromptBuilder()

        if self.logger:
            self.logger.log("CompilerAgentInit", {
                "message": "Initialized CompilerAgent",
                "config": cfg
            })

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text", "No goal text provided")
        root_id = self.tree.add_root(goal_text, "Start solving this problem.", "Generate initial plan")
        node = self.tree.nodes[root_id]

        if self.logger:
            self.logger.log("CompilerStart", {
                "goal_text": goal_text,
                "root_id": root_id
            })

        for iteration in range(5):  # Max iterations
            if self.logger:
                self.logger.log("CompilerIterationStart", {
                    "iteration": iteration,
                    "node_id": node.id,
                    "thought": node.thought
                })

            result = await self.executor.execute(node, context)
            node.response = result["response"]

            if self.logger:
                self.logger.log("ExecutionResult", {
                    "node_id": node.id,
                    "response": node.response,
                    "success": result.get("success"),
                    "raw_result": result.get("raw_result")
                })

            merged = {
                "thought": node.thought,
                **context
            }

            score = self.scorer.score(node, merged)
            node.score = score.aggregate()

            if self.logger:
                self.logger.log("ScoringComplete", {
                    "node_id": node.id,
                    "score_bundle": score.to_dict() if hasattr(score, "to_dict") else score
                })

            self.mapper.tag_node(node)

            next_steps = self.selector.select_next_steps(self.tree)

            if self.logger:
                self.logger.log("StepSelection", {
                    "node_id": node.id,
                    "next_step_count": len(next_steps)
                })

            for step in next_steps:
                child_id = self.tree.add_child(
                    parent_id=node.id,
                    thought=step["thought"],
                    action=step["action"],
                    response="",
                    score=0.0
                )
                if self.logger:
                    self.logger.log("ChildCreated", {
                        "parent_id": node.id,
                        "child_id": child_id,
                        "thought": step["thought"],
                        "action": step["action"]
                    })

                node = self.tree.nodes[child_id]

        best_path = self.tree.get_best_path()
        final_prompt = self.builder.build_prompt(best_path)

        if self.logger:
            self.logger.log("FinalPromptBuilt", {
                "final_prompt": final_prompt,
                "path_length": len(best_path)
            })

        context["final_prompt"] = final_prompt
        return context
