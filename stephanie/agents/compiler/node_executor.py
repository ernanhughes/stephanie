# stephanie/agents/compiler/node_executor.py
from stephanie.agents.compiler.reasoning_trace import ReasoningNode
from stephanie.agents.pipeline.pipeline_runner import PipelineRunnerAgent


class NodeExecutor:
    def __init__(self, cfg, memory, logger, pipeline_runner: PipelineRunnerAgent, tree):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.pipeline_runner = pipeline_runner
        self.tree = tree

    async def execute(self, node: ReasoningNode, context: dict) -> dict:
        # Construct context for pipeline execution
        goal = context.get("goal")
        if node.goal:
            goal["goal_text"] = node.goal
        merged = {
            "goal": goal,
            "current_thought": node.thought,
            "previous_actions": self._get_history(node),
            "pipeline_stages": self._build_pipeline_for_node(node),
            "tag": f"node_{node.id}",
            **context,
        }

        try:
            result = await self.pipeline_runner.run(merged)
            return {
                "response": result.get("selected", {}).get("text", ""),
                "success": True,
                "raw_result": result,
            }
        except Exception as e:
            return {"response": str(e), "success": False, "raw_result": None}

    def _get_history(self, node: ReasoningNode) -> list[dict]:
        history = []
        current = node
        while current.parent_id:
            parent = self.tree.nodes.get(current.parent_id)
            if not parent:
                break
            history.append(
                {
                    "thought": parent.thought,
                    "action": parent.action,
                    "response": parent.response,
                    "score": parent.score,
                }
            )
            current = parent
        history.reverse()
        return history

    def _build_pipeline_for_node(self, node: ReasoningNode) -> list[dict]:
        """
        Define what pipeline stages to use for this node.
        You could:
        - Use a fixed list of agents (e.g., chain_of_thought -> scorer)
        - Or dynamically modify it per node
        """
        return [
            {
                "name": "chain_of_thought",
                "type": "stephanie.agents.reasoning.ChainOfThoughtGeneratorAgent",
                "config": {
                    "input_key": "current_thought",
                    "output_key": "hypotheses",
                    "mode": "single",
                },
            },
            {
                "name": "score",
                "type": "stephanie.agents.scoring.PipelineJudgeAgent",
                "config": {"input_key": "hypotheses", "output_key": "selected"},
            },
        ]
