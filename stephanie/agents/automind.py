from stephanie.agents.ats.solution_node import SolutionNode
from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.judge import JudgeAgent
from stephanie.components.coding_strategy import SelfAdaptiveCoder
from stephanie.components.search_policy import TreeSearchPolicy
from stephanie.components.solution_tree import SolutionTree
from stephanie.memory.embedding_store import EmbeddingStore
from stephanie.scoring.scoring_manager import ScoringManager


class AutoMindAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.embedding_store = EmbeddingStore(cfg.embedding)
        self.judge = JudgeAgent(cfg.judge) if cfg.judge else None
        self.scorer = ScoringManager(cfg.scoring)
        self.tree = SolutionTree()
        self.policy = TreeSearchPolicy(cfg.policy)
        self.coder = SelfAdaptiveCoder(cfg.coder)
        self.max_iters = cfg.get("max_iters", 25)

    def run(self, goal):
        self.logger.log({"goal": goal})
        self.tree.initialize(goal)

        for step in range(self.max_iters):
            parent_node, action = self.policy.select(self.tree)

            if action == "draft":
                plan = self._generate_plan(goal)
            elif action == "improve":
                plan = self._improve_plan(parent_node)
            elif action == "debug":
                plan = self._debug_plan(parent_node)
            else:
                continue

            code = self.coder.generate_code(plan)
            output, metric = self._execute(code)

            valid = self.judge.validate(plan, code, output, metric) if self.judge else True
            node = SolutionNode(plan, code, metric, output, valid)
            self.tree.add_node(node)

            self.logger.log({"step": step, "plan": plan, "code": code, "metric": metric, "valid": valid})

        best = self.tree.get_best()
        return best.code

    def _generate_plan(self, goal):
        context = self._retrieve_knowledge(goal)
        return self.prompt_loader.render("draft_plan", goal=goal, context=context)

    def _improve_plan(self, node):
        context = self._retrieve_knowledge(node.plan)
        return self.prompt_loader.render("improve_plan", plan=node.plan, output=node.output, context=context)

    def _debug_plan(self, node):
        return self.prompt_loader.render("debug_plan", plan=node.plan, output=node.output)

    def _retrieve_knowledge(self, query):
        return self.embedding_store.query(query, top_k=3)

    def _execute(self, code):
        try:
            # Placeholder: execute code in sandboxed environment
            metric, output = 0.0, "Execution output"
        except Exception as e:
            return str(e), None
        return output, metric
