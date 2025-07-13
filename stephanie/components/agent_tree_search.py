# stephanie/components/agent_tree_search.py
import time

from stephanie.agents.ats.solution_node import SolutionNode


class AgenticTreeSearch:
    def __init__(self, agent, max_iterations=500, time_limit=86400):
        self.agent = agent
        self.tree = []
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.iteration = 0

    async def run(self, context):
        start_time = time.time()
        while (
            self.iteration < self.max_iterations
            and (time.time() - start_time) < self.time_limit
        ):
            action, parent_node = self.select_action()
            new_plan = await self.generate_plan(parent_node, action)
            new_code = self.generate_code(new_plan)
            result = self.execute_code(new_code)

            # Verify output
            verification = self.verify_output(result)
            new_node = SolutionNode(
                plan=new_plan,
                code=new_code,
                metric=verification["metric"],
                output=result,
                summary=verification["summary"],
            )
            self.tree.append(new_node)
            self.iteration += 1

        # Return best solution
        return self.get_best_solution()

    def select_action(self):
        # Implement Algorithm 1 logic here
        pass

    async def generate_plan(self, parent_node, action):
        # Use prompts to generate plan
        pass

    def generate_code(self, plan):
        # Decide one-pass vs stepwise
        pass

    def execute_code(self, code):
        # Run in sandbox
        pass

    def verify_output(self, output):
        # Check for bugs, metrics, submission file
        pass

    def get_best_solution(self):
        # Return node with highest η
        pass
