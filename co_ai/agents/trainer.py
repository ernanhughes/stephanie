from co_ai.base_agent import BaseAgent

class {{ agent_name | capitalize }}(BaseAgent):
    def run(self, goal, **kwargs):
        # Implement agent logic here
        return {"trainer": "output from trainer"}
