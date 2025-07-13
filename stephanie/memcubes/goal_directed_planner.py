class GoalDirectedPlanner:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.plan = []
    
    def generate_plan(self, goal: str) -> List[Belief]:
        """Generate plan using belief graph"""
        self.plan = self._find_path_to_goal(goal)
        return self.plan
    
    def _find_path_to_goal(self, goal: str) -> List[Belief]:
        """Find reasoning path to goal"""
        goal_node = self.world_model.belief_graph.nodes[self.world_model._find_goal_node(goal)]
        return nx.shortest_path(self.world_model.belief_graph, source=self.world_model._find_root(), target=goal_node)
    
    def execute_plan(self, context: dict):
        """Execute plan step-by-step"""
        result = context.copy()
        for belief in self.plan:
            result = self._apply_step(result, belief)
            if self._check_success(result, belief):
                result["success"] = True
                break
        return result
    
    def _apply_step(self, context: dict, belief: Belief):
        """Apply belief to context"""
        # Use belief content to transform context
        context["input"] = belief.content + "\n\n" + context.get("input", "")
        return context
    
    def _check_success(self, context: dict, belief: Belief) -> bool:
        """Check if belief solves the goal"""
        # Use EBT to verify goal alignment
        energy = self.ebt.get_energy(belief.content, context.get("input", ""))
        return energy < 0.5