
class RuleTuner:
    """
    Provides utilities to modify or tune symbolic rules based on performance analytics.
    """

    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger

    def increase_priority(self, rule_id: int, amount: float = 0.1) -> float:
        """
        Increases the priority of the rule by a given amount. If no priority is set, defaults to 1.0.
        """
        rule = self.memory.symbolic_rules.get(rule_id)
        if not rule:
            self.logger.log("RuleNotFound", {"rule_id": rule_id})
            return None

        current_priority = rule.attributes.get("priority", 1.0)
        new_priority = round(float(current_priority) + amount, 4)
        rule.attributes["priority"] = new_priority

        self.memory.symbolic_rules.update(rule)
        self.logger.log("RulePriorityUpdated", {
            "rule_id": rule_id,
            "old_priority": current_priority,
            "new_priority": new_priority,
        })

        return new_priority

    def decrease_priority(self, rule_id: int, amount: float = 0.1) -> float:
        """
        Decreases the priority of the rule by a given amount (min 0.0).
        """
        rule = self.memory.symbolic_rules.get(rule_id)
        if not rule:
            self.logger.log("RuleNotFound", {"rule_id": rule_id})
            return None

        current_priority = rule.attributes.get("priority", 1.0)
        new_priority = max(0.0, round(float(current_priority) - amount, 4))
        rule.attributes["priority"] = new_priority

        self.memory.symbolic_rules.update(rule)
        self.logger.log("RulePriorityUpdated", {
            "rule_id": rule_id,
            "old_priority": current_priority,
            "new_priority": new_priority,
        })

        return new_priority
