# stephanie/rules/rule_tuner.py


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
        rule = self.memory.symbolic_rules.get_by_id(rule_id)
        if not rule:
            self.logger.log("RuleNotFound", {"rule_id": rule_id})
            return None

        current_priority = rule.attributes.get("priority", 1.0)
        new_priority = round(float(current_priority) + amount, 4)
        rule.attributes["priority"] = new_priority

        self.memory.symbolic_rules.update(rule)
        self.logger.log(
            "RulePriorityUpdated",
            {
                "rule_id": rule_id,
                "old_priority": current_priority,
                "new_priority": new_priority,
            },
        )

        return new_priority

    def decrease_priority(self, rule_id: int, amount: float = 0.1) -> float | None:
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
        self.logger.log(
            "RulePriorityUpdated",
            {
                "rule_id": rule_id,
                "old_priority": current_priority,
                "new_priority": new_priority,
            },
        )

        return new_priority

    @staticmethod
    def build_rule_mutation_prompt(
        template_path: str,
        target: str,
        current_attributes: dict,
        available_options: dict,
        recent_performance: str = None,
    ) -> str:
        """
        Generates a rule mutation prompt from a Jinja template by injecting
        the current rule configuration, available tuning options, and optional
        performance context. Used to guide LLMs in suggesting rule improvements.
        """
        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader("/".join(template_path.split("/")[:-1]))
        )
        template = env.get_template(template_path.split("/")[-1])

        prompt = template.render(
            target=target,
            current_attributes=current_attributes,
            available_options=available_options,
            recent_performance=recent_performance,
        )
        return prompt

    def parse_mutation_response(response: str):
        import re

        rationale_match = re.search(r"Rationale:\s*(.+?)\n", response, re.DOTALL)
        attr_match = re.search(r"The attribute you want to change:\s*(.+)", response)
        value_match = re.search(r"The value you want to change to:\s*(.+)", response)

        return {
            "rationale": rationale_match.group(1).strip() if rationale_match else None,
            "attribute": attr_match.group(1).strip() if attr_match else None,
            "new_value": value_match.group(1).strip() if value_match else None,
        }
