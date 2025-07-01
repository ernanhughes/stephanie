from stephanie.agents.base_agent import BaseAgent
from stephanie.models import SymbolicRuleORM
from stephanie.rules.rule_options_config import RuleOptionsConfig
from stephanie.rules.rule_tuner import RuleTuner
from stephanie.rules.symbolic_rule_applier import SymbolicRuleApplier


class RuleMutationAgent(BaseAgent):
    """
    Suggests and applies rule mutations using LLM guidance and a configurable set of allowable rule changes.
    Ensures all mutations are valid, novel, and tracked.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.target_agent = cfg["target_agent"]
        self.rule_mutation_prompt = cfg["rule_mutation_prompt"]
        self.template_path = cfg["template_path"]
        self.options_config = RuleOptionsConfig.from_yaml(cfg["options_file"])
        self.rule_tuner = RuleTuner(memory, logger)

    async def run(self, context: dict) -> dict:
        # Load relevant symbolic rules based on goal and agent
        applicable_rules = [
            r
            for r in self.rule_applier.rules
            if r.agent_name == self.target_agent
        ]

        if not applicable_rules:
            self.logger.log("NoRulesToMutate", {"agent": context.get("agent_name")})
            context["status"] = "no_rules_found"
            return context

        mutated_rules = []

        for rule in applicable_rules:
            target = rule.agent_name
            current_attributes = rule.attributes or {}

            available_options = self.options_config.get_options_for(target)
            if not available_options:
                self.logger.log("NoAvailableMutations", {"target": target})
                continue

            recent_perf = self.memory.rule_effects.get_recent_performance(rule.id)

            merged = {
                "current_attributes":current_attributes,
                "available_options":available_options,
                "recent_performance":recent_perf,
                **context,
            }

            mutation_prompt = self.prompt_loader.from_file(self.rule_mutation_prompt, self.cfg, merged)
            response = self.call_llm(mutation_prompt, context)
            parsed = RuleTuner.parse_mutation_response(response)

            if not parsed["attribute"] or not parsed["new_value"]:
                self.logger.log(
                    "MutationParseError", {"rule_id": rule.id, "response": response}
                )
                continue

            attr = parsed["attribute"]
            new_val = parsed["new_value"]

            # Validate the mutation is legal
            if not self.options_config.is_valid_change(target, attr, new_val):
                self.logger.log(
                    "InvalidRuleMutation",
                    {
                        "rule_id": rule.id,
                        "attribute": attr,
                        "value": new_val,
                    },
                )
                continue

            # Deduplicate
            if self.memory.symbolic_rules.exists_similar(rule, attr, new_val):
                self.logger.log(
                    "RuleMutationDuplicateSkipped",
                    {
                        "rule_id": rule.id,
                        "attribute": attr,
                        "value": new_val,
                    },
                )
                continue

            # Construct and save new rule
            mutated_attrs = dict(current_attributes)
            mutated_attrs[attr] = new_val

            new_rule = SymbolicRuleORM(
                target="agent",
                agent_name=rule.agent_name,
                goal_type=rule.goal_type,
                goal_category=rule.goal_category,
                difficulty=rule.difficulty,
                attributes=mutated_attrs,
                source="mutation",
            )
            self.memory.symbolic_rules.insert(new_rule)
            self.logger.log(
                "RuleMutationApplied",
                {
                    "original_rule_id": rule.id,
                    "new_rule": new_rule.to_dict(),
                },
            )
            mutated_rules.append(new_rule)

        context["mutated_rules"] = [r.to_dict() for r in mutated_rules]
        context["total"] = len(mutated_rules)

        return context
