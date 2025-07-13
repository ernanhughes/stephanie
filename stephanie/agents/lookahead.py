# stephanie/agents/lookahead.py
import re

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE
from stephanie.models import LookaheadORM


class LookaheadAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict):
        goal = self.memory.goals.get_or_create(context.get(GOAL))

        # Build context for prompt template

        pipeline = context.get(PIPELINE, [])

        agent_registry = context.get("agent_registry", {})
        # Current agents and all available agents from the registry
        pipeline_info = {
            step: agent_registry.get(step, {"description": "No description."})
            for step in pipeline
        }
        print(f"Pipeline info: {pipeline_info}")

        all_agents_info = {name: data for name, data in agent_registry.items()}
        print(f"All agents info: {all_agents_info}")

        prompt_context = {
            "goal": goal.goal_text,
            "goal_type": goal.goal_type,
            "focus_area": goal.focus_area,
            "strategy": goal.strategy,
            "llm_suggested_strategy": goal.llm_suggested_strategy,
            PIPELINE: pipeline,
            "pipeline_info": {
                step: agent_registry.get(step, {"description": "No description"})
                for step in pipeline
            },
            "all_agents": agent_registry,
            **context,
        }

        prompt_template = self.prompt_loader.load_prompt(self.cfg, prompt_context)

        # Call LLM to generate anticipated issues and fallbacks
        response = self.call_llm(prompt_template, prompt_context).strip()

        # Store the reflection for traceability
        model_name = self.cfg.get("model").get("name")
        extracted = self.parse_response(response)
        context.update(extracted)
        pipeline = context.get(PIPELINE, [])
        lookahead_data = LookaheadORM(
            goal=goal.id,
            agent_name=self.name,
            model_name=model_name,
            input_pipeline=context.get(PIPELINE),
            suggested_pipeline=["generation", "verifier", "judge"],
            rationale="Input pipeline lacks verification step.",
            reflection="# Predicted Risks\n- Hallucination risk\n- No validation",
            backup_plans=[
                "Plan A: Add fact-checking",
                "Plan B: Use retrieval-augmented generation",
            ],
            metadata={"domain": "AI Safety"},
            run_id=context.get("run_id"),
        )
        self.memory.lookahead.insert(goal.id, lookahead_data)
        # Log the result
        self.logger.log(
            "LookaheadGenerated",
            {
                "goal": goal.goal_text,
                "lookahead": response[:250],  # short preview
            },
        )

        # Store in context
        context[self.output_key] = lookahead_data
        return context

    def parse_response(self, text: str) -> dict:
        import re

        suggested = re.search(r"# Suggested Pipeline\s*(.*?)\n#", text, re.DOTALL)
        rationale = re.search(r"# Rationale\s*(.*)", text, re.DOTALL)

        pipeline = suggested.group(1).strip().splitlines() if suggested else []
        pipeline = [line.strip("- ").strip() for line in pipeline if line.strip()]

        return {
            "suggested_pipeline": pipeline if pipeline else None,
            "rationale": rationale.group(1).strip() if rationale else None,
        }

    def extract_sections(self, text: str) -> dict:
        # Simple section splitting
        risks_match = re.search(r"# Predicted Risks\s*(.*?)(?:#|$)", text, re.DOTALL)
        backups_match = re.search(r"# Backup Plans\s*(.*)", text, re.DOTALL)

        return {
            "rationale": risks_match.group(1).strip() if risks_match else None,
            "backup_plans": [
                line.strip("- ").strip()
                for line in (
                    backups_match.group(1).strip().split("\n") if backups_match else []
                )
                if line.strip()
            ],
        }
