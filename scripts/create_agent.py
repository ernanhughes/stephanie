import os
from pathlib import Path

TEMPLATE_AGENT_CODE = '''from co_ai.base_agent import BaseAgent

class {{ agent_name | capitalize }}(BaseAgent):
    def run(self, goal, **kwargs):
        # Implement agent logic here
        return {"{{ agent_name }}": "output from {{ agent_name }}"}
'''

TEMPLATE_CONFIG = '''{{ agent_name }}:
  name: {{ agent_name }}
  enabled: true
  save_prompt: true
  save_context: true
  skip_if_completed: false
  model:
    name: ollama/qwen3
    api_base: http://localhost:11434
    api_key: null
  input_keys: ["goal", "hypotheses"]
  output_key: {{ agent_name }}
  prompt_mode: file
  prompt_file: {{ agent_name }}.txt
'''

TEMPLATE_PROMPT = '''You are a helpful assistant

Based upon this goal:
Goal: {{ '{{' }} goal.goal_text {{ '}}' }}

{% if preferences %}
And these preferences:
{% for p in preferences %}
- {{ '{{' }} p {{ '}}' }}
{% endfor %}
{% endif %}

{% if instructions %}
Additional instructions: 
{% for i in instructions %}
- {{ '{{' }} i {{ '}}' }}
{% endfor %}
{% endif %}
'''

def create_agent_files(agent_name):
    agent_file = Path(f"co_ai/agents/{agent_name}.py")
    config_file = Path(f"configs/agents/{agent_name}.yaml")
    prompt_dir = Path(f"prompts/{agent_name}")
    prompt_file = prompt_dir / f"{agent_name}.txt"

    # Ensure directories exist
    agent_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Fill templates
    agent_code = TEMPLATE_AGENT_CODE.replace("{{ agent_name }}", agent_name)
    config_code = TEMPLATE_CONFIG.replace("{{ agent_name }}", agent_name)
    prompt_code = TEMPLATE_PROMPT

    # Write files
    agent_file.write_text(agent_code)
    config_file.write_text(config_code)
    prompt_file.write_text(prompt_code)

    print(f"✅ Created agent: {agent_name}")
    print(f"- {agent_file}")
    print(f"- {config_file}")
    print(f"- {prompt_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python create_agent.py <agent_name>")
    else:
        create_agent_files(sys.argv[1])
