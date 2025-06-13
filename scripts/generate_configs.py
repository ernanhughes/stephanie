import os

CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)

configs = {
    "config.yaml": """
defaults:
  - db: postgres
  - logging/json_logger
  - agents/generation
  - agents/reflection
  - agents/ranking
  - agents/evolution
  - agents/meta_review
  - agents/proximity
  - agents/supervisor
  - prompt_refiner/default
""",
    "db/postgres.yaml": """
db:
  name: co
  user: co
  password: co
  host: localhost
  port: 5432
""",
    "prompt_refiner/default.yaml": """
prompt_refiner:
  enabled: true
  model:
    name: openai/gpt-4o
    api_key: ${envs.OPENAI_API_KEY}
  max_bootstrapped_demos: 3
  max_labeled_demos: 5
""",
    "prompt_refiner/disabled.yaml": """
prompt_refiner:
  enabled: false
""",
    "agents/generation.yaml": """
defaults:
  - prompt_refiner: default

agent:
  name: generation
  temperature: 0.7
  max_tokens: 512
""",
    "agents/reflection.yaml": """
defaults:
  - prompt_refiner: disabled

agent:
  name: reflection
  review_type: full
""",
    "agents/ranking.yaml": """
defaults:
  - prompt_refiner: disabled

agent:
  name: ranking
  tournament_type: elo
  max_matches_per_round: 10
""",
    "agents/evolution.yaml": """
defaults:
  - prompt_refiner: default

agent:
  name: evolution
  strategy: grafting
  use_grafting: true
""",
    "agents/meta_review.yaml": """
defaults:
  - prompt_refiner: disabled

agent:
  name: meta_review
  summary_length: long
""",
    "agents/proximity.yaml": """
agent:
  name: proximity
  similarity_threshold: 0.85
  clustering_enabled: true
""",
    "agents/supervisor.yaml": """
agent:
  name: supervisor
  iterations: 3
  max_parallel_agents: 3
  use_async: true
""",
    "logging/json_logger.yaml": """
logger:
  type: JSONLogger
  log_path: logs/pipeline_log.jsonl
"""
}

for filename, content in configs.items():
    path = os.path.join(CONFIG_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content.strip())
    print(f"Created {path}")