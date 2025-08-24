import copy
import json
import yaml


SENSITIVE_KEYS = {"password", "db_password", "secret", "api_key", "token"}

def sanitize_config(config: dict) -> dict:
    """
    Recursively walks the config dict and replaces sensitive values with '***'.
    """
    safe = copy.deepcopy(config)
    for k, v in safe.items():
        if isinstance(v, dict):
            safe[k] = sanitize_config(v)
        elif isinstance(v, list):
            safe[k] = [sanitize_config(i) if isinstance(i, dict) else i for i in v]
        else:
            if k.lower() in SENSITIVE_KEYS:
                safe[k] = "***"
    return safe

def get_run_config(run) -> dict:
    """
    Extracts and sanitizes the run configuration from the pipeline run dict.
    """
    config_yaml = None
    if run.run_config:
        try:
            if isinstance(run.run_config, str):
                config_dict = json.loads(run.run_config)
            else:
                config_dict = run.run_config
            safe_config = sanitize_config(config_dict)
            config_yaml = yaml.dump(safe_config, sort_keys=False, indent=2)
        except Exception as e:
            config_yaml = f"# Error converting config to YAML: {e}"

    return config_yaml