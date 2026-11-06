"""Regression guard for literal credentials in the dev-container compose file.

CC-SEC-01-R1.1 replaced the tracked default database password and embedded
DATABASE_URL password with environment references. This test fails if a future
edit puts a literal back.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

COMPOSE = Path(__file__).resolve().parents[1] / ".devcontainer" / "docker-compose.yml"
ENV_REFERENCE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::[-?][^}]*)?\}$")


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_database_url_and_password_are_environment_references() -> None:
    data = _compose()
    assert ENV_REFERENCE.fullmatch(data["services"]["app"]["environment"]["DATABASE_URL"])
    assert ENV_REFERENCE.fullmatch(data["services"]["db"]["environment"]["POSTGRES_PASSWORD"])


def test_no_literal_secret_values_under_compose_environment() -> None:
    data = _compose()
    for service_name, service in data.get("services", {}).items():
        environment = service.get("environment", {})
        if not isinstance(environment, dict):
            continue
        for key, value in environment.items():
            upper = str(key).upper()
            if any(marker in upper for marker in ("PASSWORD", "TOKEN", "SECRET")):
                assert ENV_REFERENCE.fullmatch(str(value)), (
                    f"{service_name}.{key} must be an environment reference, not a literal"
                )
