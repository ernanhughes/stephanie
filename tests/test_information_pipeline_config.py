"""Regression guard for the Hydra pipeline/agent config contract.

`Supervisor._parse_pipeline_stages` resolves every stage through
``cfg.agents[name]``, which only exists if the stage has a
``config/agents/<name>.yaml`` file listed in ``defaults:``. A stage that is
declared in ``pipeline.stages`` but has no agent config fails at Supervisor
construction -- *after* the expensive model stack has loaded.

This test composes the real Hydra config and asserts the contract holds, so a
missing agent config is caught in CI instead of at run time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="module")
def information_cfg():
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra.compose(config_name="information")


def test_every_enabled_stage_has_an_agent_config(information_cfg):
    stages = information_cfg.pipeline.stages
    assert len(stages) > 0

    missing = []
    for stage in stages:
        if not stage.get("enabled"):
            continue
        name = stage.get("name")
        assert name, f"a pipeline stage has no name: {stage}"
        if name not in information_cfg.agents:
            missing.append(name)

    assert not missing, (
        "these enabled stages have no cfg.agents entry; add "
        "config/agents/<name>.yaml and list it under defaults: in "
        f"config/information.yaml -> {missing}"
    )


def test_writer_handoff_stage_is_wired(information_cfg):
    stages = {s.get("name"): s for s in information_cfg.pipeline.stages}
    assert "writer_handoff" in stages, "writer_handoff stage is not declared"

    stage = stages["writer_handoff"]
    assert stage["cls"] == (
        "stephanie.components.information.agents.writer_handoff.WriterHandoffAgent"
    )

    agent_cfg = information_cfg.agents.writer_handoff
    assert agent_cfg.enabled is True
    assert agent_cfg.profile == "fiction"
    assert int(agent_cfg.timeout_seconds) > 0


def test_paper_pipeline_cache_flags_live_in_the_agent_config(information_cfg):
    """The section-cache flags must be on the agent config (where the runner
    reads them), not on the pipeline stage entry."""
    agent_cfg = information_cfg.agents.paper_pipeline
    assert "force_rebuild_sections" in agent_cfg
    assert "sections_cache_enabled" in agent_cfg
