from pathlib import Path

import yaml

from stephanie.services.bus.hybrid_bus import HybridKnowledgeBus

ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_bus_defaults_to_zero_mq_and_loopback_endpoints():
    normalized = HybridKnowledgeBus({})._norm()
    assert normalized["backend"] == "zmq"
    assert normalized["frontend"] == "tcp://127.0.0.1:5555"
    assert normalized["backend_endpoint"] == "tcp://127.0.0.1:5556"


def test_hydra_bus_group_and_compose_stack_do_not_select_nats():
    assert not (ROOT / "config/bus/nats.yaml").exists()
    bus = yaml.safe_load((ROOT / "config/bus/enabled.yaml").read_text())
    assert bus["backend"] == "zmq"
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    assert {"bus": "zmq"} in config["defaults"]
    compose = (ROOT / "docker-compose.yml").read_text().lower()
    assert "nats:" not in compose
    assert "nats-py" not in (ROOT / "requirements_docker.txt").read_text().lower()


def test_legacy_nats_facade_is_explicitly_zero_mq_backed():
    text = (ROOT / "stephanie/services/bus/nats_client.py").read_text().lower()
    assert "legacy" in text and "zeromq" in text
    assert 'os.getenv("steph_bus_backend", "zmq")' in text
