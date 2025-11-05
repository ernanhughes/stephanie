# stephanie/components/nexus/cli/commands.py
from __future__ import annotations

import json

import click

from stephanie.components.nexus.protocol.protocol import NexusProtocol


@click.group()
def cli():
    pass

@cli.command()
@click.option("--from-vpms", type=str, required=True, help="Path to an NDJSON of (node_id, meta) rows")
def build_index(from_vpms: str):
    # load cfg + memory from your container in real app; stubbing here
    cfg = {
        "graph": {"knn_k": 32},
        "path": {"steps_max": 12, "weights": {}},
    }
    memory = None
    proto = NexusProtocol(cfg, memory)

    with open(from_vpms, "r", encoding="utf-8") as f:
        items = [tuple(json.loads(line)) for line in f]  # [(node_id, meta), ...]

    count = proto.svc.build_index_from_vpms(items)
    print(f"Indexed {count} VPM nodes")

@cli.command()
@click.option("--start", type=str, required=True)
def find_path(start: str):
    cfg = {
        "graph": {"knn_k": 32},
        "path": {"steps_max": 12, "weights": {}},
    }
    memory = None
    proto = NexusProtocol(cfg, memory)
    result = proto.svc.find_path(start)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    cli()
