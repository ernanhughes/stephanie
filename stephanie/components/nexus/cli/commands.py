# stephanie/components/nexus/cli/commands.py
from __future__ import annotations

import json
from pathlib import Path

import click

from stephanie.components.nexus.graph.builder import build_graph_from_manifest
from stephanie.components.nexus.protocol.protocol import NexusProtocol


@click.group()
def cli():
    pass

@cli.command("build-graph-from-manifest")
@click.option("--manifest", type=str, required=True)
@click.option("--embed-key", type=str, default="global")
@click.option("--knn-k", type=int, default=12)
@click.option("--sim-threshold", type=float, default=0.35)
@click.option("--out", type=str, required=True, help="Output dir for nodes.jsonl and edges.jsonl")
def build_graph_from_manifest_cmd(manifest, embed_key, knn_k, sim_threshold, out):
    nodes, edges = build_graph_from_manifest(manifest, embed_key=embed_key, knn_k=knn_k, sim_threshold=sim_threshold)
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for n in nodes.values():
            f.write(json.dumps({
                "node_id": n.node_id,
                "scorable_id": n.scorable_id,
                "scorable_type": n.scorable_type,
                "metrics": n.metrics,
                "embed_global": n.embed_global.tolist() if n.embed_global is not None else None
            }) + "\n")
    with (out_dir / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps({
                "src": e.src, "dst": e.dst, "type": e.type, "weight": e.weight, "extras": e.extras
            }) + "\n")
    print(f"Wrote {len(nodes)} nodes and {len(edges)} edges to {out_dir}")

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
