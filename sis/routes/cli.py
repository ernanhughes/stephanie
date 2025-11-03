from __future__ import annotations
import json, typer
from .service import NexusService

app = typer.Typer(add_completion=False)

@app.command()
def build_index(vpm_manifest: str, cfg_path: str = ""):
    # vpm_manifest: JSONL with records used by add_nodes_from_vpms()
    cfg = {}  # TODO: load Hydra or yaml
    svc = NexusService(cfg)
    items = []
    with open(vpm_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append((rec["node_id"], rec))
    n = svc.build_index_from_vpms(items)
    typer.echo(f"Indexed {n} nodes")

@app.command()
def find_path(start_node_id: str):
    cfg = {}
    svc = NexusService(cfg)
    out = svc.find_path(start_node_id)
    typer.echo(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()
