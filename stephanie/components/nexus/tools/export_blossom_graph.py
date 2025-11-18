# stephanie/tools/export_blossom_graph.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import networkx as nx
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

try:
    import pygraphviz as pgv  # optional, best quality with dot layout
    HAS_PGV = True
except Exception:
    HAS_PGV = False

@dataclass
class BlossomGraphExporter:
    session_maker: sessionmaker

    def load_graph(self, *, blossom_id: int) -> nx.DiGraph:
        with self.session_maker() as s:
            nodes = s.execute(
                text("""
                SELECT node_id, label, full_text, extra
                FROM blossom_graph_nodes_v
                WHERE blossom_id = :bid
                """),
                {"bid": blossom_id},
            ).mappings().all()

            edges = s.execute(
                text("""
                SELECT src_id, dst_id, relation, score, rationale
                FROM blossom_graph_edges_v
                WHERE blossom_id = :bid
                """),
                {"bid": blossom_id},
            ).mappings().all()

        G = nx.DiGraph()
        for n in nodes:
            G.add_node(
                int(n["node_id"]),
                label=n["label"],
                full_text=n["full_text"],
                **({"extra": dict(n["extra"])} if n["extra"] is not None else {}),
            )
        for e in edges:
            G.add_edge(
                int(e["src_id"]),
                int(e["dst_id"]),
                relation=e["relation"],
                score=(float(e["score"]) if e["score"] is not None else None),
                rationale=e["rationale"],
            )
        return G

    def to_dot(self, G: nx.DiGraph) -> str:
        # Produce a DOT string (works even without pygraphviz)
        lines = ["digraph blossom {", '  graph [rankdir=LR, splines=true, overlap=false];', '  node [shape=box, style="rounded,filled", fillcolor=white];']
        for n, data in G.nodes(data=True):
            label = data.get("label", str(n)).replace('"', r'\"')
            lines.append(f'  {n} [label="{label}"];')
        for u, v, data in G.edges(data=True):
            rel = data.get("relation", "edge")
            lines.append(f'  {u} -> {v} [label="{rel}"];')
        lines.append("}")
        return "\n".join(lines)

    def write(self, *, blossom_id: int, out_dir: str, fmt: str = "svg") -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        G = self.load_graph(blossom_id=blossom_id)

        dot_path = os.path.join(out_dir, f"blossom-{blossom_id}.dot")
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(self.to_dot(G))

        outputs = {"dot": dot_path}

        # Also save a lightweight JSON for your SIS viewer
        import json
        json_path = os.path.join(out_dir, f"blossom-{blossom_id}.json")
        payload = {
            "nodes": [{"id": int(n), **data} for n, data in G.nodes(data=True)],
            "edges": [{"src": int(u), "dst": int(v), **data} for u, v, data in G.edges(data=True)],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        outputs["json"] = json_path

        # Render image
        if fmt.lower() in {"svg", "png"}:
            if HAS_PGV:
                A = nx.nx_agraph.to_agraph(G)
                A.graph_attr.update(rankdir="LR", splines="true", overlap="false")
                A.node_attr.update(shape="box", style="rounded,filled", fillcolor="white")
                img_path = os.path.join(out_dir, f"blossom-{blossom_id}.{fmt.lower()}")
                A.draw(img_path, prog="dot")
                outputs[fmt.lower()] = img_path
            else:
                # Fallback: basic spring layout PNG
                import matplotlib.pyplot as plt
                pos = nx.spring_layout(G, seed=42, k=0.4)
                plt.figure(figsize=(14, 8))
                nx.draw_networkx_nodes(G, pos)
                nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n].get("label", str(n)) for n in G.nodes})
                nx.draw_networkx_edges(G, pos, arrows=True, width=0.8, alpha=0.6)
                img_path = os.path.join(out_dir, f"blossom-{blossom_id}.png")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(img_path, dpi=150)
                plt.close()
                outputs["png"] = img_path

        return outputs
