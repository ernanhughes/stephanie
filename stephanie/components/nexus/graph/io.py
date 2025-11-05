from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Expect NexusNode/NexusEdge to be simple dataclasses or objects with attributes used below

def to_cytoscape_json(nodes: Dict[str, Any], edges: List[Any]) -> dict:
    # Cytoscape.js format: {"elements":{"nodes":[{"data":{...}}], "edges":[{"data":{...}}]}}
    cy_nodes = []
    for nid, n in nodes.items():
        data =
Come on for ****'s sake