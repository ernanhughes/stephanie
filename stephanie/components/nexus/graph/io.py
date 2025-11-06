from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Expect NexusNode/NexusEdge to be simple dataclasses or objects with attributes used below

def to_cytoscape_json(nodes: Dict[str, Any], edges: List[Any]) -> dict:
    # Cytoscape.js format: {"elements":{"nodes":[{"data":{...}}], "edges":[{"data":{...}}]}}
    cy_nodes = []
    for nid, n in nodes.items():
        data =
Come on for ****'s sake