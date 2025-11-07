import json, sys, pathlib as p
root = p.Path("runs/nexus_vpm")
rid = sys.argv[1]
b = json.load(open(root/f"{rid}-baseline/run_metrics.json"))
t = json.load(open(root/f"{rid}-targeted/run_metrics.json"))

def g(d,k,sk=None):
    return (d[k][sk] if sk else d[k])

pairs = [
  ("goal_alignment.mean", g(b,"goal_alignment","mean"), g(t,"goal_alignment","mean")),
  ("goal_alignment.p90",  g(b,"goal_alignment","p90"),  g(t,"goal_alignment","p90")),
  ("mutual_knn_frac",     g(b,"mutual_knn_frac"),       g(t,"mutual_knn_frac")),
  ("clustering_coeff",    g(b,"clustering_coeff"),      g(t,"clustering_coeff")),
  ("spatial.mean_edge_len", g(b,"spatial","mean_edge_len"), g(t,"spatial","mean_edge_len")),
]
for name, vb, vt in pairs:
    imp = (vb - vt)/max(1e-9, abs(vb)) if "spatial" in name else (vt - vb)/max(1e-9, abs(vb))
    print(f"{name:24s}  base={vb:.4f}  targ={vt:.4f}  Δ%={(imp*100):+.1f}%")
