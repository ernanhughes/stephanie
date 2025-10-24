# scripts/debug_gap_run.py
from pathlib import Path
import json
import pandas as pd
import sys

def main(run_dir: str):
    rd = Path(run_dir)
    rows_path = rd / "raw" / "rows_for_df.parquet"
    prov_path = rd / "raw" / "row_provenance.json"

    print(f"[i] Run dir: {rd}")
    print(f"[i] rows_for_df: {rows_path.exists()}  | provenance: {prov_path.exists()}")

    if not rows_path.exists() or not prov_path.exists():
        print("[!] Missing required artifacts. Re-run scoring or check ScoringProcessor.save_rows_df / provenance writes.")
        return

    df = pd.read_parquet(rows_path)
    print("\n=== rows_for_df.parquet ===")
    print(df.head(3))
    print(df.columns.tolist())

    with open(prov_path, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov_df = pd.DataFrame(prov)
    print("\n=== row_provenance.json (first 3) ===")
    print(prov_df.head(3))
    print(prov_df.columns.tolist())

    # Column presence checks
    missing_rows_cols = [c for c in ["node_id"] if c not in df.columns]
    missing_prov_cols = [c for c in ["node_id","goal_text","output_text"] if c not in prov_df.columns]
    if missing_rows_cols:
        print(f"[!] rows_for_df missing columns: {missing_rows_cols}")
    if missing_prov_cols:
        print(f"[!] row_provenance missing columns: {missing_prov_cols}")

    # Node-id sanity
    if "node_id" in df.columns:
        print(f"[i] rows_for_df node_id non-null: {(~df['node_id'].isna()).sum()}/{len(df)}  unique={df['node_id'].nunique()}")
    if "node_id" in prov_df.columns:
        print(f"[i] provenance node_id non-null: {(~prov_df['node_id'].isna()).sum()}/{len(prov_df)}  unique={prov_df['node_id'].nunique()}")

    # Quick merge trial
    if "node_id" in df.columns and set(["node_id","goal_text","output_text"]).issubset(prov_df.columns):
        m = df.merge(prov_df[["node_id","goal_text","output_text"]], on="node_id", how="left")
        print("\n[i] Merge OK. Sample:")
        print(m.head(3)[["node_id","goal_text","output_text"]])
    else:
        print("\n[!] Merge not attempted due to missing columns as noted above.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_gap_run.py <RUN_DIR>")
        sys.exit(1)
    main(sys.argv[1])
