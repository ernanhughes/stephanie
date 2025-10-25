# scripts/train_risk_from_df.py
from pathlib import Path
import pandas as pd
from stephanie.scoring.training.risk_trainer import RiskTrainer

df_path = Path("reports/risk_dataset.parquet")
df = pd.read_parquet(df_path)

trainer = RiskTrainer(
    cfg={
        "out_dir": "models/risk",
        "xgb": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "random_state": 42,
        },
        "calibration": "isotonic",
        "domain_sweep": {
            "enable": True,
            "domains": ["science","history","geography","tech","general"],
            "gate_lows":  [0.05,0.10,0.15,0.20,0.25],
            "gate_highs": [0.45,0.50,0.55,0.60,0.65],
        },
        "features": {
            # ensure these match what your featurizer emits / is in DataFrame
            "use": ["q_len","ctx_len","overlap_ratio","ner_count","num_tokens_est","coverage_gap","prior_max_energy_ema"]
        },
        "eval": {"val_frac": 0.2, "pr_auc": True},
    },
    memory=None, container=None, logger=print,
)

stats = trainer.train_dataframe(df)   # <— the method we implemented
print(stats)
