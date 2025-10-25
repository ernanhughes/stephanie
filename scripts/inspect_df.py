import pandas as pd

df = pd.read_parquet("reports/risk_dataset.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.head(5))
print(df.describe(include='all'))
