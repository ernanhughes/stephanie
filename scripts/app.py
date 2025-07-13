# app.py
import streamlit as st
import pandas as pd

DEFAULT_DIMENSIONS = [
    "alignment",
    "implementability",
    "clarity",
    "relevance",
    "novelty",
]


st.set_page_config(page_title="Scoring Policy Dashboard", layout="wide")

# Load data
df = pd.read_csv("scoring_results.csv")

# Filters
st.sidebar.header("Filters")
selected_dim = st.sidebar.selectbox("Select Dimension", df.columns[4::3])

# Summary stats
st.header("Policy Performance Summary")
summary = df.describe()
st.write(summary)

# Uncertainty heatmap
st.header("Uncertainty Heatmap")
uncertainty_cols = [f"{dim}_uncertainty" for dim in DEFAULT_DIMENSIONS]
st.dataframe(df[["document_id"] + uncertainty_cols].style.background_gradient(cmap='Reds'))

# Energy vs. Score correlation
st.header("Energy vs. Final Score")
for dim in DEFAULT_DIMENSIONS:
    st.subheader(dim)
    col1, col2 = st.columns(2)
    col1.scatter(df[f"{dim}_energy"], df[f"{dim}_score"])
    col1.set_xlabel("EBT Energy")
    col1.set_ylabel("Final Score")
    col2.barh(df[f"{dim}_uncertainty"].quantile([0.25, 0.5, 0.75]))