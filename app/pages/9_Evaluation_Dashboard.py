import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import load_css
from src import config

st.set_page_config(page_title="Evaluation Dashboard \u2014 CineSignal", page_icon="\U0001F4CA", layout="wide")
load_css()
st.markdown("## \U0001F4CA Evaluation Dashboard")
st.caption("All 5 recommenders evaluated on the SAME held-out, time-aware test split. Numbers are computed, "
           "never hard-coded.")

results_path = config.PROJECT_ROOT / "docs" / f"evaluation_results_{config.DATASET}.json"

if not results_path.exists():
    st.warning(f"No evaluation results yet for {config.DATASET}. Run: "
               f"`python scripts/train_and_evaluate.py`")
    st.stop()

results = json.loads(results_path.read_text())

k_values = config.EVAL_K_VALUES
metric_names = {"precision": "Precision@K", "recall": "Recall@K", "ndcg": "NDCG@K",
                 "hit_rate": "Hit Rate@K", "map": "MAP@K"}

rows = []
for model_name, metrics in results.items():
    for metric_key, per_k in metrics.items():
        for k, val in per_k.items():
            rows.append({"Model": model_name.replace("_", " ").title(), "Metric": metric_names[metric_key],
                         "K": int(k), "Value": val})
df = pd.DataFrame(rows)

metric_choice = st.selectbox("Metric", list(metric_names.values()), index=2)
plot_df = df[df.Metric == metric_choice]
fig = px.bar(plot_df, x="K", y="Value", color="Model", barmode="group",
             title=f"{metric_choice} by model", template="plotly_dark")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width='stretch')

st.markdown("### Full results table (K=10)")
table = df[df.K == 10].pivot(index="Model", columns="Metric", values="Value").round(4)
st.dataframe(table, width='stretch')

best_model = df[(df.Metric == "NDCG@K") & (df.K == 10)].sort_values("Value", ascending=False).iloc[0]
st.success(f"Best NDCG@10 on {config.DATASET}: **{best_model['Model']}** ({best_model['Value']:.4f})")
st.caption("A model winning here is reported because the evaluation says so \u2014 not asserted in advance "
           "(spec requirement: never claim hybrid/neural superiority without measured evidence).")
