"""Create a hand-labelling sheet: ~300 real 2025 postings (stratified), with the rule label hidden in a separate column.

  uv run python scripts/make_gold_sample.py        -> data/gold/gold_to_label.csv
Fill `true_role` (exact role name from configs/role_rules.yaml, or NONE) and save as data/gold/gold_labelled.csv.
The evaluate stage then reports rule accuracy and model accuracy on it automatically.
"""
import pandas as pd

from src.data.common import ROOT

df = pd.read_parquet(ROOT / "data/processed/test.parquet")
n_per = max(1, 300 // df["role"].nunique())
s = df.groupby("role", group_keys=False).apply(lambda g: g.sample(min(len(g), n_per), random_state=7)).sample(frac=1, random_state=7)
out = pd.DataFrame({"title": s["title"], "skills": s["skills"].map(lambda x: ", ".join(x[:8])), "rule_role": s["role"], "true_role": ""})
(ROOT / "data/gold").mkdir(exist_ok=True)
out.to_csv(ROOT / "data/gold/gold_to_label.csv", index=False)
print(f"wrote {len(out)} rows to data/gold/gold_to_label.csv")
