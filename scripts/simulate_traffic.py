"""Send realistic traffic to the running app so the Insights tab has live data to show.

  uv run python scripts/simulate_traffic.py            # normal traffic (users resemble the training data)
  uv run python scripts/simulate_traffic.py --shift    # users suddenly ask about a narrow new set of skills -> drift alert
"""
import argparse
import random

import httpx
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--shift", action="store_true")
ap.add_argument("--url", default="http://127.0.0.1:8000")
a = ap.parse_args()

rng = random.Random(1)
test = pd.read_parquet("data/processed/test.parquet")
if a.shift:  # only people with AI / cloud skills show up
    test = test[test["role"].isin(["Machine Learning / AI Engineer", "DevOps / Cloud Engineer", "Data Engineer", "Cyber Security Analyst"])]
rows = test[test["skills"].map(len) >= 5].sample(a.n, random_state=1, replace=True)
ok = 0
with httpx.Client(base_url=a.url, timeout=20) as c:
    for skills, exp in zip(rows["skills"], rows["experience"]):
        r = c.post("/recommend", json={"skills": rng.sample(list(skills), 5), "experience": float(exp)})
        if r.status_code == 200:
            ok += 1
            if rng.random() < 0.6:
                c.post("/feedback", json={"request_id": r.json()["request_id"], "helpful": rng.random() < 0.8})
    print(f"sent {ok}/{a.n} requests")
    d = c.get("/drift").json()
    print({k: d[k] for k in ("n_live_requests", "status", "js_distance", "mean_confidence") if k in d})
