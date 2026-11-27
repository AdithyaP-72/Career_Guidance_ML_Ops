# Model Comparison Report

**Selection metric:** `top3_5skills` (top-3 accuracy on 5-skill test inputs — closest to real usage)
**Champion:** `complement_nb`

## Baselines

| Model | Verdict | Top-3 (5 skills) | Top-1 (5 skills) | Top-5 (5 skills) | Macro-F1 (5 skills) | Family Top-1 | Ref-2019 Top-3 | Top-3 (full) | Fit (s) | Size (MB) | Latency (ms/1k) | Drift drop (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **complement_nb** | 🏆 Champion | 80.0% | 56.0% | 86.6% | 0.519 | 72.0% | 69.9% | 83.6% | 1.4 | 1.80 | 11.5 | 12.6 |
| **logreg** | ✅ Strong | 79.4% | 56.5% | 85.8% | 0.553 | 72.1% | 66.8% | 84.6% | 73.1 | 4.80 | 8.5 | 15.9 |
| **lightgbm** | ✅ Strong | 78.2% | 56.8% | 84.5% | 0.559 | 72.0% | 67.1% | 84.8% | 321.7 | 8.15 | 379.6 | 14.2 |
| **mlp** | 🟡 Acceptable | 76.4% | 53.8% | 83.3% | 0.524 | 70.5% | 61.3% | 81.3% | 3002.9 | 36.67 | 20.7 | 19.8 |
| **sgd** | 🟡 Acceptable | 76.2% | 56.8% | 81.1% | 0.549 | 71.8% | 62.1% | 83.2% | 4.7 | 2.89 | 9.3 | 18.6 |
| **random_forest** | 🟡 Acceptable | 73.8% | 49.3% | 80.7% | 0.440 | 66.8% | 62.1% | 79.3% | 19.9 | 111.96 | 82.3 | 15.8 |
| **profile** | 🔴 Weak | 66.3% | 43.5% | 75.5% | 0.455 | 62.9% | 56.0% | 73.9% | 0.8 | 0.31 | 11.8 | 15.6 |
| **dummy** | 🔴 Weak | 19.8% | 8.5% | 25.7% | 0.002 | 17.5% | 17.5% | 19.8% | 0.9 | 0.07 | 5.7 | 11.3 |

> **Drift drop (%)**: how much top-3 accuracy drops when evaluating on the 2019 reference set.
> A large drop means the model is less reliable for older job-posting patterns.
>
> Generated automatically by `src.monitoring.model_comparison`.
