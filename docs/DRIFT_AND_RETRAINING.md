# Drift and experimental retraining

The reference is the real 2025 training split scored by the frozen, selected model. The generator uses seed 42 and creates 7,200 training, 2,400 evaluation and 2,400 holdout rows. Training examples derive from real training postings. Evaluation and holdout derive from separate source groups in the real test file, after excluding overlap with training signatures. The generator checks schema, role and family compatibility, source groups and duplicate features. Each partition and assumptions manifest lives under `data/drift/`.

The generator adds role-appropriate AI, cloud and platform skills, changes role and family mix and modifies experience. The 2026 distributions are simulation settings. Printed values do not estimate the real 2026 labour market. Scores on derived postings estimate behaviour under this scenario and inherit the limits of rule-derived job labels. The evaluation window guides the decision. The holdout is read after that comparison, with veto checks on the decision and final reporting. Changing test conditions after inspecting holdout would require a new untouched holdout.

For a category with reference probability `r` and current probability `c`, its PSI contribution is `(c − r) × ln(c / r)`. Suppose a skill is present in 20% of reference postings and 40% of current postings. The two categories are present and absent. Their contributions are `0.2 × ln(2)` and `−0.2 × ln(0.75)`. The sum is about **0.196**, which is a warning under the defaults.

The detector applies a small floor to bin probabilities and renormalizes each distribution. It compares binary skill presence for common reference skills and aggregate AI/cloud skill groups. It uses fixed bins for experience, confidence and unknown-skill share, and a fixed class list for predicted roles. Overall PSI is the largest feature score. That makes an individual strong shift visible and can also raise sensitivity to noise. The configurable defaults are 0.10 for warning and 0.25 for critical, with 100 minimum accepted rows. Calibrate these heuristics on stable windows before using them for operational decisions.

```bash
uv run python -m scripts.generate_2026_drift
uv run python -m scripts.prepare_phase3
uv run python -m src.models.drift compare --reference .phase3/bundle/drift_reference.json --current data/drift/synthetic_2026_eval.parquet --output reports/phase3/offline_drift.json
uv run python -m scripts.retrain_phase3
```

## Real temporal drift (2019 vs 2025)

The synthetic set is a controlled scenario. `scripts/real_drift_2019.py` (`scripts/phase3_lab.sh real-drift`) applies the same frozen champion and the same PSI reference to real postings from 2019, with the 2025 test split as the control. Measured on 28 November 2026:

| Split | Rows | Top-3 | Max PSI | Status | Largest shifts |
|---|---:|---:|---:|---|---|
| 2025 test (control) | 13,688 | 80.2% | 0.007 | normal | none |
| 2019 reference | 20,183 | 70.2% | 0.262 | critical | predicted role, confidence, `html`, `javascript` |

From 2019 to 2025, `javascript`, `html` and `jquery` lost the most share of postings, and SAP, project management and sales gained the most. The worst-hit roles include Management Consultant, Mobile App Developer, UI/UX Designer and BI Developer. The report is `reports/phase3/real_drift/latest.json`, with a per-role CSV beside it, and the Grafana drift row shows it. This direction is backwards in time, but it is real-data evidence that the detector fires when the market differs from training and that accuracy drops with it.

The drift CLI returns **0** for a normal comparison, **2** for a warning or critical result, **3** for insufficient samples and **1** for an execution error. The retraining command writes the disposition to `reports/phase3/retraining/latest.json`. Its successful run can report a skip, rejection or eligible candidate. Its exit code 2 means the command failed to execute correctly.

The retraining pipeline checks drift before fitting. It reuses the calibrated ComplementNB architecture and compares the candidate against the frozen model under the same evaluation conditions. It also checks real 2025 performance, 2019 historical performance, model size, calibration, behaviour, robustness and latency. A local audit links the dataset fingerprints, DVC lock, configuration, code revision and MLflow run. An identical experiment in the same output directory reuses the fitted model and completed evidence.

GitHub scheduled runs perform the same synthetic experiment. The workflow must exist on the default branch before manual dispatch becomes available. Schedules run on the default branch. Use the local command while the change is on its feature branch. See [GitHub workflow triggers](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows) and [manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
