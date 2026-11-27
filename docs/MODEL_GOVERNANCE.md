# Model governance

CareerCompass supports career exploration with a fixed role taxonomy. The model card documents weak groups, calibration, data sources and intended use. Role coverage is a model-quality check. A demographic fairness assessment would require suitable data and separately defined measures.

| Decision | Evidence and accountability |
|---|---|
| Detect and investigate | Check sample count and data origin, inspect top drifted features and confirm whether the shift is sustained |
| Compare models | Compare both on the same labelled data, enforce the real quality gate and the allowed real-score regression, then record the holdout check |
| Register | Write the version, algorithm, configuration, fit timestamp, Git revision, dataset and model hashes and run ID to isolated MLflow and JSON reports |
| Approve | An eligible local experiment needs `--promote --approved-by NAME`. The identifier records the operator's declaration |
| Roll back | Restore the previous experimental champion and record who did it and when |
| Release production | Obtain independent real-world validation and explicit approval. Registration, alias promotion and deployment each have their own records |

The lab's `career-role-classifier-phase3-synthetic-2026` registry is separate from the hosted production registry. The reference champion and candidate are stored in an isolated SQLite database under the report directory. Even an eligible synthetic candidate requires a separate operator action.

```powershell
uv run python -m scripts.retrain_phase3 --promote --approved-by "Demo reviewer"
uv run python -m scripts.retrain_phase3 --rollback --approved-by "Demo reviewer"
```

Run the first command after an **eligible** result. If the report rejects promotion, show the failed gate and use the isolated alias tests to demonstrate that a passing candidate can be promoted and rolled back. In production, an alert starts investigation. Available options include examining input validation, checking service health, obtaining labelled evaluation data, and reviewing release history.

The monitor stores normalized vocabulary identifiers and aggregate counts, with no resume contents in its observations. Limit dashboard access to reviewers and keep credential files out of Git. For production use, assign a responsible model owner, define a review and incident procedure, and agree on retention and access rules for aggregate evidence.
